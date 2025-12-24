from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from app.config import get_settings
from app.models.database import SessionLocal, TraderAnalysis
from app.models.schemas import (
    AIEvaluation,
    AnalysisStatus,
    BacktestSummary,
    FullAnalysisResult,
    HighFrequencyWarning,
    TraderHistoryItem,
)
from app.services.ai_scoring import get_ai_scoring_service
from app.services.analyzer import compute_trade_metrics_timeframes, parse_superx_stats
from app.services.backtest import period_to_time_range_ms, run_backtest
from app.services.hyperliquid import get_hyperliquid_service, get_superx_service

# Threshold for high-frequency trader detection
HIGH_FREQUENCY_THRESHOLD = 8000  # 90d fills that likely exceed 10K API limit

logger = logging.getLogger(__name__)

_current_analysis: dict[str, AnalysisStatus] = {}
_analysis_lock = asyncio.Lock()


async def get_analysis_status(address: str) -> AnalysisStatus | None:
    addr = address.lower()
    return _current_analysis.get(addr)


async def is_analysis_running() -> bool:
    async with _analysis_lock:
        return any(s.status == "analyzing" for s in _current_analysis.values())


async def reset_analysis_status():
    """Force reset all analysis status (use when stuck)"""
    async with _analysis_lock:
        for addr in list(_current_analysis.keys()):
            if _current_analysis[addr].status == "analyzing":
                _current_analysis[addr] = AnalysisStatus(
                    address=addr,
                    status="failed",
                    progress=0,
                    current_step=None,
                    error="Analysis reset by user",
                    high_frequency_warning=None,
                )
        return True


def _update_status(
    address: str,
    status: str,
    progress: int,
    step: str | None = None,
    error: str | None = None,
    high_freq_warning: HighFrequencyWarning | None = None,
):
    _current_analysis[address.lower()] = AnalysisStatus(
        address=address.lower(),
        status=status,
        progress=progress,
        current_step=step,
        error=error,
        high_frequency_warning=high_freq_warning,
    )


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert value to float with fallback"""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def calculate_win_rate_from_fills(fills: list) -> float | None:
    """Calculate win rate from fills data as fallback when SuperX fails"""
    if not fills:
        return None

    winning_trades = 0
    total_trades = 0

    for f in fills:
        closed_pnl = _safe_float(f.get("closedPnl"))
        if closed_pnl != 0:  # Only count trades with closed PnL
            total_trades += 1
            if closed_pnl > 0:
                winning_trades += 1

    if total_trades == 0:
        return None

    return round((winning_trades / total_trades) * 100, 2)


def calculate_profit_factor_from_fills(fills: list) -> float | None:
    """Calculate profit factor from fills data as fallback when SuperX fails"""
    if not fills:
        return None

    gross_profit = 0.0
    gross_loss = 0.0

    for f in fills:
        closed_pnl = _safe_float(f.get("closedPnl"))
        fee = _safe_float(f.get("fee"))
        net_pnl = closed_pnl - fee

        if net_pnl > 0:
            gross_profit += net_pnl
        elif net_pnl < 0:
            gross_loss += abs(net_pnl)

    if gross_loss == 0:
        if gross_profit > 0:
            return 100.0  # Cap at 100 (near-infinite profit factor)
        return None

    pf = gross_profit / gross_loss
    # Cap at reasonable range
    if pf > 100:
        pf = 100.0
    elif pf < 0.01:
        pf = 0.01

    return round(pf, 2)


def calculate_trading_days_from_fills(fills: list) -> tuple[int | None, datetime | None]:
    """
    Calculate trading days (survival days) from fills data.
    Returns (trading_days, first_trade_date)
    """
    if not fills:
        return None, None

    timestamps = []
    for f in fills:
        t = f.get("time")
        if t is not None:
            try:
                timestamps.append(int(t))
            except (TypeError, ValueError):
                continue

    if not timestamps:
        return None, None

    first_t = min(timestamps)
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    trading_days = max(1, int((now_ms - first_t) / 86400000))

    # Convert first trade timestamp to datetime
    first_trade_date = datetime.fromtimestamp(first_t / 1000, tz=UTC)

    return trading_days, first_trade_date


async def run_full_analysis(
    address: str, capital: float = 10000.0, use_full_history: bool = False
) -> FullAnalysisResult:
    addr = address.lower()

    async with _analysis_lock:
        if any(s.status == "analyzing" for s in _current_analysis.values()):
            raise ValueError("Another analysis is already running. Please wait.")
        _update_status(addr, "analyzing", 0, "初始化分析")

    hl = get_hyperliquid_service()
    superx = get_superx_service()

    result = FullAnalysisResult(address=addr, analyzed_at=datetime.now(tz=UTC))

    try:
        # Phase 1: Sequential fetch to avoid 429 rate limiting
        _update_status(addr, "analyzing", 5, "获取账户数据")

        start_90d, end_90d = period_to_time_range_ms("90d")

        # Fetch account value first (lightweight call)
        account_value = await hl.get_account_value(addr)
        _update_status(addr, "analyzing", 10, "账户余额获取完成")

        await asyncio.sleep(0.5)

        # Fetch SuperX stats (separate API, can run after HL call)
        superx_data = None
        superx_failed = False
        try:
            _update_status(addr, "analyzing", 12, "获取SuperX数据")
            superx_data = await superx.get_trader_stats(addr)
        except Exception as e:
            logger.warning("Failed to fetch SuperX stats: %s", e)
            superx_failed = True

        await asyncio.sleep(0.5)

        _update_status(addr, "analyzing", 15, "获取90天交易记录")

        # Fetch 90d fills (heaviest call)
        fills_90d = await hl.get_user_fills_windowed(
            addr, start_time=start_90d, end_time=end_90d, aggregate=False
        )

        _update_status(addr, "analyzing", 25, f"获取到 {len(fills_90d)} 笔交易记录")

        result.account_value = account_value

        # Check if whale account based on cumulative volume (>$1B)
        is_whale_account = await hl.is_high_frequency_account(addr)
        deposit_pnl_data = None

        if is_whale_account:
            _update_status(addr, "analyzing", 28, "高频账户：使用存取款方式计算PnL")
            deposit_pnl_data = await hl.calculate_pnl_from_deposits(addr)
            logger.info("Whale account detected: %s, using deposit-based PnL: $%.2f", addr[:10], deposit_pnl_data["total_pnl"])

        # Check if high-frequency trader (likely to exceed API limit)
        fills_90d_count = len(fills_90d)
        is_high_frequency = fills_90d_count >= HIGH_FREQUENCY_THRESHOLD or is_whale_account
        data_limited = False

        if is_high_frequency and not use_full_history:
            logger.info(
                "High-frequency trader detected: %s has %d fills in 90d, using 90d data",
                addr[:10], fills_90d_count
            )
            _update_status(
                addr, "analyzing", 30,
                f"高频交易员检测：90天内{fills_90d_count}笔交易，使用90天数据"
            )
            fills_all = fills_90d
            data_limited = True
        else:
            _update_status(addr, "analyzing", 30, "获取全历史数据（可能需要5-10分钟）")

            def s3_progress(completed: int, total: int):
                if total > 0:
                    pct = 30 + int((completed / total) * 20)
                    _update_status(
                        addr, "analyzing", pct,
                        f"从S3获取历史数据 ({completed}/{total} 小时, {int(completed/total*100)}%)"
                    )

            start_all, end_all = period_to_time_range_ms("all")
            fills_all = []
            try:
                fills_all = await asyncio.wait_for(
                    hl.get_user_fills_windowed(
                        addr, start_time=start_all, end_time=end_all, aggregate=False,
                        s3_progress_callback=s3_progress if use_full_history else None,
                    ),
                    timeout=600.0 if use_full_history else 60.0
                )
            except TimeoutError:
                logger.warning("All-time fills fetch timed out for %s, using 90d data", addr)
                fills_all = fills_90d
                data_limited = True

        # Calculate trading_days (survival days) - ALWAYS calculate this
        trading_days, first_trade_date = calculate_trading_days_from_fills(
            fills_all if fills_all else fills_90d
        )
        result.trading_days = trading_days
        result.first_trade_date = first_trade_date

        if data_limited:
            _update_status(addr, "analyzing", 50, "使用90天数据（全历史数据受限）")
            result.data_limited = True
            result.data_coverage_days = 90
        else:
            _update_status(addr, "analyzing", 50, "处理历史数据")
            if fills_all:
                first_t = min(int(f.get("time", 0)) for f in fills_all)
                last_t = max(int(f.get("time", 0)) for f in fills_all)
                result.data_coverage_days = int((last_t - first_t) / 86400000)

        # Parse SuperX stats OR calculate from fills as fallback
        _update_status(addr, "analyzing", 52, "计算胜率和盈亏比")

        if superx_data and not superx_failed:
            base_stats = parse_superx_stats(addr, superx_data)
            result.win_rate = base_stats.win_rate
            pf = base_stats.profit_factor
            if pf is not None:
                if pf > 100:
                    pf = 100.0
                elif pf < 0.01:
                    pf = 0.01
            result.profit_factor = pf
            result.total_trades = base_stats.total_trades
        else:
            # Fallback: calculate from fills data
            logger.info("SuperX failed, calculating win_rate/profit_factor from fills")
            result.win_rate = calculate_win_rate_from_fills(fills_90d)
            result.profit_factor = calculate_profit_factor_from_fills(fills_90d)
            # Count trades with non-zero PnL
            trade_count = sum(1 for f in fills_90d if _safe_float(f.get("closedPnl")) != 0)
            result.total_trades = trade_count if trade_count > 0 else len(fills_90d)

        _update_status(addr, "analyzing", 55, "计算交易风格指标")
        metrics = compute_trade_metrics_timeframes(fills_90d, now_ms=end_90d)
        result.long_short_ratio = metrics.get("long_short_ratio_30d")
        result.avg_trade_size = metrics.get("avg_trade_size_30d")
        result.avg_holding_time = metrics.get("avg_holding_time_30d")
        result.trade_frequency = metrics.get("trade_frequency_30d")

        # Phase 2: Calculate actual PnL/ROE from fills
        def calc_actual_pnl(fills: list) -> tuple[float, int]:
            total_pnl = 0.0
            count = 0
            for f in fills:
                pnl = _safe_float(f.get("closedPnl"))
                fee = _safe_float(f.get("fee"))
                total_pnl += pnl - fee
                if pnl != 0:
                    count += 1
            return total_pnl, count

        start_7d, end_7d = period_to_time_range_ms("7d")
        start_30d, end_30d = period_to_time_range_ms("30d")

        fills_7d = [f for f in fills_90d if start_7d <= int(f.get("time", 0)) <= end_7d]
        fills_30d = [f for f in fills_90d if start_30d <= int(f.get("time", 0)) <= end_30d]

        pnl_7d, _ = calc_actual_pnl(fills_7d)
        pnl_30d, _ = calc_actual_pnl(fills_30d)
        pnl_90d, trade_count_90d = calc_actual_pnl(fills_90d)

        result.pnl_7d = round(pnl_7d, 2)
        result.pnl_30d = round(pnl_30d, 2)
        result.pnl_90d = round(pnl_90d, 2)

        if account_value and account_value > 0:
            result.roe_7d = round(pnl_7d / account_value * 100, 2)
            result.roe_30d = round(pnl_30d / account_value * 100, 2)
            result.roe_90d = round(pnl_90d / account_value * 100, 2)

        # All-time metrics
        if data_limited:
            if deposit_pnl_data:
                result.pnl_all_time = round(deposit_pnl_data["total_pnl"], 2)
                if account_value and account_value > 0:
                    result.roe_all_time = round(deposit_pnl_data["total_pnl"] / account_value * 100, 2)
            else:
                result.pnl_all_time = None
                result.roe_all_time = None
            result.total_trades = result.total_trades or (trade_count_90d if trade_count_90d > 0 else None)
        else:
            pnl_all, trade_count_all = calc_actual_pnl(fills_all)
            result.pnl_all_time = round(pnl_all, 2)
            if account_value and account_value > 0:
                result.roe_all_time = round(pnl_all / account_value * 100, 2)
            if trade_count_all > 0:
                result.total_trades = trade_count_all

        _update_status(addr, "analyzing", 60, "计算PnL/ROE完成")

        # Phase 3: Run backtests for drawdown calculation
        if account_value and account_value > 0:
            _update_status(addr, "analyzing", 65, "运行7天回测")

            try:
                if fills_7d:
                    bt_7d = run_backtest(
                        address=addr,
                        period="7d",
                        capital=capital,
                        account_value=account_value,
                        fills=fills_7d,
                    )
                    result.backtest_7d = BacktestSummary(
                        period="7d",
                        pnl=bt_7d.simulated_pnl,
                        roe=bt_7d.simulated_roe,
                        max_drawdown_pct=bt_7d.max_drawdown_pct,
                        trade_count=bt_7d.trade_count,
                    )
                else:
                    # No fills in 7d, create empty backtest
                    result.backtest_7d = BacktestSummary(
                        period="7d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                    )
            except Exception as e:
                logger.warning("7d backtest failed: %s", e)
                result.backtest_7d = BacktestSummary(
                    period="7d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                )

            _update_status(addr, "analyzing", 70, "运行30天回测")
            try:
                if fills_30d:
                    bt_30d = run_backtest(
                        address=addr,
                        period="30d",
                        capital=capital,
                        account_value=account_value,
                        fills=fills_30d,
                    )
                    result.backtest_30d = BacktestSummary(
                        period="30d",
                        pnl=bt_30d.simulated_pnl,
                        roe=bt_30d.simulated_roe,
                        max_drawdown_pct=bt_30d.max_drawdown_pct,
                        trade_count=bt_30d.trade_count,
                    )
                else:
                    result.backtest_30d = BacktestSummary(
                        period="30d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                    )
            except Exception as e:
                logger.warning("30d backtest failed: %s", e)
                result.backtest_30d = BacktestSummary(
                    period="30d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                )

            _update_status(addr, "analyzing", 75, "运行90天回测")
            try:
                if fills_90d:
                    bt_90d = run_backtest(
                        address=addr,
                        period="90d",
                        capital=capital,
                        account_value=account_value,
                        fills=fills_90d,
                    )
                    result.backtest_90d = BacktestSummary(
                        period="90d",
                        pnl=bt_90d.simulated_pnl,
                        roe=bt_90d.simulated_roe,
                        max_drawdown_pct=bt_90d.max_drawdown_pct,
                        trade_count=bt_90d.trade_count,
                    )
                else:
                    result.backtest_90d = BacktestSummary(
                        period="90d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                    )
            except Exception as e:
                logger.warning("90d backtest failed: %s", e)
                result.backtest_90d = BacktestSummary(
                    period="90d", pnl=0, roe=0, max_drawdown_pct=0, trade_count=0
                )

            _update_status(addr, "analyzing", 80, "运行全部时间回测")
            if data_limited:
                if result.backtest_90d:
                    result.max_drawdown_pct = result.backtest_90d.max_drawdown_pct
                result.backtest_all_time = None
            else:
                try:
                    if fills_all:
                        bt_all = run_backtest(
                            address=addr,
                            period="all",
                            capital=capital,
                            account_value=account_value,
                            fills=fills_all,
                        )
                        result.backtest_all_time = BacktestSummary(
                            period="all",
                            pnl=bt_all.simulated_pnl,
                            roe=bt_all.simulated_roe,
                            max_drawdown_pct=bt_all.max_drawdown_pct,
                            trade_count=bt_all.trade_count,
                        )
                        result.max_drawdown_pct = bt_all.max_drawdown_pct
                except Exception as e:
                    logger.warning("all-time backtest failed: %s", e)
                    if result.backtest_90d:
                        result.max_drawdown_pct = result.backtest_90d.max_drawdown_pct

        # Ensure max_drawdown_pct is set
        if result.max_drawdown_pct is None and result.backtest_90d:
            result.max_drawdown_pct = result.backtest_90d.max_drawdown_pct

        _update_status(addr, "analyzing", 85, "AI双模型评估中")
        ai_service = get_ai_scoring_service()
        result.ai_evaluation = await ai_service.evaluate(result)

        _update_status(addr, "analyzing", 95, "保存到数据库")
        _save_to_database(result)

        _update_status(addr, "completed", 100, "分析完成")
        return result

    except Exception as e:
        logger.exception("Full analysis failed for %s", addr)
        _update_status(addr, "failed", 0, error=str(e))
        raise


def _save_to_database(result: FullAnalysisResult):
    db = SessionLocal()
    try:
        existing = db.query(TraderAnalysis).filter(TraderAnalysis.address == result.address).first()

        data = {
            "address": result.address,
            "account_value": result.account_value,
            "analyzed_at": result.analyzed_at,
            "roe_7d": result.roe_7d,
            "roe_30d": result.roe_30d,
            "roe_90d": result.roe_90d,
            "roe_all_time": result.roe_all_time,
            "pnl_7d": result.pnl_7d,
            "pnl_30d": result.pnl_30d,
            "pnl_90d": result.pnl_90d,
            "pnl_all_time": result.pnl_all_time,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "max_drawdown_pct": result.max_drawdown_pct,
            "total_trades": result.total_trades,
            "long_short_ratio": result.long_short_ratio,
            "avg_trade_size": result.avg_trade_size,
            "avg_holding_time": result.avg_holding_time,
            "trade_frequency": result.trade_frequency,
            "trading_days": result.trading_days,
            "status": "completed",
            "data_limited": result.data_limited,
            "data_coverage_days": result.data_coverage_days,
        }

        # Save first_trade_date if available
        if result.first_trade_date:
            data["first_trade_date"] = result.first_trade_date

        if result.backtest_7d:
            data["backtest_pnl_7d"] = result.backtest_7d.pnl
            data["backtest_roe_7d"] = result.backtest_7d.roe
            data["backtest_max_dd_7d"] = result.backtest_7d.max_drawdown_pct
            data["backtest_trade_count_7d"] = result.backtest_7d.trade_count

        if result.backtest_30d:
            data["backtest_pnl_30d"] = result.backtest_30d.pnl
            data["backtest_roe_30d"] = result.backtest_30d.roe
            data["backtest_max_dd_30d"] = result.backtest_30d.max_drawdown_pct
            data["backtest_trade_count_30d"] = result.backtest_30d.trade_count

        if result.backtest_90d:
            data["backtest_pnl_90d"] = result.backtest_90d.pnl
            data["backtest_roe_90d"] = result.backtest_90d.roe
            data["backtest_max_dd_90d"] = result.backtest_90d.max_drawdown_pct
            data["backtest_trade_count_90d"] = result.backtest_90d.trade_count

        if result.backtest_all_time:
            data["backtest_pnl_all_time"] = result.backtest_all_time.pnl
            data["backtest_roe_all_time"] = result.backtest_all_time.roe
            data["backtest_max_dd_all_time"] = result.backtest_all_time.max_drawdown_pct
            data["backtest_trade_count_all_time"] = result.backtest_all_time.trade_count

        if result.ai_evaluation:
            data["ai_score"] = result.ai_evaluation.score
            data["ai_recommendation"] = result.ai_evaluation.recommendation
            data["ai_risk_level"] = result.ai_evaluation.risk_level
            data["ai_reasoning"] = result.ai_evaluation.reasoning
            data["ai_claude_score"] = result.ai_evaluation.claude_score
            data["ai_codex_score"] = result.ai_evaluation.codex_score
            data["ai_models_used"] = ",".join(result.ai_evaluation.models_used)
            data["ai_trading_tags"] = ",".join(result.ai_evaluation.trading_tags)
            if result.ai_evaluation.score_breakdown:
                data["ai_score_breakdown"] = [
                    {"item": b.item if hasattr(b, 'item') else b.get("item", ""),
                     "points": b.points if hasattr(b, 'points') else b.get("points", 0),
                     "type": b.type if hasattr(b, 'type') else b.get("type", "positive")}
                    for b in result.ai_evaluation.score_breakdown
                ]

        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            db.add(TraderAnalysis(**data))

        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to save analysis to database")
        raise
    finally:
        db.close()


def get_analysis_history(
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "ai_score",
    order: str = "desc",
    min_score: float = None,
    recommendation: str = None,
) -> list[TraderHistoryItem]:
    db = SessionLocal()
    try:
        query = db.query(TraderAnalysis)

        if min_score is not None:
            query = query.filter(TraderAnalysis.ai_score >= min_score)
        if recommendation:
            query = query.filter(TraderAnalysis.ai_recommendation == recommendation)

        sort_columns = {
            "ai_score": TraderAnalysis.ai_score,
            "analyzed_at": TraderAnalysis.analyzed_at,
            "roe_30d": TraderAnalysis.roe_30d,
            "account_value": TraderAnalysis.account_value,
        }
        sort_col = sort_columns.get(sort_by, TraderAnalysis.ai_score)

        if order == "desc":
            query = query.order_by(sort_col.desc().nullslast())
        else:
            query = query.order_by(sort_col.asc().nullslast())

        if sort_by != "analyzed_at":
            query = query.order_by(TraderAnalysis.analyzed_at.desc())

        records = query.offset(offset).limit(limit).all()

        return [
            TraderHistoryItem(
                address=r.address,
                analyzed_at=r.analyzed_at,
                account_value=r.account_value,
                roe_30d=r.roe_30d,
                ai_score=r.ai_score,
                ai_recommendation=r.ai_recommendation,
                trading_tags=r.ai_trading_tags.split(",") if r.ai_trading_tags else [],
            )
            for r in records
        ]
    finally:
        db.close()


def get_cached_analysis(address: str) -> FullAnalysisResult | None:
    db = SessionLocal()
    try:
        r = db.query(TraderAnalysis).filter(TraderAnalysis.address == address.lower()).first()

        if not r:
            return None

        result = FullAnalysisResult(
            address=r.address,
            analyzed_at=r.analyzed_at,
            account_value=r.account_value,
            roe_7d=r.roe_7d,
            roe_30d=r.roe_30d,
            roe_90d=r.roe_90d,
            roe_all_time=r.roe_all_time,
            pnl_7d=r.pnl_7d,
            pnl_30d=r.pnl_30d,
            pnl_90d=r.pnl_90d,
            pnl_all_time=r.pnl_all_time,
            win_rate=r.win_rate,
            profit_factor=r.profit_factor,
            max_drawdown_pct=r.max_drawdown_pct,
            total_trades=r.total_trades,
            long_short_ratio=r.long_short_ratio,
            avg_trade_size=r.avg_trade_size,
            avg_holding_time=r.avg_holding_time,
            trade_frequency=r.trade_frequency,
            trading_days=getattr(r, "trading_days", None),
            first_trade_date=getattr(r, "first_trade_date", None),
            status=r.status or "completed",
            data_limited=getattr(r, "data_limited", False) or False,
            data_coverage_days=getattr(r, "data_coverage_days", None),
        )

        if r.backtest_roe_7d is not None:
            result.backtest_7d = BacktestSummary(
                period="7d",
                pnl=r.backtest_pnl_7d or 0,
                roe=r.backtest_roe_7d,
                max_drawdown_pct=r.backtest_max_dd_7d or 0,
                trade_count=r.backtest_trade_count_7d or 0,
            )

        if r.backtest_roe_30d is not None:
            result.backtest_30d = BacktestSummary(
                period="30d",
                pnl=r.backtest_pnl_30d or 0,
                roe=r.backtest_roe_30d,
                max_drawdown_pct=r.backtest_max_dd_30d or 0,
                trade_count=r.backtest_trade_count_30d or 0,
            )

        if r.backtest_roe_90d is not None:
            result.backtest_90d = BacktestSummary(
                period="90d",
                pnl=r.backtest_pnl_90d or 0,
                roe=r.backtest_roe_90d,
                max_drawdown_pct=r.backtest_max_dd_90d or 0,
                trade_count=r.backtest_trade_count_90d or 0,
            )

        if r.backtest_roe_all_time is not None:
            result.backtest_all_time = BacktestSummary(
                period="all",
                pnl=r.backtest_pnl_all_time or 0,
                roe=r.backtest_roe_all_time,
                max_drawdown_pct=r.backtest_max_dd_all_time or 0,
                trade_count=r.backtest_trade_count_all_time or 0,
            )

        if r.ai_score is not None:
            score_breakdown_data = getattr(r, "ai_score_breakdown", None) or []
            result.ai_evaluation = AIEvaluation(
                score=r.ai_score,
                recommendation=r.ai_recommendation or "neutral",
                risk_level=r.ai_risk_level or "medium",
                reasoning=r.ai_reasoning or "",
                trading_tags=r.ai_trading_tags.split(",") if r.ai_trading_tags else [],
                score_breakdown=score_breakdown_data,
                claude_score=getattr(r, "ai_claude_score", None),
                codex_score=getattr(r, "ai_codex_score", None),
                models_used=r.ai_models_used.split(",") if getattr(r, "ai_models_used", None) else [],
            )

        return result
    finally:
        db.close()


def is_cache_valid(address: str) -> bool:
    """Check if cached analysis is still valid (within cache_valid_hours)"""
    settings = get_settings()
    db = SessionLocal()
    try:
        r = db.query(TraderAnalysis).filter(TraderAnalysis.address == address.lower()).first()

        if not r or not r.analyzed_at:
            return False

        analyzed_at = r.analyzed_at
        if analyzed_at.tzinfo is None:
            analyzed_at = analyzed_at.replace(tzinfo=UTC)

        now = datetime.now(tz=UTC)
        valid_until = analyzed_at + timedelta(hours=settings.cache_valid_hours)
        return now < valid_until
    finally:
        db.close()


def delete_analysis(address: str) -> bool:
    """Delete analysis record for an address"""
    db = SessionLocal()
    try:
        result = db.query(TraderAnalysis).filter(TraderAnalysis.address == address.lower()).delete()
        db.commit()
        return result > 0
    except Exception as e:
        db.rollback()
        logger.error("Failed to delete analysis for %s: %s", address, e)
        return False
    finally:
        db.close()
