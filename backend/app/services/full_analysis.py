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
    TraderHistoryItem,
)
from app.services.ai_scoring import get_ai_scoring_service
from app.services.analyzer import compute_trade_metrics_timeframes, parse_superx_stats
from app.services.backtest import period_to_time_range_ms, run_backtest
from app.services.hyperliquid import get_hyperliquid_service, get_superx_service

logger = logging.getLogger(__name__)

_current_analysis: dict[str, AnalysisStatus] = {}
_analysis_lock = asyncio.Lock()


async def get_analysis_status(address: str) -> AnalysisStatus | None:
    addr = address.lower()
    return _current_analysis.get(addr)


async def is_analysis_running() -> bool:
    async with _analysis_lock:
        return any(s.status == "analyzing" for s in _current_analysis.values())


def _update_status(
    address: str, status: str, progress: int, step: str | None = None, error: str | None = None
):
    _current_analysis[address.lower()] = AnalysisStatus(
        address=address.lower(), status=status, progress=progress, current_step=step, error=error
    )


async def run_full_analysis(address: str, capital: float = 10000.0) -> FullAnalysisResult:
    addr = address.lower()

    async with _analysis_lock:
        if any(s.status == "analyzing" for s in _current_analysis.values()):
            raise ValueError("Another analysis is already running. Please wait.")
        _update_status(addr, "analyzing", 0, "初始化分析")

    hl = get_hyperliquid_service()
    superx = get_superx_service()

    result = FullAnalysisResult(address=addr, analyzed_at=datetime.now(tz=UTC))

    try:
        # Phase 1: Parallel fetch - account value, SuperX stats, 90-day fills, and all-time fills
        _update_status(addr, "analyzing", 10, "并行获取数据")

        start_90d, end_90d = period_to_time_range_ms("90d")
        start_all, end_all = period_to_time_range_ms("all")

        async def fetch_account_value():
            return await hl.get_account_value(addr)

        async def fetch_superx_stats():
            try:
                return await superx.get_trader_stats(addr)
            except Exception as e:
                logger.warning("Failed to fetch SuperX stats: %s", e)
                return None

        async def fetch_fills_90d():
            return await hl.get_user_fills_windowed(
                addr, start_time=start_90d, end_time=end_90d, aggregate=False
            )

        async def fetch_fills_all():
            return await hl.get_user_fills_windowed(
                addr, start_time=start_all, end_time=end_all, aggregate=False
            )

        # Run all four in parallel
        account_value, superx_data, fills_90d, fills_all = await asyncio.gather(
            fetch_account_value(), fetch_superx_stats(), fetch_fills_90d(), fetch_fills_all()
        )

        result.account_value = account_value

        _update_status(addr, "analyzing", 40, "处理SuperX数据")
        if superx_data:
            base_stats = parse_superx_stats(addr, superx_data)
            # Keep win_rate and profit_factor from SuperX (these are calculated correctly)
            result.win_rate = base_stats.win_rate
            # Sanitize profit_factor: cap at reasonable range to avoid overflow values
            pf = base_stats.profit_factor
            if pf is not None:
                if pf > 100:
                    pf = 100.0  # Cap at 100 (extremely profitable)
                elif pf < 0.01:
                    pf = 0.01  # Floor at 0.01 (extremely unprofitable)
            result.profit_factor = pf
            result.total_trades = base_stats.total_trades
            # Note: ROE/PnL/max_drawdown will be calculated from fills below

        _update_status(addr, "analyzing", 50, "计算交易风格指标")
        metrics = compute_trade_metrics_timeframes(fills_90d, now_ms=end_90d)
        result.long_short_ratio = metrics.get("long_short_ratio_30d")
        result.avg_trade_size = metrics.get("avg_trade_size_30d")
        result.avg_holding_time = metrics.get("avg_holding_time_30d")
        result.trade_frequency = metrics.get("trade_frequency_30d")

        # Phase 2: Calculate actual PnL/ROE from fills (not scaled)
        # This is more accurate than SuperX because it excludes deposits/withdrawals
        def calc_actual_pnl(fills: list) -> tuple[float, int]:
            """Calculate actual PnL from fills (sum of closedPnl - fee)"""
            total_pnl = 0.0
            count = 0
            for f in fills:
                pnl = float(f.get("closedPnl", 0) or 0)
                fee = float(f.get("fee", 0) or 0)
                total_pnl += pnl - fee
                if pnl != 0:
                    count += 1
            return total_pnl, count

        # Calculate actual PnL for each period
        start_7d, end_7d = period_to_time_range_ms("7d")
        start_30d, end_30d = period_to_time_range_ms("30d")

        fills_7d = [f for f in fills_90d if start_7d <= int(f.get("time", 0)) <= end_7d]
        fills_30d = [f for f in fills_90d if start_30d <= int(f.get("time", 0)) <= end_30d]

        pnl_7d, _ = calc_actual_pnl(fills_7d)
        pnl_30d, _ = calc_actual_pnl(fills_30d)
        pnl_90d, _ = calc_actual_pnl(fills_90d)
        pnl_all, trade_count = calc_actual_pnl(fills_all)

        result.pnl_7d = round(pnl_7d, 2)
        result.pnl_30d = round(pnl_30d, 2)
        result.pnl_90d = round(pnl_90d, 2)
        result.pnl_all_time = round(pnl_all, 2)

        # Calculate ROE based on account value
        if account_value and account_value > 0:
            result.roe_7d = round(pnl_7d / account_value * 100, 2)
            result.roe_30d = round(pnl_30d / account_value * 100, 2)
            result.roe_90d = round(pnl_90d / account_value * 100, 2)
            result.roe_all_time = round(pnl_all / account_value * 100, 2)

        if trade_count > 0:
            result.total_trades = trade_count

        # Phase 3: Run backtests for drawdown calculation
        if account_value and account_value > 0:
            _update_status(addr, "analyzing", 60, "运行回测")

            # fills_7d, fills_30d already calculated above

            # Run backtests (CPU operations, negligible time)
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
            except Exception as e:
                logger.warning("7d backtest failed: %s", e)

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
            except Exception as e:
                logger.warning("30d backtest failed: %s", e)

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
            except Exception as e:
                logger.warning("90d backtest failed: %s", e)

            # All-time backtest (using fills_all)
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
                    # Use our own calculated max_drawdown_pct (more accurate than SuperX)
                    result.max_drawdown_pct = bt_all.max_drawdown_pct
            except Exception as e:
                logger.warning("all-time backtest failed: %s", e)

        _update_status(addr, "analyzing", 80, "AI双模型评估中")
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
            "status": "completed",
        }

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

        # Apply filters
        if min_score is not None:
            query = query.filter(TraderAnalysis.ai_score >= min_score)
        if recommendation:
            query = query.filter(TraderAnalysis.ai_recommendation == recommendation)

        # Determine sort column
        sort_columns = {
            "ai_score": TraderAnalysis.ai_score,
            "analyzed_at": TraderAnalysis.analyzed_at,
            "roe_30d": TraderAnalysis.roe_30d,
            "account_value": TraderAnalysis.account_value,
        }
        sort_col = sort_columns.get(sort_by, TraderAnalysis.ai_score)

        # Apply sorting with nulls last
        if order == "desc":
            # For DESC, nulls should be at the bottom
            query = query.order_by(sort_col.desc().nullslast())
        else:
            # For ASC, nulls should be at the bottom too
            query = query.order_by(sort_col.asc().nullslast())

        # Secondary sort by analyzed_at for ties
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
            status=r.status or "completed",
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
            result.ai_evaluation = AIEvaluation(
                score=r.ai_score,
                recommendation=r.ai_recommendation or "neutral",
                risk_level=r.ai_risk_level or "medium",
                reasoning=r.ai_reasoning or "",
                trading_tags=r.ai_trading_tags.split(",") if r.ai_trading_tags else [],
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

        # Make sure both datetimes are timezone-aware
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
