from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.models.schemas import BacktestResult, DrawdownPeriod, Period

MS_PER_DAY = 86_400_000


def _now_ms() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def period_to_time_range_ms(period: Period, now_ms: int | None = None) -> tuple[int, int]:
    """Convert period string to (start_ms, end_ms) tuple"""
    end_ms = _now_ms() if now_ms is None else int(now_ms)
    if period == "all":
        HYPERLIQUID_GENESIS_MS = 1672531200000  # 2023-01-01 00:00:00 UTC
        return HYPERLIQUID_GENESIS_MS, end_ms

    days = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "180d": 180,
    }[period]
    start_ms = end_ms - (days * MS_PER_DAY)
    return start_ms, end_ms


def _to_int_ms(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


@dataclass(frozen=True)
class EquityPoint:
    time_ms: int
    equity: float


def _compute_drawdowns(points: Sequence[EquityPoint]) -> tuple[float, float, list[DrawdownPeriod]]:
    """Compute max drawdown and drawdown periods from equity curve"""
    if not points:
        return 0.0, 0.0, []

    peak_equity = points[0].equity
    peak_time = points[0].time_ms
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    periods: list[DrawdownPeriod] = []
    in_dd = False
    dd_peak_equity = peak_equity
    dd_peak_time = peak_time
    trough_equity = peak_equity
    trough_time = peak_time

    for p in points[1:]:
        if p.equity >= peak_equity:
            if in_dd:
                drawdown = trough_equity - dd_peak_equity
                drawdown_pct = (drawdown / dd_peak_equity * 100.0) if dd_peak_equity > 0 else 0.0
                periods.append(
                    DrawdownPeriod(
                        start_time=_ms_to_dt(dd_peak_time),
                        trough_time=_ms_to_dt(trough_time),
                        end_time=_ms_to_dt(p.time_ms),
                        drawdown=float(drawdown),
                        drawdown_pct=float(drawdown_pct),
                    )
                )
                in_dd = False
            peak_equity = p.equity
            peak_time = p.time_ms
            continue

        dd = p.equity - peak_equity
        if dd < max_drawdown:
            max_drawdown = float(dd)
            max_drawdown_pct = float((dd / peak_equity * 100.0) if peak_equity > 0 else 0.0)

        if not in_dd:
            in_dd = True
            dd_peak_equity = peak_equity
            dd_peak_time = peak_time
            trough_equity = p.equity
            trough_time = p.time_ms
        else:
            if p.equity < trough_equity:
                trough_equity = p.equity
                trough_time = p.time_ms

    if in_dd:
        last = points[-1]
        drawdown = trough_equity - dd_peak_equity
        drawdown_pct = (drawdown / dd_peak_equity * 100.0) if dd_peak_equity > 0 else 0.0
        periods.append(
            DrawdownPeriod(
                start_time=_ms_to_dt(dd_peak_time),
                trough_time=_ms_to_dt(trough_time),
                end_time=_ms_to_dt(last.time_ms),
                drawdown=float(drawdown),
                drawdown_pct=float(drawdown_pct),
            )
        )

    return float(max_drawdown), float(max_drawdown_pct), periods


def run_backtest(
    *,
    address: str,
    period: Period,
    capital: float,
    account_value: float,
    fills: Sequence[Mapping[str, Any]],
    min_position_usdt: float = 11.0,
) -> BacktestResult:
    """
    Run backtest by scaling trader's historical trades proportionally.

    The scale ensures every trade meets HyperLiquid's minimum position size (11 USDT).
    Small capital can still follow every trade.
    """
    if capital <= 0:
        raise ValueError("capital must be > 0")
    if account_value <= 0:
        raise ValueError("account_value must be > 0")

    sorted_fills = sorted(
        (dict(f) for f in fills),
        key=lambda f: _to_int_ms(f.get("time")) or 0,
    )

    # Find the smallest trade notional that's above the min threshold
    min_notional = float("inf")
    for f in sorted_fills:
        px = _to_float(f.get("px"), default=0.0)
        sz = _to_float(f.get("sz"), default=0.0)
        notional = abs(px * sz)
        # Only consider trades that won't be skipped
        if notional >= min_position_usdt:
            min_notional = min(min_notional, notional)

    # Calculate scale: ensure smallest trade >= min_position_usdt after scaling
    base_scale = float(capital) / float(account_value)

    if min_notional != float("inf") and min_notional > 0:
        min_required_scale = float(min_position_usdt) / min_notional
        scale = max(base_scale, min_required_scale)
    else:
        scale = base_scale

    equity = float(capital)
    points: list[EquityPoint] = []

    first_time = _to_int_ms(sorted_fills[0].get("time")) if sorted_fills else None
    if first_time is None:
        first_time = _now_ms()
    points.append(EquityPoint(time_ms=int(first_time), equity=equity))

    trade_count = 0
    skipped_count = 0

    for f in sorted_fills:
        t = _to_int_ms(f.get("time"))
        if t is None:
            continue

        px = _to_float(f.get("px"), default=0.0)
        sz = _to_float(f.get("sz"), default=0.0)
        notional = abs(px * sz)

        if notional < min_position_usdt:
            skipped_count += 1
            continue

        closed_pnl = _to_float(
            f.get("closedPnl"), default=_to_float(f.get("closed_pnl"), default=0.0)
        )
        fee = _to_float(f.get("fee"), default=0.0)

        equity += (closed_pnl - fee) * scale
        points.append(EquityPoint(time_ms=int(t), equity=float(equity)))
        trade_count += 1

    simulated_pnl = float(equity - float(capital))
    simulated_roe = float((simulated_pnl / float(capital)) * 100.0)

    max_drawdown, max_drawdown_pct, drawdown_periods = _compute_drawdowns(points)

    return BacktestResult(
        address=address,
        period=period,
        simulated_pnl=round(simulated_pnl, 2),
        simulated_roe=round(simulated_roe, 2),
        max_drawdown=round(max_drawdown, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        drawdown_periods=drawdown_periods,
        trade_count=int(trade_count),
        skipped_trade_count=int(skipped_count),
        scale=round(float(scale), 6),
        baseline_account_value=round(float(account_value), 2),
    )
