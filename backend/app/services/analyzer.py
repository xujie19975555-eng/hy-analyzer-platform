from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from app.models.schemas import PerformanceSummary, TraderStats


def parse_superx_stats(address: str, data: dict) -> TraderStats:
    """Parse SuperX API response to TraderStats model"""
    return TraderStats(
        address=address,
        roe_all_time=data.get("roeAllTime"),
        roe_7d=data.get("roe7d"),
        roe_30d=data.get("roe30d"),
        roe_90d=data.get("roe90d"),
        pnl_all_time=data.get("pnlAllTime"),
        pnl_7d=data.get("pnl7d"),
        pnl_30d=data.get("pnl30d"),
        pnl_90d=data.get("pnl90d"),
        win_rate=data.get("winRate"),
        profit_factor=data.get("profitFactorAll"),
        max_drawdown_pct=data.get("maxDrawdownPercentAll"),
        total_trades=data.get("tradeCount"),
    )


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


def _is_buy(side: Any) -> bool:
    s = str(side).strip().lower()
    return s in {"b", "buy", "long"}


def _compute_avg_holding_time_seconds(
    fills_sorted: list[Mapping[str, Any]],
    close_time_start_ms: int,
) -> float | None:
    """
    FIFO lot matching per-coin to compute average holding time.
    Only counts closes where close time >= close_time_start_ms.
    Weighted by entry notional.
    """
    long_lots: dict[str, deque[tuple[int, float, float]]] = defaultdict(deque)
    short_lots: dict[str, deque[tuple[int, float, float]]] = defaultdict(deque)

    weighted_seconds = 0.0
    weighted_notional = 0.0

    for f in fills_sorted:
        t = _to_int_ms(f.get("time"))
        if t is None:
            continue
        coin = str(f.get("coin") or "UNKNOWN")
        px = _to_float(f.get("px"), default=0.0)
        sz = abs(_to_float(f.get("sz"), default=0.0))
        if px <= 0 or sz <= 0:
            continue

        if _is_buy(f.get("side")):
            remaining = sz
            q = short_lots[coin]
            while remaining > 0 and q:
                entry_ms, entry_px, entry_sz = q[0]
                matched = min(remaining, entry_sz)
                entry_notional = matched * entry_px
                if t >= close_time_start_ms:
                    weighted_seconds += ((t - entry_ms) / 1000.0) * entry_notional
                    weighted_notional += entry_notional
                entry_sz -= matched
                remaining -= matched
                if entry_sz <= 1e-12:
                    q.popleft()
                else:
                    q[0] = (entry_ms, entry_px, entry_sz)
            if remaining > 0:
                long_lots[coin].append((t, px, remaining))
        else:
            remaining = sz
            q = long_lots[coin]
            while remaining > 0 and q:
                entry_ms, entry_px, entry_sz = q[0]
                matched = min(remaining, entry_sz)
                entry_notional = matched * entry_px
                if t >= close_time_start_ms:
                    weighted_seconds += ((t - entry_ms) / 1000.0) * entry_notional
                    weighted_notional += entry_notional
                entry_sz -= matched
                remaining -= matched
                if entry_sz <= 1e-12:
                    q.popleft()
                else:
                    q[0] = (entry_ms, entry_px, entry_sz)
            if remaining > 0:
                short_lots[coin].append((t, px, remaining))

    if weighted_notional <= 0:
        return None
    return float(weighted_seconds / weighted_notional)


def _compute_simple_metrics(
    fills_in_range: list[Mapping[str, Any]],
    period_days: float,
) -> dict[str, float | None]:
    """Compute long_short_ratio, avg_trade_size, trade_frequency"""
    if not fills_in_range or period_days <= 0:
        return {
            "long_short_ratio": None,
            "avg_trade_size": None,
            "trade_frequency": None,
        }

    long_notional = 0.0
    short_notional = 0.0
    total_notional = 0.0
    trade_count = 0

    for f in fills_in_range:
        px = _to_float(f.get("px"), default=0.0)
        sz = _to_float(f.get("sz"), default=0.0)
        notional = abs(px * sz)
        if notional <= 0:
            continue
        total_notional += notional
        trade_count += 1
        if _is_buy(f.get("side")):
            long_notional += notional
        else:
            short_notional += notional

    if trade_count == 0:
        return {
            "long_short_ratio": None,
            "avg_trade_size": None,
            "trade_frequency": None,
        }

    long_short_ratio = None
    if short_notional > 0:
        long_short_ratio = float(long_notional / short_notional)

    avg_trade_size = float(total_notional / trade_count)
    trade_frequency = float(trade_count / period_days)

    return {
        "long_short_ratio": long_short_ratio,
        "avg_trade_size": avg_trade_size,
        "trade_frequency": trade_frequency,
    }


def compute_trade_metrics_timeframes(
    fills: list[Mapping[str, Any]],
    now_ms: int | None = None,
) -> dict[str, float | None]:
    """
    Compute fill-derived metrics for 7d/30d/90d/all_time:
    - long_short_ratio_*
    - avg_trade_size_*
    - avg_holding_time_* (seconds)
    - trade_frequency_* (trades per day)
    """
    if now_ms is None:
        now_ms = int(datetime.utcnow().timestamp() * 1000)

    fills_sorted = sorted(fills, key=lambda f: _to_int_ms(f.get("time")) or 0)

    def _in_range(f: Mapping[str, Any], start_ms: int, end_ms: int) -> bool:
        t = _to_int_ms(f.get("time"))
        return t is not None and start_ms <= t <= end_ms

    timeframes: list[tuple[str, int, float]] = [
        ("7d", now_ms - 7 * 86_400_000, 7.0),
        ("30d", now_ms - 30 * 86_400_000, 30.0),
        ("90d", now_ms - 90 * 86_400_000, 90.0),
    ]

    first_ms = _to_int_ms(fills_sorted[0].get("time")) if fills_sorted else None
    last_ms = _to_int_ms(fills_sorted[-1].get("time")) if fills_sorted else None
    if first_ms is not None and last_ms is not None and last_ms > first_ms:
        all_days = max(1.0, (last_ms - first_ms) / 86_400_000.0)
    else:
        all_days = 1.0

    out: dict[str, float | None] = {}

    all_simple = _compute_simple_metrics(fills_sorted, period_days=all_days)
    out["long_short_ratio_all_time"] = all_simple["long_short_ratio"]
    out["avg_trade_size_all_time"] = all_simple["avg_trade_size"]
    out["trade_frequency_all_time"] = all_simple["trade_frequency"]
    out["avg_holding_time_all_time"] = _compute_avg_holding_time_seconds(
        fills_sorted, close_time_start_ms=0
    )

    for label, start_ms, days in timeframes:
        subset = [f for f in fills_sorted if _in_range(f, start_ms, now_ms)]
        simple = _compute_simple_metrics(subset, period_days=days)
        out[f"long_short_ratio_{label}"] = simple["long_short_ratio"]
        out[f"avg_trade_size_{label}"] = simple["avg_trade_size"]
        out[f"trade_frequency_{label}"] = simple["trade_frequency"]
        out[f"avg_holding_time_{label}"] = _compute_avg_holding_time_seconds(
            fills_sorted, close_time_start_ms=start_ms
        )

    return out


def analyze_trades(fills: list) -> PerformanceSummary:
    """Analyze trade fills and return performance summary"""
    if not fills:
        raise ValueError("No trade data provided")

    df = pd.DataFrame(fills)

    # Convert data types
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["closedPnl"] = pd.to_numeric(df["closedPnl"], errors="coerce").fillna(0)
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0)

    # Sort by time
    df = df.sort_values("time")

    # Calculate cumulative PnL
    df["cumulative_pnl"] = df["closedPnl"].cumsum()

    # Basic stats
    total_pnl = df["closedPnl"].sum()
    total_fees = df["fee"].sum()
    net_pnl = total_pnl - total_fees

    # Max drawdown
    cumulative = df["cumulative_pnl"]
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = float(drawdown.min())
    max_drawdown_pct = (max_drawdown / rolling_max.max() * 100) if rolling_max.max() != 0 else 0

    # Win rate
    winning_trades = int((df["closedPnl"] > 0).sum())
    losing_trades = int((df["closedPnl"] < 0).sum())
    total_trades = len(df)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return PerformanceSummary(
        address=df["user"].iloc[0] if "user" in df.columns else "",
        total_trades=total_trades,
        total_pnl=round(total_pnl, 2),
        total_fees=round(total_fees, 2),
        net_pnl=round(net_pnl, 2),
        max_drawdown=round(max_drawdown, 2),
        max_drawdown_pct=round(float(max_drawdown_pct), 2),
        win_rate=round(win_rate, 2),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        first_trade=df["time"].min(),
        last_trade=df["time"].max(),
    )
