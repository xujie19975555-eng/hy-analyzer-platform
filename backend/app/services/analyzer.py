import pandas as pd
from typing import Optional
from datetime import datetime

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
        total_trades=data.get("tradeCount")
    )


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
        last_trade=df["time"].max()
    )
