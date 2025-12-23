from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class WalletAddress(BaseModel):
    address: str = Field(..., description="Ethereum wallet address")

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.lower().strip()
        if not re.match(r"^0x[a-f0-9]{40}$", v):
            raise ValueError("Invalid Ethereum address format")
        return v


class TraderStats(BaseModel):
    address: str
    roe_all_time: Optional[float] = None
    roe_7d: Optional[float] = None
    roe_30d: Optional[float] = None
    roe_90d: Optional[float] = None
    pnl_all_time: Optional[float] = None
    pnl_7d: Optional[float] = None
    pnl_30d: Optional[float] = None
    pnl_90d: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    total_trades: Optional[int] = None


class PnLDataPoint(BaseModel):
    timestamp: datetime
    pnl: float
    account_value: float


class TraderPnLHistory(BaseModel):
    address: str
    period: str  # "7d", "30d", "90d", "all"
    data_points: list[PnLDataPoint]


class TradeRecord(BaseModel):
    time: datetime
    coin: str
    side: str  # "B" or "A" (buy/sell)
    px: float
    sz: float
    closed_pnl: float
    fee: float
    hash: Optional[str] = None


class TraderTradesResponse(BaseModel):
    address: str
    total_count: int
    trades: list[TradeRecord]


class PerformanceSummary(BaseModel):
    address: str
    total_trades: int
    total_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    winning_trades: int
    losing_trades: int
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None


class APIError(BaseModel):
    error: str
    detail: Optional[str] = None
