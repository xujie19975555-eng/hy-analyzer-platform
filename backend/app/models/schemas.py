import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Period = Literal["7d", "30d", "90d", "180d", "all"]


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
    roe_all_time: float | None = None
    roe_7d: float | None = None
    roe_30d: float | None = None
    roe_90d: float | None = None
    pnl_all_time: float | None = None
    pnl_7d: float | None = None
    pnl_30d: float | None = None
    pnl_90d: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    max_drawdown_pct: float | None = None
    total_trades: int | None = None
    account_value: float | None = Field(
        default=None, description="Current account value (USDT) from Hyperliquid"
    )
    long_short_ratio_all_time: float | None = None
    long_short_ratio_7d: float | None = None
    long_short_ratio_30d: float | None = None
    long_short_ratio_90d: float | None = None
    avg_trade_size_all_time: float | None = Field(
        default=None, description="Average trade notional size (USDT)"
    )
    avg_trade_size_7d: float | None = None
    avg_trade_size_30d: float | None = None
    avg_trade_size_90d: float | None = None
    avg_holding_time_all_time: float | None = Field(
        default=None, description="Average holding time in seconds"
    )
    avg_holding_time_7d: float | None = None
    avg_holding_time_30d: float | None = None
    avg_holding_time_90d: float | None = None
    trade_frequency_all_time: float | None = Field(default=None, description="Trades per day")
    trade_frequency_7d: float | None = None
    trade_frequency_30d: float | None = None
    trade_frequency_90d: float | None = None


class PnLDataPoint(BaseModel):
    timestamp: datetime
    pnl: float
    account_value: float


class TraderPnLHistory(BaseModel):
    address: str
    period: str
    data_points: list[PnLDataPoint]


class TradeRecord(BaseModel):
    time: datetime
    coin: str
    side: str
    px: float
    sz: float
    closed_pnl: float
    fee: float
    hash: str | None = None


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
    first_trade: datetime | None = None
    last_trade: datetime | None = None


class APIError(BaseModel):
    error: str
    detail: str | None = None


class BacktestRequest(BaseModel):
    capital: float = Field(
        ..., gt=0, le=100_000_000, description="Starting capital (USDT), max 100M"
    )
    period: Period = Field(..., description="Backtest period: 7d/30d/90d/180d/all")


class DrawdownPeriod(BaseModel):
    start_time: datetime
    trough_time: datetime
    end_time: datetime
    drawdown: float
    drawdown_pct: float


class BacktestResult(BaseModel):
    address: str
    period: Period
    simulated_pnl: float
    simulated_roe: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_periods: list[DrawdownPeriod]
    trade_count: int
    skipped_trade_count: int = Field(default=0, description="Trades skipped due to min notional")
    scale: float = Field(..., description="capital / account_value")
    baseline_account_value: float = Field(..., description="Account value used for scaling (USDT)")


class BacktestSummary(BaseModel):
    """Backtest results for a single period"""

    period: Period
    pnl: float
    roe: float
    max_drawdown_pct: float
    trade_count: int


class ScoreBreakdownItem(BaseModel):
    """Individual scoring factor"""

    item: str = Field(..., description="Description of the scoring factor in Chinese")
    points: float = Field(..., description="Points added (positive) or deducted (negative)")
    type: Literal["positive", "negative"] = Field(..., description="Whether this is a bonus or penalty")


class AIEvaluation(BaseModel):
    """AI evaluation for copy-trading recommendation"""

    score: float = Field(..., ge=0, le=100, description="Overall score 0-100")
    recommendation: Literal["strong_follow", "follow", "neutral", "avoid", "strong_avoid"]
    risk_level: Literal["low", "medium", "high", "extreme"]
    reasoning: str
    trading_tags: list[str] = Field(
        default_factory=list, description="Trading style tags like 高频, 中线, 小资金"
    )
    score_breakdown: list[ScoreBreakdownItem | dict] = Field(
        default_factory=list, description="Detailed scoring factors (加分/扣分项)"
    )
    claude_score: float | None = Field(None, ge=0, le=100, description="Claude model score")
    codex_score: float | None = Field(None, ge=0, le=100, description="Codex/GPT model score")
    models_used: list[str] = Field(default_factory=list, description="Models used for evaluation")
    data_coverage_warning: str | None = Field(
        None, description="Warning if full history data is incomplete"
    )


class FullAnalysisRequest(BaseModel):
    """Request for full trader analysis"""

    capital: float = Field(
        default=10000, gt=0, le=100_000_000, description="Capital for backtest (USDT)"
    )
    force_refresh: bool = Field(
        default=False, description="Force re-analysis even if cached data exists"
    )
    use_full_history: bool = Field(
        default=False, description="For high-frequency traders, fetch full S3 history (slow)"
    )


class HighFrequencyWarning(BaseModel):
    """Warning when trader has too many fills for quick analysis"""

    address: str
    fills_90d: int = Field(..., description="Number of fills in 90 days")
    estimated_all_time: int = Field(..., description="Estimated total fills")
    requires_s3: bool = Field(..., description="Whether S3 fetch is required")
    estimated_time_seconds: int = Field(..., description="Estimated time for full fetch")
    message: str


class FullAnalysisResult(BaseModel):
    """Complete trader analysis result"""

    address: str
    analyzed_at: datetime
    account_value: float | None = None

    # Stats
    roe_7d: float | None = None
    roe_30d: float | None = None
    roe_90d: float | None = None
    roe_all_time: float | None = None
    pnl_7d: float | None = None
    pnl_30d: float | None = None
    pnl_90d: float | None = None
    pnl_all_time: float | None = None

    # Risk metrics
    win_rate: float | None = None
    profit_factor: float | None = None
    max_drawdown_pct: float | None = None
    total_trades: int | None = None

    # Trading style
    long_short_ratio: float | None = None
    avg_trade_size: float | None = None
    avg_holding_time: float | None = None
    trade_frequency: float | None = None

    # Survival days - NEW: Added first_trade_date for better display
    trading_days: int | None = Field(
        None, description="Days since first trade (trader survival days / 存活天数)"
    )
    first_trade_date: datetime | None = Field(
        None, description="Date of the first trade (首笔交易日期)"
    )

    # Backtest results
    backtest_7d: BacktestSummary | None = None
    backtest_30d: BacktestSummary | None = None
    backtest_90d: BacktestSummary | None = None
    backtest_all_time: BacktestSummary | None = None

    # AI evaluation
    ai_evaluation: AIEvaluation | None = None

    # Status and data coverage
    status: str = "completed"
    data_limited: bool = Field(
        default=False, description="True if using 90d data due to API limit"
    )
    data_coverage_days: int | None = Field(
        None, description="Actual number of days of data used"
    )

    # Computed property for survival time display (like SuperX "1 year")
    @property
    def survival_time_display(self) -> str | None:
        """Return human-readable survival time like '1 year', '6 months', '30 days'"""
        if self.trading_days is None:
            return None
        days = self.trading_days
        if days >= 365:
            years = days / 365
            if years >= 2:
                return f"{int(years)} years"
            return "1 year" if years >= 1 else f"{int(days/30)} months"
        elif days >= 30:
            months = days / 30
            return f"{int(months)} months" if months >= 2 else "1 month"
        else:
            return f"{days} days"


class AnalysisStatus(BaseModel):
    """Status of an ongoing analysis"""

    address: str
    status: Literal["queued", "analyzing", "completed", "failed", "needs_confirmation"]
    progress: int = Field(default=0, ge=0, le=100)
    current_step: str | None = None
    error: str | None = None
    high_frequency_warning: "HighFrequencyWarning | None" = None


class TraderHistoryItem(BaseModel):
    """Item in trader analysis history"""

    address: str
    analyzed_at: datetime
    account_value: float | None = None
    roe_30d: float | None = None
    ai_score: float | None = None
    ai_recommendation: str | None = None
    trading_tags: list[str] = Field(default_factory=list, description="Trading style tags")
    trading_days: int | None = Field(None, description="Days since first trade")


# === Watchlist Schemas ===


class WatchlistAddRequest(BaseModel):
    """Request to add address to watchlist"""

    address: str = Field(..., description="Ethereum wallet address")
    alias: str | None = Field(None, max_length=50, description="User-defined nickname")
    notes: str | None = Field(None, description="User notes about this trader")
    priority: int = Field(
        default=0, ge=0, le=100, description="Priority level (higher = more important)"
    )
    auto_update: bool = Field(default=True, description="Auto-update fills data in background")

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        v = v.lower().strip()
        if not re.match(r"^0x[a-f0-9]{40}$", v):
            raise ValueError("Invalid Ethereum address format")
        return v


class WatchlistUpdateRequest(BaseModel):
    """Request to update watchlist entry"""

    alias: str | None = Field(None, max_length=50)
    notes: str | None = None
    priority: int | None = Field(None, ge=0, le=100)
    auto_update: bool | None = None


class DataCoverage(BaseModel):
    """Data coverage information"""

    start: int | None = Field(None, description="Earliest fill timestamp (ms)")
    end: int | None = Field(None, description="Latest fill timestamp (ms)")
    start_date: str | None = None
    end_date: str | None = None


class WatchlistAnalysis(BaseModel):
    """Cached analysis for watchlist entry"""

    account_value: float | None = None
    roe_30d: float | None = None
    ai_score: float | None = None
    ai_recommendation: str | None = None
    analyzed_at: str | None = None


class WatchlistItem(BaseModel):
    """Watchlist entry with analysis data"""

    address: str
    alias: str | None = None
    notes: str | None = None
    priority: int = 0
    auto_update: bool = True
    added_at: str | None = None
    last_updated: str | None = None
    total_cached_fills: int = 0
    data_coverage: DataCoverage | None = None
    analysis: WatchlistAnalysis | None = None


class SyncResult(BaseModel):
    """Result of a data sync operation"""

    address: str
    sync_type: str
    fills_added: int = 0
    total_fills: int = 0
    status: str
    error: str | None = None


class SyncLogItem(BaseModel):
    """Sync log entry"""

    id: int
    address: str
    sync_type: str
    started_at: str | None = None
    completed_at: str | None = None
    fills_added: int = 0
    status: str
    source: str | None = None
    error_message: str | None = None


class CacheCoverage(BaseModel):
    """Cache coverage for an address"""

    address: str
    total_fills: int = 0
    earliest_time: int | None = None
    latest_time: int | None = None
    earliest_date: str | None = None
    latest_date: str | None = None


class BackgroundSyncStatus(BaseModel):
    """Status of background sync"""

    running: bool
    watchlist_count: int
    next_sync_addresses: list[str] = Field(default_factory=list)
