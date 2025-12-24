import os
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./traders.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TraderAnalysis(Base):
    __tablename__ = "trader_analyses"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(42), unique=True, index=True, nullable=False)

    # Basic info
    account_value = Column(Float, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # ROE metrics
    roe_7d = Column(Float, nullable=True)
    roe_30d = Column(Float, nullable=True)
    roe_90d = Column(Float, nullable=True)
    roe_all_time = Column(Float, nullable=True)

    # PnL metrics
    pnl_7d = Column(Float, nullable=True)
    pnl_30d = Column(Float, nullable=True)
    pnl_90d = Column(Float, nullable=True)
    pnl_all_time = Column(Float, nullable=True)

    # Risk metrics
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)

    # Trading style
    long_short_ratio = Column(Float, nullable=True)
    avg_trade_size = Column(Float, nullable=True)
    avg_holding_time = Column(Float, nullable=True)  # seconds
    trade_frequency = Column(Float, nullable=True)  # trades per day
    trading_days = Column(Integer, nullable=True)  # Days since first trade (survival days)
    first_trade_date = Column(DateTime, nullable=True)  # Date of first trade

    # Backtest results (default 10000 USDT)
    backtest_pnl_7d = Column(Float, nullable=True)
    backtest_pnl_30d = Column(Float, nullable=True)
    backtest_pnl_90d = Column(Float, nullable=True)
    backtest_pnl_all_time = Column(Float, nullable=True)
    backtest_roe_7d = Column(Float, nullable=True)
    backtest_roe_30d = Column(Float, nullable=True)
    backtest_roe_90d = Column(Float, nullable=True)
    backtest_roe_all_time = Column(Float, nullable=True)
    backtest_max_dd_7d = Column(Float, nullable=True)
    backtest_max_dd_30d = Column(Float, nullable=True)
    backtest_max_dd_90d = Column(Float, nullable=True)
    backtest_max_dd_all_time = Column(Float, nullable=True)
    backtest_trade_count_7d = Column(Integer, nullable=True)
    backtest_trade_count_30d = Column(Integer, nullable=True)
    backtest_trade_count_90d = Column(Integer, nullable=True)
    backtest_trade_count_all_time = Column(Integer, nullable=True)

    # AI evaluation
    ai_score = Column(Float, nullable=True)  # 0-100
    ai_recommendation = Column(
        String(20), nullable=True
    )  # "strong_follow", "follow", "neutral", "avoid", "strong_avoid"
    ai_reasoning = Column(Text, nullable=True)
    ai_risk_level = Column(String(20), nullable=True)  # "low", "medium", "high", "extreme"
    ai_claude_score = Column(Float, nullable=True)  # Claude model individual score
    ai_codex_score = Column(Float, nullable=True)  # Codex/GPT model individual score
    ai_models_used = Column(String(100), nullable=True)  # Comma-separated: "claude,openai"
    ai_trading_tags = Column(String(200), nullable=True)  # Comma-separated trading style tags
    ai_score_breakdown = Column(JSON, nullable=True)  # Score breakdown items [{item, points, type}]

    # Raw data storage
    raw_stats = Column(JSON, nullable=True)
    raw_backtest = Column(JSON, nullable=True)

    # Status
    status = Column(String(20), default="pending")  # pending, analyzing, completed, failed
    error_message = Column(Text, nullable=True)

    # Data completeness indicators
    data_limited = Column(Boolean, default=False)  # True if only 90-day data due to API limits
    data_coverage_days = Column(Integer, nullable=True)  # Actual days of data coverage


class AnalysisQueue(Base):
    __tablename__ = "analysis_queue"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(42), index=True, nullable=False)
    status = Column(String(20), default="queued")  # queued, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)


class FillsCache(Base):
    """Cache for historical fills data from S3/API"""

    __tablename__ = "fills_cache"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(42), index=True, nullable=False)
    coin = Column(String(20), nullable=False)
    fill_time = Column(Integer, index=True, nullable=False)  # timestamp in ms
    fill_hash = Column(String(66), unique=True, index=True, nullable=False)
    side = Column(String(1), nullable=False)  # B or A
    px = Column(String(30), nullable=False)
    sz = Column(String(30), nullable=False)
    oid = Column(Integer, nullable=True)
    start_position = Column(String(30), nullable=True)
    closed_pnl = Column(String(30), nullable=True)
    fee = Column(String(30), nullable=True)
    raw_data = Column(JSON, nullable=True)
    source = Column(String(20), nullable=True)  # api, node_trades, node_fills, node_fills_by_block
    created_at = Column(DateTime, default=datetime.utcnow)


class Watchlist(Base):
    """User watchlist for tracking traders"""

    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(42), unique=True, index=True, nullable=False)
    alias = Column(String(50), nullable=True)  # User-defined nickname
    notes = Column(Text, nullable=True)  # User notes
    priority = Column(Integer, default=0)  # Higher = more important
    auto_update = Column(Boolean, default=True)  # Auto-update fills data
    added_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=True)  # Last fills sync time
    last_fill_time = Column(Integer, nullable=True)  # Latest fill timestamp in ms
    total_cached_fills = Column(Integer, default=0)
    data_coverage_start = Column(Integer, nullable=True)  # Earliest fill timestamp
    data_coverage_end = Column(Integer, nullable=True)  # Latest fill timestamp


class DataSyncLog(Base):
    """Log of data sync operations"""

    __tablename__ = "data_sync_log"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(42), index=True, nullable=False)
    sync_type = Column(String(20), nullable=False)  # incremental, full, api_backfill
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    fills_added = Column(Integer, default=0)
    time_range_start = Column(Integer, nullable=True)
    time_range_end = Column(Integer, nullable=True)
    source = Column(String(20), nullable=True)
    status = Column(String(20), default="running")  # running, completed, failed
    error_message = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
