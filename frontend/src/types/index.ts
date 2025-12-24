export type Period = '7d' | '30d' | '90d' | '180d' | 'all'

export type Recommendation = 'strong_follow' | 'follow' | 'neutral' | 'avoid' | 'strong_avoid'

export type RiskLevel = 'low' | 'medium' | 'high' | 'extreme'

export type AnalysisStatusType = 'queued' | 'analyzing' | 'completed' | 'failed'

export interface TraderStats {
  address: string
  roe_all_time: number | null
  roe_7d: number | null
  roe_30d: number | null
  roe_90d: number | null
  pnl_all_time: number | null
  pnl_7d: number | null
  pnl_30d: number | null
  pnl_90d: number | null
  win_rate: number | null
  profit_factor: number | null
  max_drawdown_pct: number | null
  total_trades: number | null
  account_value: number | null
  long_short_ratio_all_time: number | null
  long_short_ratio_7d: number | null
  long_short_ratio_30d: number | null
  long_short_ratio_90d: number | null
  avg_trade_size_all_time: number | null
  avg_trade_size_7d: number | null
  avg_trade_size_30d: number | null
  avg_trade_size_90d: number | null
  avg_holding_time_all_time: number | null
  avg_holding_time_7d: number | null
  avg_holding_time_30d: number | null
  avg_holding_time_90d: number | null
  trade_frequency_all_time: number | null
  trade_frequency_7d: number | null
  trade_frequency_30d: number | null
  trade_frequency_90d: number | null
}

export interface DrawdownPeriod {
  start_time: string
  trough_time: string
  end_time: string
  drawdown: number
  drawdown_pct: number
}

export interface BacktestRequest {
  capital: number
  period: Period
}

export interface BacktestResult {
  address: string
  period: Period
  simulated_pnl: number
  simulated_roe: number
  max_drawdown: number
  max_drawdown_pct: number
  drawdown_periods: DrawdownPeriod[]
  trade_count: number
  skipped_trade_count: number
  scale: number
  baseline_account_value: number
}

export interface PortfolioData {
  day?: PnLHistory
  week?: PnLHistory
  month?: PnLHistory
  allTime?: PnLHistory
}

export interface PnLHistory {
  pnlHistory: number[][]
  accountValueHistory: number[][]
}

export interface BacktestSummary {
  period: Period
  pnl: number
  roe: number
  max_drawdown_pct: number
  trade_count: number
}

export interface AIEvaluation {
  score: number
  recommendation: Recommendation
  risk_level: RiskLevel
  reasoning: string
  trading_tags: string[]
  claude_score?: number | null
  codex_score?: number | null
  models_used?: string[]
}

export interface FullAnalysisRequest {
  capital?: number
  force_refresh?: boolean
}

export interface FullAnalysisResult {
  address: string
  analyzed_at: string
  account_value: number | null
  roe_7d: number | null
  roe_30d: number | null
  roe_90d: number | null
  roe_all_time: number | null
  pnl_7d: number | null
  pnl_30d: number | null
  pnl_90d: number | null
  pnl_all_time: number | null
  win_rate: number | null
  profit_factor: number | null
  max_drawdown_pct: number | null
  total_trades: number | null
  long_short_ratio: number | null
  avg_trade_size: number | null
  avg_holding_time: number | null
  trade_frequency: number | null
  backtest_7d: BacktestSummary | null
  backtest_30d: BacktestSummary | null
  backtest_90d: BacktestSummary | null
  backtest_all_time: BacktestSummary | null
  ai_evaluation: AIEvaluation | null
  status: string
}

export interface AnalysisStatus {
  address: string
  status: AnalysisStatusType
  progress: number
  current_step: string | null
  error: string | null
}

export interface TraderHistoryItem {
  address: string
  analyzed_at: string
  account_value: number | null
  roe_30d: number | null
  ai_score: number | null
  ai_recommendation: string | null
  trading_tags: string[]
}

export interface DataCoverage {
  start: number | null
  end: number | null
  start_date: string | null
  end_date: string | null
}

export interface WatchlistAnalysis {
  account_value: number | null
  roe_30d: number | null
  ai_score: number | null
  ai_recommendation: string | null
  analyzed_at: string | null
}

export interface WatchlistItem {
  address: string
  alias: string | null
  notes: string | null
  priority: number
  auto_update: boolean
  added_at: string | null
  last_updated: string | null
  total_cached_fills: number
  data_coverage: DataCoverage | null
  analysis: WatchlistAnalysis | null
}

export interface SyncResult {
  address: string
  sync_type: string
  fills_added: number
  total_fills: number
  status: string
  error?: string
}

export interface BackgroundSyncStatus {
  running: boolean
  watchlist_count: number
  next_sync_addresses: string[]
}

export interface CacheCoverage {
  address: string
  total_fills: number
  earliest_time: number | null
  latest_time: number | null
  earliest_date: string | null
  latest_date: string | null
}
