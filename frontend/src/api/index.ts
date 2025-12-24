import axios from 'axios'
import type {
  TraderStats, BacktestRequest, BacktestResult, PortfolioData,
  FullAnalysisRequest, FullAnalysisResult, AnalysisStatus, TraderHistoryItem
} from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  timeout: 300000,
})

export async function getTraderStats(address: string): Promise<TraderStats> {
  const { data } = await api.get<TraderStats>(`/traders/${address}/stats`)
  return data
}

export async function getTraderPortfolio(address: string): Promise<PortfolioData> {
  const { data } = await api.get<PortfolioData>(`/traders/${address}/portfolio`)
  return data
}

export async function runBacktest(address: string, request: BacktestRequest): Promise<BacktestResult> {
  const { data } = await api.post<BacktestResult>(`/traders/${address}/backtest`, request)
  return data
}

export async function runFullAnalysis(address: string, request?: FullAnalysisRequest): Promise<FullAnalysisResult> {
  const { data } = await api.post<FullAnalysisResult>(`/analyze/${address}`, request || {})
  return data
}

export async function getAnalysisStatus(address: string): Promise<AnalysisStatus> {
  const { data } = await api.get<AnalysisStatus>(`/analyze/${address}/status`)
  return data
}

export async function getAnalysisResult(address: string): Promise<FullAnalysisResult> {
  const { data } = await api.get<FullAnalysisResult>(`/analyze/${address}/result`)
  return data
}

export interface HistoryParams {
  limit?: number
  offset?: number
  sort_by?: 'ai_score' | 'analyzed_at' | 'roe_30d' | 'account_value'
  order?: 'asc' | 'desc'
  min_score?: number
  recommendation?: string
}

export async function getHistory(params: HistoryParams = {}): Promise<TraderHistoryItem[]> {
  const { data } = await api.get<TraderHistoryItem[]>('/history', {
    params: {
      limit: params.limit ?? 50,
      offset: params.offset ?? 0,
      sort_by: params.sort_by ?? 'ai_score',
      order: params.order ?? 'desc',
      ...(params.min_score !== undefined && { min_score: params.min_score }),
      ...(params.recommendation && { recommendation: params.recommendation }),
    }
  })
  return data
}

export async function isAnalysisRunning(): Promise<boolean> {
  const { data } = await api.get<{ running: boolean }>('/analysis/running')
  return data.running
}

export async function healthCheck(): Promise<{ status: string }> {
  const { data } = await api.get('/health')
  return data
}

export async function deleteTraderHistory(address: string): Promise<{ address: string; status: string }> {
  const { data } = await api.delete<{ address: string; status: string }>(`/history/${address}`)
  return data
}

// === Watchlist API ===

export interface WatchlistItem {
  address: string
  alias: string | null
  notes: string | null
  priority: number
  auto_update: boolean
  added_at: string | null
  last_updated: string | null
  total_cached_fills: number
  data_coverage: {
    start: number | null
    end: number | null
    start_date: string | null
    end_date: string | null
  } | null
  analysis: {
    account_value: number | null
    roe_30d: number | null
    ai_score: number | null
    ai_recommendation: string | null
    analyzed_at: string | null
  } | null
}

export interface WatchlistAddRequest {
  address: string
  alias?: string
  notes?: string
  priority?: number
  auto_update?: boolean
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

export async function getWatchlist(): Promise<WatchlistItem[]> {
  const { data } = await api.get<WatchlistItem[]>('/watchlist')
  return data
}

export async function addToWatchlist(request: WatchlistAddRequest): Promise<any> {
  const { data } = await api.post('/watchlist', request)
  return data
}

export async function removeFromWatchlist(address: string): Promise<any> {
  const { data } = await api.delete(`/watchlist/${address}`)
  return data
}

export async function updateWatchlistEntry(address: string, updates: Partial<WatchlistAddRequest>): Promise<any> {
  const { data } = await api.put(`/watchlist/${address}`, updates)
  return data
}

export async function syncWatchlistAddress(address: string, full: boolean = false): Promise<SyncResult> {
  const { data } = await api.post<SyncResult>(`/watchlist/${address}/sync`, null, {
    params: { full }
  })
  return data
}

export async function getSyncStatus(): Promise<BackgroundSyncStatus> {
  const { data } = await api.get<BackgroundSyncStatus>('/sync/status')
  return data
}

export async function startBackgroundSync(interval: number = 60): Promise<any> {
  const { data } = await api.post('/sync/start', null, {
    params: { interval }
  })
  return data
}

export async function stopBackgroundSync(): Promise<any> {
  const { data } = await api.post('/sync/stop')
  return data
}

export async function getCacheCoverage(address: string): Promise<CacheCoverage> {
  const { data } = await api.get<CacheCoverage>(`/cache/${address}/coverage`)
  return data
}
