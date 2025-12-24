import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { FullAnalysisResult, AnalysisStatus, TraderHistoryItem } from '../types'
import { runFullAnalysis, getAnalysisStatus, getHistory, isAnalysisRunning, getAnalysisResult, deleteTraderHistory } from '../api'

type SortField = 'ai_score' | 'analyzed_at' | 'roe_30d' | 'account_value'
type SortOrder = 'asc' | 'desc'

export const useTraderStore = defineStore('trader', () => {
  const address = ref('')
  const analysis = ref<FullAnalysisResult | null>(null)
  const status = ref<AnalysisStatus | null>(null)
  const history = ref<TraderHistoryItem[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const capital = ref(10000)
  const fromCache = ref(false)

  // History sorting state
  const historySortBy = ref<SortField>('ai_score')
  const historySortOrder = ref<SortOrder>('desc')

  // Modal state for AI evaluation detail
  const modalVisible = ref(false)
  const modalData = ref<FullAnalysisResult | null>(null)
  const modalLoading = ref(false)

  const isValidAddress = computed(() => {
    return /^0x[a-fA-F0-9]{40}$/.test(address.value)
  })

  const isAnalyzing = computed(() => {
    return status.value?.status === 'analyzing'
  })

  async function startAnalysis(forceRefresh = false) {
    if (!isValidAddress.value) {
      error.value = '请输入有效的以太坊地址'
      return
    }

    const running = await isAnalysisRunning()
    if (running) {
      error.value = '有其他分析正在进行中，请稍后再试'
      return
    }

    loading.value = true
    error.value = null
    analysis.value = null
    fromCache.value = false

    try {
      status.value = {
        address: address.value,
        status: 'analyzing',
        progress: 0,
        current_step: forceRefresh ? '强制刷新中...' : '初始化...',
        error: null
      }

      const result = await runFullAnalysis(address.value, {
        capital: capital.value,
        force_refresh: forceRefresh
      })
      analysis.value = result

      // Check if result is from cache by comparing analyzed_at
      const analyzedAt = new Date(result.analyzed_at)
      const now = new Date()
      const diffMs = now.getTime() - analyzedAt.getTime()
      fromCache.value = diffMs > 5000 // If analyzed more than 5s ago, it's from cache

      status.value = {
        address: address.value,
        status: 'completed',
        progress: 100,
        current_step: fromCache.value ? '已从缓存加载' : '完成',
        error: null
      }
      await fetchHistory()
    } catch (e: any) {
      error.value = e.response?.data?.detail || '分析失败'
      status.value = {
        address: address.value,
        status: 'failed',
        progress: 0,
        current_step: null,
        error: error.value
      }
    } finally {
      loading.value = false
    }
  }

  async function pollStatus() {
    if (!address.value) return
    try {
      status.value = await getAnalysisStatus(address.value)
    } catch {
      // ignore
    }
  }

  async function fetchHistory() {
    try {
      history.value = await getHistory({
        sort_by: historySortBy.value,
        order: historySortOrder.value
      })
    } catch {
      history.value = []
    }
  }

  function setHistorySort(field: SortField) {
    if (historySortBy.value === field) {
      // Toggle order if same field
      historySortOrder.value = historySortOrder.value === 'desc' ? 'asc' : 'desc'
    } else {
      historySortBy.value = field
      historySortOrder.value = 'desc' // Default to desc for new field
    }
    fetchHistory()
  }

  function setAddress(addr: string) {
    address.value = addr.toLowerCase().trim()
  }

  function setCapital(value: number) {
    capital.value = value
  }

  function selectFromHistory(item: TraderHistoryItem) {
    address.value = item.address
  }

  function clear() {
    analysis.value = null
    status.value = null
    error.value = null
  }

  async function showTraderDetail(addr: string) {
    modalLoading.value = true
    modalVisible.value = true
    modalData.value = null
    try {
      modalData.value = await getAnalysisResult(addr)
    } catch (e: any) {
      console.error('Failed to load trader detail:', e)
      modalData.value = null
    } finally {
      modalLoading.value = false
    }
  }

  function closeModal() {
    modalVisible.value = false
    modalData.value = null
  }

  async function deleteFromHistory(addr: string): Promise<boolean> {
    console.log('[Store] deleteFromHistory called:', addr)
    try {
      const response = await deleteTraderHistory(addr)
      console.log('[Store] deleteTraderHistory response:', response)
      await fetchHistory()
      console.log('[Store] fetchHistory done, new length:', history.value.length)
      return true
    } catch (e: any) {
      console.error('[Store] deleteFromHistory error:', e)
      error.value = e.response?.data?.detail || '删除失败'
      return false
    }
  }

  return {
    address,
    analysis,
    status,
    history,
    loading,
    error,
    capital,
    fromCache,
    historySortBy,
    historySortOrder,
    modalVisible,
    modalData,
    modalLoading,
    isValidAddress,
    isAnalyzing,
    startAnalysis,
    pollStatus,
    fetchHistory,
    setHistorySort,
    setAddress,
    setCapital,
    selectFromHistory,
    clear,
    showTraderDetail,
    closeModal,
    deleteFromHistory,
  }
})
