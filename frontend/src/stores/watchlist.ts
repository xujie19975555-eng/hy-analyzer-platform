import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { WatchlistItem, BackgroundSyncStatus, SyncResult } from '../types'
import {
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  updateWatchlistEntry,
  syncWatchlistAddress,
  getSyncStatus,
  startBackgroundSync,
  stopBackgroundSync,
} from '../api'

export const useWatchlistStore = defineStore('watchlist', () => {
  const items = ref<WatchlistItem[]>([])
  const syncStatus = ref<BackgroundSyncStatus | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const syncingAddresses = ref<Set<string>>(new Set())

  const addForm = ref({
    address: '',
    alias: '',
    notes: '',
    priority: 50,
    auto_update: true,
  })

  const editingItem = ref<WatchlistItem | null>(null)
  const showAddModal = ref(false)
  const showEditModal = ref(false)

  const watchlistCount = computed(() => items.value.length)

  const sortedItems = computed(() => {
    return [...items.value].sort((a, b) => {
      if (b.priority !== a.priority) return b.priority - a.priority
      const aScore = a.analysis?.ai_score ?? -1
      const bScore = b.analysis?.ai_score ?? -1
      return bScore - aScore
    })
  })

  async function fetchWatchlist() {
    loading.value = true
    error.value = null
    try {
      items.value = await getWatchlist()
    } catch (e: any) {
      error.value = e.response?.data?.detail || '获取关注列表失败'
      items.value = []
    } finally {
      loading.value = false
    }
  }

  async function fetchSyncStatus() {
    try {
      syncStatus.value = await getSyncStatus()
    } catch {
      syncStatus.value = null
    }
  }

  async function addItem() {
    if (!addForm.value.address) {
      error.value = '请输入钱包地址'
      return false
    }

    loading.value = true
    error.value = null
    try {
      await addToWatchlist({
        address: addForm.value.address,
        alias: addForm.value.alias || undefined,
        notes: addForm.value.notes || undefined,
        priority: addForm.value.priority,
        auto_update: addForm.value.auto_update,
      })
      resetAddForm()
      showAddModal.value = false
      await fetchWatchlist()
      return true
    } catch (e: any) {
      error.value = e.response?.data?.detail || '添加失败'
      return false
    } finally {
      loading.value = false
    }
  }

  async function removeItem(address: string) {
    loading.value = true
    error.value = null
    try {
      await removeFromWatchlist(address)
      await fetchWatchlist()
      return true
    } catch (e: any) {
      error.value = e.response?.data?.detail || '删除失败'
      return false
    } finally {
      loading.value = false
    }
  }

  async function updateItem(address: string, updates: Partial<WatchlistItem>) {
    loading.value = true
    error.value = null
    try {
      await updateWatchlistEntry(address, {
        alias: updates.alias ?? undefined,
        notes: updates.notes ?? undefined,
        priority: updates.priority,
        auto_update: updates.auto_update,
      })
      showEditModal.value = false
      editingItem.value = null
      await fetchWatchlist()
      return true
    } catch (e: any) {
      error.value = e.response?.data?.detail || '更新失败'
      return false
    } finally {
      loading.value = false
    }
  }

  async function syncItem(address: string, full: boolean = false): Promise<SyncResult | null> {
    syncingAddresses.value.add(address)
    try {
      const result = await syncWatchlistAddress(address, full)
      await fetchWatchlist()
      return result
    } catch (e: any) {
      error.value = e.response?.data?.detail || '同步失败'
      return null
    } finally {
      syncingAddresses.value.delete(address)
    }
  }

  async function startSync(interval: number = 60) {
    try {
      await startBackgroundSync(interval)
      await fetchSyncStatus()
    } catch (e: any) {
      error.value = e.response?.data?.detail || '启动同步失败'
    }
  }

  async function stopSync() {
    try {
      await stopBackgroundSync()
      await fetchSyncStatus()
    } catch (e: any) {
      error.value = e.response?.data?.detail || '停止同步失败'
    }
  }

  function resetAddForm() {
    addForm.value = {
      address: '',
      alias: '',
      notes: '',
      priority: 50,
      auto_update: true,
    }
  }

  function openAddModal() {
    resetAddForm()
    showAddModal.value = true
  }

  function closeAddModal() {
    showAddModal.value = false
    resetAddForm()
  }

  function openEditModal(item: WatchlistItem) {
    editingItem.value = { ...item }
    showEditModal.value = true
  }

  function closeEditModal() {
    showEditModal.value = false
    editingItem.value = null
  }

  function isSyncing(address: string): boolean {
    return syncingAddresses.value.has(address)
  }

  return {
    items,
    syncStatus,
    loading,
    error,
    addForm,
    editingItem,
    showAddModal,
    showEditModal,
    watchlistCount,
    sortedItems,
    fetchWatchlist,
    fetchSyncStatus,
    addItem,
    removeItem,
    updateItem,
    syncItem,
    startSync,
    stopSync,
    openAddModal,
    closeAddModal,
    openEditModal,
    closeEditModal,
    isSyncing,
  }
})
