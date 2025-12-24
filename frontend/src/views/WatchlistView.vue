<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useWatchlistStore } from '../stores/watchlist'
import type { Recommendation, WatchlistItem } from '../types'

const store = useWatchlistStore()
const router = useRouter()

const recommendationText: Record<Recommendation, string> = {
  strong_follow: '强烈推荐',
  follow: '推荐',
  neutral: '中性',
  avoid: '不建议',
  strong_avoid: '强烈不建议'
}

function formatNumber(val: number | null | undefined, decimals = 2): string {
  if (val === null || val === undefined) return '-'
  return val.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function formatPercent(val: number | null | undefined): string {
  if (val === null || val === undefined) return '-'
  const sign = val >= 0 ? '+' : ''
  return `${sign}${val.toFixed(2)}%`
}

function shortAddress(addr: string): string {
  return `${addr.slice(0, 6)}...${addr.slice(-4)}`
}

function formatDate(isoString: string | null): string {
  if (!isoString) return '-'
  const date = new Date(isoString)
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function getScoreClass(score: number | null): string {
  if (score === null) return ''
  if (score >= 80) return 'score-elite'
  if (score >= 60) return 'score-good'
  if (score >= 40) return 'score-average'
  return 'score-low'
}

function getPriorityLabel(priority: number): string {
  if (priority >= 80) return '高'
  if (priority >= 40) return '中'
  return '低'
}

function goToAnalyze(address: string) {
  router.push({ path: '/', query: { address } })
}

async function handleSync(item: WatchlistItem, full: boolean = false) {
  await store.syncItem(item.address, full)
}

async function handleRemove(item: WatchlistItem) {
  if (confirm(`确定要从关注列表中移除 ${item.alias || shortAddress(item.address)} 吗？`)) {
    await store.removeItem(item.address)
  }
}

async function handleSaveEdit() {
  if (!store.editingItem) return
  await store.updateItem(store.editingItem.address, store.editingItem)
}

const isValidAddress = computed(() => {
  return /^0x[a-fA-F0-9]{40}$/.test(store.addForm.address)
})

onMounted(() => {
  store.fetchWatchlist()
  store.fetchSyncStatus()
})
</script>

<template>
  <div class="watchlist-page">
    <div class="page-header">
      <h1>关注列表</h1>
      <p class="subtitle">追踪和管理你关注的交易员</p>
    </div>

    <div class="toolbar">
      <button class="btn-primary" @click="store.openAddModal">
        + 添加交易员
      </button>
      <div class="sync-controls">
        <span v-if="store.syncStatus" class="sync-status">
          <span :class="['status-dot', store.syncStatus.running ? 'active' : '']"></span>
          {{ store.syncStatus.running ? '自动同步中' : '自动同步已停止' }}
          <span class="sync-count">({{ store.syncStatus.watchlist_count }} 个地址)</span>
        </span>
        <button
          v-if="store.syncStatus && !store.syncStatus.running"
          class="btn-outline"
          @click="store.startSync(60)"
        >
          启动自动同步
        </button>
        <button
          v-if="store.syncStatus?.running"
          class="btn-outline danger"
          @click="store.stopSync"
        >
          停止同步
        </button>
      </div>
    </div>

    <div v-if="store.error" class="error-message">
      {{ store.error }}
    </div>

    <div v-if="store.loading && !store.items.length" class="loading">
      <div class="spinner"></div>
      <p>加载中...</p>
    </div>

    <div v-else-if="!store.items.length" class="empty-state">
      <p>还没有关注任何交易员</p>
      <button class="btn-primary" @click="store.openAddModal">添加第一个</button>
    </div>

    <div v-else class="watchlist-grid">
      <div
        v-for="item in store.sortedItems"
        :key="item.address"
        class="watchlist-card"
        :class="{ syncing: store.isSyncing(item.address) }"
      >
        <div class="card-header">
          <div class="card-title">
            <span class="alias">{{ item.alias || shortAddress(item.address) }}</span>
            <span class="priority-badge" :class="'priority-' + getPriorityLabel(item.priority).toLowerCase()">
              {{ getPriorityLabel(item.priority) }}优先级
            </span>
          </div>
          <div class="card-actions">
            <button class="btn-icon" title="编辑" @click="store.openEditModal(item)">
              <span class="icon">✎</span>
            </button>
            <button class="btn-icon" title="同步数据" @click="handleSync(item)" :disabled="store.isSyncing(item.address)">
              <span class="icon" :class="{ spinning: store.isSyncing(item.address) }">↻</span>
            </button>
            <button class="btn-icon danger" title="移除" @click="handleRemove(item)">
              <span class="icon">✕</span>
            </button>
          </div>
        </div>

        <div class="card-address">
          <a :href="`https://app.hyperliquid.xyz/portfolio/${item.address}`" target="_blank">
            {{ shortAddress(item.address) }}
          </a>
          <button class="btn-link" @click="goToAnalyze(item.address)">分析</button>
        </div>

        <div class="card-stats">
          <div class="stat-item" v-if="item.analysis">
            <span class="stat-label">AI评分</span>
            <span class="stat-value" :class="getScoreClass(item.analysis.ai_score)">
              {{ item.analysis.ai_score?.toFixed(1) ?? '-' }}
            </span>
          </div>
          <div class="stat-item" v-if="item.analysis">
            <span class="stat-label">30天ROE</span>
            <span class="stat-value" :class="{ positive: (item.analysis.roe_30d ?? 0) > 0, negative: (item.analysis.roe_30d ?? 0) < 0 }">
              {{ formatPercent(item.analysis.roe_30d) }}
            </span>
          </div>
          <div class="stat-item" v-if="item.analysis">
            <span class="stat-label">账户价值</span>
            <span class="stat-value">${{ formatNumber(item.analysis.account_value, 0) }}</span>
          </div>
          <div class="stat-item" v-if="item.analysis?.ai_recommendation">
            <span class="stat-label">建议</span>
            <span class="stat-value recommendation">
              {{ recommendationText[item.analysis.ai_recommendation as Recommendation] || item.analysis.ai_recommendation }}
            </span>
          </div>
        </div>

        <div class="card-data">
          <div class="data-row">
            <span class="data-label">缓存数据</span>
            <span class="data-value">{{ item.total_cached_fills.toLocaleString() }} 笔交易</span>
          </div>
          <div class="data-row" v-if="item.data_coverage">
            <span class="data-label">数据范围</span>
            <span class="data-value">
              {{ item.data_coverage.start_date?.slice(5, 10) || '-' }} ~ {{ item.data_coverage.end_date?.slice(5, 10) || '-' }}
            </span>
          </div>
          <div class="data-row">
            <span class="data-label">最后更新</span>
            <span class="data-value">{{ formatDate(item.last_updated) }}</span>
          </div>
        </div>

        <div class="card-notes" v-if="item.notes">
          <p>{{ item.notes }}</p>
        </div>
      </div>
    </div>

    <!-- Add Modal -->
    <div v-if="store.showAddModal" class="modal-overlay" @click.self="store.closeAddModal">
      <div class="modal-content">
        <button class="modal-close" @click="store.closeAddModal">&times;</button>
        <h2>添加交易员到关注列表</h2>

        <div class="form-group">
          <label>钱包地址 *</label>
          <input
            v-model="store.addForm.address"
            type="text"
            placeholder="0x..."
            :class="{ invalid: store.addForm.address && !isValidAddress }"
          />
          <span v-if="store.addForm.address && !isValidAddress" class="field-error">
            请输入有效的以太坊地址
          </span>
        </div>

        <div class="form-group">
          <label>别名</label>
          <input
            v-model="store.addForm.alias"
            type="text"
            placeholder="给这个交易员起个名字"
            maxlength="50"
          />
        </div>

        <div class="form-group">
          <label>备注</label>
          <textarea
            v-model="store.addForm.notes"
            placeholder="添加一些备注..."
            rows="3"
          ></textarea>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>优先级 ({{ store.addForm.priority }})</label>
            <input
              v-model.number="store.addForm.priority"
              type="range"
              min="0"
              max="100"
            />
          </div>
          <div class="form-group checkbox">
            <label>
              <input
                v-model="store.addForm.auto_update"
                type="checkbox"
              />
              自动更新数据
            </label>
          </div>
        </div>

        <div class="modal-actions">
          <button class="btn-outline" @click="store.closeAddModal">取消</button>
          <button
            class="btn-primary"
            @click="store.addItem"
            :disabled="!isValidAddress || store.loading"
          >
            {{ store.loading ? '添加中...' : '添加' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Edit Modal -->
    <div v-if="store.showEditModal && store.editingItem" class="modal-overlay" @click.self="store.closeEditModal">
      <div class="modal-content">
        <button class="modal-close" @click="store.closeEditModal">&times;</button>
        <h2>编辑关注项</h2>
        <p class="modal-address">{{ shortAddress(store.editingItem.address) }}</p>

        <div class="form-group">
          <label>别名</label>
          <input
            v-model="store.editingItem.alias"
            type="text"
            placeholder="给这个交易员起个名字"
            maxlength="50"
          />
        </div>

        <div class="form-group">
          <label>备注</label>
          <textarea
            v-model="store.editingItem.notes"
            placeholder="添加一些备注..."
            rows="3"
          ></textarea>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label>优先级 ({{ store.editingItem.priority }})</label>
            <input
              v-model.number="store.editingItem.priority"
              type="range"
              min="0"
              max="100"
            />
          </div>
          <div class="form-group checkbox">
            <label>
              <input
                v-model="store.editingItem.auto_update"
                type="checkbox"
              />
              自动更新数据
            </label>
          </div>
        </div>

        <div class="modal-actions">
          <button class="btn-outline danger" @click="handleRemove(store.editingItem)">删除</button>
          <div class="spacer"></div>
          <button class="btn-outline" @click="store.closeEditModal">取消</button>
          <button
            class="btn-primary"
            @click="handleSaveEdit"
            :disabled="store.loading"
          >
            {{ store.loading ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.watchlist-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.page-header {
  margin-bottom: 2rem;
}

.page-header h1 {
  font-size: 2rem;
  margin: 0 0 0.5rem 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  color: #888;
  margin: 0;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.sync-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.sync-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #888;
  font-size: 0.9rem;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #666;
}

.status-dot.active {
  background: #48bb78;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.sync-count {
  color: #666;
}

.btn-primary {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: bold;
  transition: all 0.2s;
}

.btn-primary:hover:not(:disabled) {
  transform: scale(1.02);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-outline {
  padding: 0.5rem 1rem;
  background: transparent;
  color: #667eea;
  border: 1px solid #667eea;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-outline:hover {
  background: #667eea20;
}

.btn-outline.danger {
  color: #f56565;
  border-color: #f56565;
}

.btn-outline.danger:hover {
  background: #f5656520;
}

.btn-icon {
  width: 32px;
  height: 32px;
  padding: 0;
  background: transparent;
  border: 1px solid #333;
  border-radius: 6px;
  color: #888;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-icon:hover:not(:disabled) {
  border-color: #667eea;
  color: #667eea;
}

.btn-icon.danger:hover {
  border-color: #f56565;
  color: #f56565;
}

.btn-icon:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-link {
  background: none;
  border: none;
  color: #667eea;
  cursor: pointer;
  padding: 0;
  font-size: 0.85rem;
}

.btn-link:hover {
  text-decoration: underline;
}

.icon {
  font-size: 1rem;
}

.icon.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-message {
  background: #f5656520;
  border: 1px solid #f56565;
  color: #f56565;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.loading {
  text-align: center;
  padding: 3rem;
  color: #888;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #333;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  background: #1a1a1a;
  border-radius: 16px;
  color: #888;
}

.empty-state p {
  margin-bottom: 1.5rem;
}

.watchlist-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 1.5rem;
}

.watchlist-card {
  background: #1a1a1a;
  border-radius: 12px;
  padding: 1.25rem;
  border: 1px solid #333;
  transition: all 0.2s;
}

.watchlist-card:hover {
  border-color: #667eea40;
}

.watchlist-card.syncing {
  opacity: 0.7;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.75rem;
}

.card-title {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.alias {
  font-size: 1.1rem;
  font-weight: 600;
}

.priority-badge {
  font-size: 0.7rem;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  width: fit-content;
}

.priority-badge.priority-高 { background: #f6ad5520; color: #f6ad55; }
.priority-badge.priority-中 { background: #4299e120; color: #4299e1; }
.priority-badge.priority-低 { background: #a0aec020; color: #a0aec0; }

.card-actions {
  display: flex;
  gap: 0.5rem;
}

.card-address {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.card-address a {
  font-family: monospace;
  color: #667eea;
  text-decoration: none;
  font-size: 0.9rem;
}

.card-address a:hover {
  text-decoration: underline;
}

.card-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
  margin-bottom: 1rem;
  padding: 0.75rem;
  background: #0a0a0a;
  border-radius: 8px;
}

.stat-item {
  display: flex;
  flex-direction: column;
}

.stat-label {
  font-size: 0.75rem;
  color: #888;
}

.stat-value {
  font-size: 1rem;
  font-weight: 600;
}

.stat-value.score-elite { color: #ffd700; }
.stat-value.score-good { color: #48bb78; }
.stat-value.score-average { color: #a0aec0; }
.stat-value.score-low { color: #f56565; }

.stat-value.positive { color: #48bb78; }
.stat-value.negative { color: #f56565; }

.stat-value.recommendation {
  font-size: 0.85rem;
}

.card-data {
  font-size: 0.85rem;
  color: #888;
}

.data-row {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
}

.data-value {
  color: #aaa;
}

.card-notes {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid #333;
}

.card-notes p {
  margin: 0;
  color: #888;
  font-size: 0.85rem;
  font-style: italic;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}

.modal-content {
  background: #1a1a1a;
  border-radius: 16px;
  max-width: 500px;
  width: 100%;
  padding: 2rem;
  position: relative;
}

.modal-close {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: transparent;
  border: none;
  color: #888;
  font-size: 1.5rem;
  cursor: pointer;
}

.modal-close:hover {
  color: #fff;
}

.modal-content h2 {
  margin: 0 0 1.5rem 0;
  font-size: 1.25rem;
}

.modal-address {
  font-family: monospace;
  color: #667eea;
  margin: -1rem 0 1.5rem 0;
  font-size: 0.9rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  color: #888;
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
}

.form-group input[type="text"],
.form-group textarea {
  width: 100%;
  padding: 0.75rem;
  background: #0a0a0a;
  border: 1px solid #333;
  border-radius: 8px;
  color: #fff;
  font-size: 1rem;
}

.form-group input[type="text"]:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
}

.form-group input.invalid {
  border-color: #f56565;
}

.field-error {
  color: #f56565;
  font-size: 0.8rem;
  margin-top: 0.25rem;
}

.form-group input[type="range"] {
  width: 100%;
}

.form-group.checkbox label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.form-row {
  display: flex;
  gap: 1rem;
}

.form-row .form-group {
  flex: 1;
}

.modal-actions {
  display: flex;
  gap: 0.75rem;
  margin-top: 1.5rem;
  justify-content: flex-end;
}

.modal-actions .spacer {
  flex: 1;
}

@media (max-width: 768px) {
  .toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .sync-controls {
    flex-wrap: wrap;
    justify-content: center;
  }

  .watchlist-grid {
    grid-template-columns: 1fr;
  }

  .card-stats {
    grid-template-columns: 1fr;
  }
}
</style>
