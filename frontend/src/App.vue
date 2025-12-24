<script setup lang="ts">
import { RouterView, RouterLink } from 'vue-router'
</script>

<template>
  <div class="app">
    <nav class="navbar">
      <RouterLink to="/" class="nav-brand">HY 分析平台</RouterLink>
      <div class="nav-links">
        <RouterLink to="/" class="nav-link">分析</RouterLink>
        <RouterLink to="/watchlist" class="nav-link">关注列表</RouterLink>
      </div>
    </nav>
    <main>
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  background: #0a0a0a;
  padding: 1rem 2rem;
  border-bottom: 1px solid #333;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-brand {
  font-size: 1.25rem;
  font-weight: bold;
  color: #fff;
  text-decoration: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-link {
  color: #888;
  text-decoration: none;
  font-size: 0.95rem;
  transition: color 0.2s;
}

.nav-link:hover {
  color: #fff;
}

.nav-link.router-link-active {
  color: #667eea;
}

main {
  flex: 1;
}
</style>
