# HY Analyzer Platform

Hyperliquid 交易员分析平台 - 查看和分析 Hyperliquid 交易员的绩效数据。

## 功能

- 🔍 交易员搜索（钱包地址）
- 📊 统计数据展示（ROE、PnL、胜率、回撤等）
- 📈 PnL 图表（多时间段）
- 📋 交易记录列表
- ⭐ 关注列表

## 技术栈

- **后端**: Python 3.12 + FastAPI
- **前端**: Vue.js 3 + TypeScript (计划中)
- **图表**: ECharts

## 快速开始

### 后端

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API 文档: http://localhost:8000/docs

### 运行测试

```bash
cd backend
pytest
```

## API 端点

| 端点 | 描述 |
|------|------|
| `GET /api/v1/traders/{address}/stats` | 获取交易员统计数据 |
| `GET /api/v1/traders/{address}/portfolio` | 获取投资组合数据 |
| `GET /api/v1/traders/{address}/trades` | 获取交易记录 |
| `GET /api/v1/health` | 健康检查 |

## 开发

本项目使用 AI 辅助开发，遵循严格的质量保障流程：

1. **测试驱动**: 所有功能必须有测试覆盖
2. **CI/CD**: GitHub Actions 自动运行测试和代码检查
3. **代码审查**: CodeRabbit 自动 PR 审查
4. **小步提交**: 每个功能/修复一个 PR

## License

MIT
# Test
