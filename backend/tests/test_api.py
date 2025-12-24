"""Tests for API endpoints"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_check():
    """Health check should return healthy status"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_root():
    """Root endpoint should return app info"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


@pytest.mark.anyio
async def test_get_trader_stats_invalid_address():
    """Should return 400 for invalid address"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/traders/invalid/stats")

    assert response.status_code == 400
    assert "Invalid" in response.json()["detail"]


@pytest.mark.anyio
async def test_get_trader_stats_valid_address():
    """Should call SuperX API for valid address"""
    mock_data = {"roeAllTime": 100.0, "pnlAllTime": 50000.0, "winRate": 65.0}

    with (
        patch(
            "app.services.hyperliquid.SuperXService.get_trader_stats", new_callable=AsyncMock
        ) as mock_get,
        patch(
            "app.services.hyperliquid.HyperliquidService.get_account_value", new_callable=AsyncMock
        ) as mock_av,
        patch(
            "app.services.hyperliquid.HyperliquidService.get_user_fills_windowed",
            new_callable=AsyncMock,
        ) as mock_fills,
    ):
        mock_get.return_value = mock_data
        mock_av.return_value = 1000.0
        mock_fills.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/traders/0x6c06bfd51ea8032ddaeea8b69009417b54f3587b/stats"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["roe_all_time"] == 100.0
        assert data["account_value"] == 1000.0


@pytest.mark.anyio
async def test_backtest_endpoint():
    """Should run backtest with valid params"""
    fills = [
        {
            "time": 1700000000000,
            "coin": "BTC",
            "side": "B",
            "px": "1",
            "sz": "20",
            "closedPnl": "100",
            "fee": "0",
        },
        {
            "time": 1700000001000,
            "coin": "BTC",
            "side": "A",
            "px": "1",
            "sz": "20",
            "closedPnl": "-200",
            "fee": "0",
        },
        {
            "time": 1700000002000,
            "coin": "BTC",
            "side": "B",
            "px": "1",
            "sz": "20",
            "closedPnl": "250",
            "fee": "0",
        },
    ]

    with (
        patch(
            "app.services.hyperliquid.HyperliquidService.get_account_value", new_callable=AsyncMock
        ) as mock_av,
        patch(
            "app.services.hyperliquid.HyperliquidService.get_user_fills_windowed",
            new_callable=AsyncMock,
        ) as mock_fills,
    ):
        mock_av.return_value = 1000.0
        mock_fills.return_value = fills

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/traders/0x6c06bfd51ea8032ddaeea8b69009417b54f3587b/backtest",
                json={"capital": 1000.0, "period": "7d"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["trade_count"] == 3
        assert data["simulated_pnl"] == 150.0
        assert data["max_drawdown"] == -200.0


@pytest.mark.anyio
async def test_backtest_min_notional_filter():
    """Should skip trades below 11 USDT min notional"""
    fills = [
        # notional = 1 * 5 = 5 USDT, below 11 -> skipped
        {
            "time": 1700000000000,
            "coin": "BTC",
            "side": "B",
            "px": "1",
            "sz": "5",
            "closedPnl": "10",
            "fee": "0",
        },
        # notional = 100 * 1 = 100 USDT, above 11 -> counted
        {
            "time": 1700000001000,
            "coin": "BTC",
            "side": "B",
            "px": "100",
            "sz": "1",
            "closedPnl": "100",
            "fee": "0",
        },
    ]

    with (
        patch(
            "app.services.hyperliquid.HyperliquidService.get_account_value", new_callable=AsyncMock
        ) as mock_av,
        patch(
            "app.services.hyperliquid.HyperliquidService.get_user_fills_windowed",
            new_callable=AsyncMock,
        ) as mock_fills,
    ):
        mock_av.return_value = 1000.0
        mock_fills.return_value = fills

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/traders/0x6c06bfd51ea8032ddaeea8b69009417b54f3587b/backtest",
                json={"capital": 1000.0, "period": "30d"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["trade_count"] == 1
        assert data["skipped_trade_count"] == 1
