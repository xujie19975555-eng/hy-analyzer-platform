"""Tests for API endpoints"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

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
    mock_data = {
        "roeAllTime": 100.0,
        "pnlAllTime": 50000.0,
        "winRate": 65.0
    }

    with patch("app.api.routes.get_superx_service") as mock_service:
        mock_instance = AsyncMock()
        mock_instance.get_trader_stats.return_value = mock_data
        mock_service.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/traders/0x6c06bfd51ea8032ddaeea8b69009417b54f3587b/stats"
            )

    assert response.status_code == 200
    data = response.json()
    assert data["roe_all_time"] == 100.0
