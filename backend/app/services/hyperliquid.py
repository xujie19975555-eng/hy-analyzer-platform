import httpx
from typing import Optional
from datetime import datetime, timedelta
import time

from app.config import get_settings


class HyperliquidService:
    """Hyperliquid API service"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.hyperliquid_api_url

    async def _post(self, payload: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/info",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            resp.raise_for_status()
            return resp.json()

    async def get_user_state(self, address: str) -> dict:
        return await self._post({
            "type": "clearinghouseState",
            "user": address.lower()
        })

    async def get_portfolio(self, address: str) -> dict:
        return await self._post({
            "type": "portfolio",
            "user": address.lower()
        })

    async def get_user_fills(self, address: str, aggregate: bool = True) -> list:
        return await self._post({
            "type": "userFills",
            "user": address.lower(),
            "aggregateByTime": aggregate
        })

    async def get_user_fills_by_time(
        self,
        address: str,
        start_time: int,
        end_time: Optional[int] = None,
        aggregate: bool = True
    ) -> list:
        payload = {
            "type": "userFillsByTime",
            "user": address.lower(),
            "startTime": start_time,
            "aggregateByTime": aggregate
        }
        if end_time:
            payload["endTime"] = end_time
        return await self._post(payload)


class SuperXService:
    """SuperX API service for trader stats"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.superx_api_url

    async def get_trader_stats(self, address: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/v1/wallets/{address.lower()}",
                timeout=30.0
            )
            resp.raise_for_status()
            return resp.json()


def get_hyperliquid_service() -> HyperliquidService:
    return HyperliquidService()


def get_superx_service() -> SuperXService:
    return SuperXService()
