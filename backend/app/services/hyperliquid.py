import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings
from app.services.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


@dataclass
class FillsCache:
    """Simple TTL cache for fills data"""

    data: list[dict]
    timestamp: float

    def is_valid(self, ttl_seconds: int = 300) -> bool:
        return time.time() - self.timestamp < ttl_seconds


class HyperliquidService:
    """Hyperliquid API service with weighted rate limiting"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.hyperliquid_api_url
        self.rate_limiter = get_rate_limiter()
        self._fills_cache: dict[str, FillsCache] = {}  # address -> cache
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    async def _post(self, payload: dict, max_retries: int = 3) -> Any:
        request_type = payload.get("type", "default")
        await self.rate_limiter.acquire(request_type)

        delay = 1.0
        client = await self._get_client()
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    f"{self.base_url}/info",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    logger.warning("Rate limited (429), retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise

    async def get_user_state(self, address: str) -> dict:
        return await self._post({"type": "clearinghouseState", "user": address.lower()})

    async def get_portfolio(self, address: str) -> dict:
        return await self._post({"type": "portfolio", "user": address.lower()})

    async def get_user_fills(self, address: str, aggregate: bool = True) -> list:
        return await self._post(
            {"type": "userFills", "user": address.lower(), "aggregateByTime": aggregate}
        )

    async def get_user_fills_by_time(
        self, address: str, start_time: int, end_time: int | None = None, aggregate: bool = True
    ) -> list:
        payload = {
            "type": "userFillsByTime",
            "user": address.lower(),
            "startTime": start_time,
            "aggregateByTime": aggregate,
        }
        if end_time is not None:
            payload["endTime"] = end_time
        return await self._post(payload)

    async def get_account_value(self, address: str) -> float | None:
        """Get account value from clearinghouseState"""
        state = await self.get_user_state(address)

        def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
            cur = obj
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            return cur

        candidates = [
            ("marginSummary", "accountValue"),
            ("crossMarginSummary", "accountValue"),
            ("marginSummary", "totalAccountValue"),
        ]
        for path in candidates:
            val = _get_path(state, path)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        return None

    async def get_user_fills_windowed(
        self,
        address: str,
        start_time: int,
        end_time: int | None = None,
        aggregate: bool = False,
        window_ms: int = 1000 * 60 * 60 * 24 * 30,  # 30 days per window
        use_cache: bool = True,
        cache_ttl: int = 300,  # 5 minutes
        max_fills: int = 10000,  # Hyperliquid API limit
        use_s3_fallback: bool = True,  # Try S3 if API hits limit
    ) -> list[dict]:
        """
        Fetch fills in time windows to handle large date ranges.

        Note: Hyperliquid API limits:
        - Max 2000 fills per request
        - Max 10000 most recent fills per address

        If use_s3_fallback=True and we hit the 10K limit, will attempt
        to fetch older data from S3 archive (unlimited history).
        """
        addr_lower = address.lower()
        cache_key = f"{addr_lower}:{start_time}:{end_time}:{aggregate}:{use_s3_fallback}"

        # Check cache first
        if use_cache and cache_key in self._fills_cache:
            cached = self._fills_cache[cache_key]
            if cached.is_valid(cache_ttl):
                logger.debug("Cache hit for fills: %s", cache_key)
                return cached.data

        all_fills: list[dict] = []
        seen: set[tuple[Any, ...]] = set()

        if end_time is None:
            end_time = int(time.time() * 1000)

        def _to_int_ms(v: Any) -> int | None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        def _dedupe_key(fill: dict) -> tuple[Any, ...]:
            h = fill.get("hash")
            if h:
                return ("hash", h)
            return (
                "t",
                _to_int_ms(fill.get("time")),
                fill.get("coin"),
                fill.get("side"),
                fill.get("px"),
                fill.get("sz"),
                fill.get("closedPnl"),
                fill.get("fee"),
                fill.get("oid"),
            )

        hit_api_limit = False
        earliest_api_time: int | None = None

        cur = int(start_time)
        while cur < int(end_time):
            req_end = cur + int(window_ms)
            if end_time is not None:
                req_end = min(req_end, int(end_time))

            fills = await self.get_user_fills_by_time(
                address=address,
                start_time=cur,
                end_time=req_end,
                aggregate=aggregate,
            )

            if not fills:
                cur = req_end + 1
                continue

            # Process fills and track max time for pagination
            max_time: int | None = None
            batch_count = 0
            for f in fills:
                key = _dedupe_key(f)
                if key in seen:
                    continue
                seen.add(key)
                all_fills.append(f)
                batch_count += 1
                t = _to_int_ms(f.get("time"))
                if t is not None:
                    max_time = t if max_time is None else max(max_time, t)
                    if earliest_api_time is None or t < earliest_api_time:
                        earliest_api_time = t

            # Check if we hit the API limit and need to paginate within this window
            if len(fills) >= 2000 and max_time is not None:
                cur = max_time + 1
                logger.debug("Hit 2000 fill limit, paginating from %d", cur)
            else:
                cur = req_end + 1

            # Check global limit
            if len(all_fills) >= max_fills:
                hit_api_limit = True
                logger.warning(
                    "Hit Hyperliquid API limit of %d fills for %s.", max_fills, addr_lower
                )
                break

        # S3 fallback: if we hit the API limit and there's older data we need
        if (
            hit_api_limit
            and use_s3_fallback
            and earliest_api_time
            and earliest_api_time > start_time
        ):
            logger.info(
                "Attempting S3 fallback for %s: fetching %d to %d",
                addr_lower[:10],
                start_time,
                earliest_api_time,
            )
            try:
                from app.services.s3_fills import get_s3_fills_service

                s3_service = get_s3_fills_service()

                s3_fills = await s3_service.fetch_fills_range(
                    address=address,
                    start_time=start_time,
                    end_time=earliest_api_time - 1,
                )

                # Merge S3 fills (avoiding duplicates)
                for f in s3_fills:
                    key = _dedupe_key(f)
                    if key not in seen:
                        seen.add(key)
                        all_fills.append(f)

                logger.info("S3 fallback added %d fills for %s", len(s3_fills), addr_lower[:10])

            except Exception as e:
                logger.warning("S3 fallback failed for %s: %s", addr_lower[:10], e)

        all_fills.sort(key=lambda f: _to_int_ms(f.get("time")) or 0)

        # Store in cache
        if use_cache:
            self._fills_cache[cache_key] = FillsCache(data=all_fills, timestamp=time.time())

        return all_fills




    async def get_cumulative_volume(self, address: str) -> float:
        """Get cumulative trading volume from userRateLimit API"""
        try:
            data = await self._post({"type": "userRateLimit", "user": address.lower()})
            return float(data.get("cumVlm", 0))
        except Exception as e:
            logger.warning("Failed to get cumulative volume for %s: %s", address[:10], e)
            return 0.0

    async def get_spot_state(self, address: str) -> dict:
        """Get spot clearinghouse state"""
        return await self._post({"type": "spotClearinghouseState", "user": address.lower()})

    async def get_ledger_updates(self, address: str, start_time: int = 0) -> list[dict]:
        """Get non-funding ledger updates (deposits, withdrawals, transfers)"""
        return await self._post({
            "type": "userNonFundingLedgerUpdates",
            "user": address.lower(),
            "startTime": start_time,
        })

    async def is_high_frequency_account(self, address: str, threshold: float = 1_000_000_000) -> bool:
        """
        Detect if account is high-frequency (>$1B cumulative volume).
        High-frequency accounts may only return recent data from userFillsByTime API.
        """
        volume = await self.get_cumulative_volume(address)
        is_hf = volume > threshold
        if is_hf:
            logger.info("High-frequency account detected: %s, volume=$%.2fB", address[:10], volume / 1e9)
        return is_hf

    async def calculate_pnl_from_deposits(self, address: str) -> dict:
        """
        Calculate PnL using deposit/withdrawal method for high-frequency accounts.
        Formula: PnL = (Perp Account Value + Spot Value) - (Net Deposits)
        """
        addr_lower = address.lower()

        perp_state = await self.get_user_state(addr_lower)
        spot_state = await self.get_spot_state(addr_lower)
        ledger = await self.get_ledger_updates(addr_lower, start_time=0)

        perp_value = 0.0
        for path in [("marginSummary", "accountValue"), ("crossMarginSummary", "accountValue")]:
            cur = perp_state
            for key in path:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    cur = None
                    break
            if cur is not None:
                try:
                    perp_value = float(cur)
                    break
                except (TypeError, ValueError):
                    pass

        spot_value = 0.0
        spot_balances = spot_state.get("balances", [])
        for bal in spot_balances:
            try:
                token = bal.get("coin", "")
                total = float(bal.get("total", 0))
                if token in ("USDC", "USDT", "USD"):
                    spot_value += total
            except (TypeError, ValueError):
                pass

        total_deposits = 0.0
        total_withdrawals = 0.0
        for entry in ledger:
            delta = entry.get("delta", {})
            entry_type = delta.get("type", "")
            try:
                usdc = float(delta.get("usdc", 0))
                if entry_type == "deposit":
                    total_deposits += usdc
                elif entry_type == "withdraw":
                    total_withdrawals += usdc
            except (TypeError, ValueError):
                pass

        net_deposits = total_deposits - total_withdrawals
        current_value = perp_value + spot_value
        total_pnl = current_value - net_deposits

        logger.info(
            "PnL calculation for %s: perp=$%.2f, spot=$%.2f, deposits=$%.2f, withdrawals=$%.2f, pnl=$%.2f",
            addr_lower[:10], perp_value, spot_value, total_deposits, total_withdrawals, total_pnl
        )

        return {
            "total_pnl": total_pnl,
            "perp_value": perp_value,
            "spot_value": spot_value,
            "total_deposits": total_deposits,
            "total_withdrawals": total_withdrawals,
            "net_deposits": net_deposits,
            "method": "deposit_based",
        }

class SuperXService:
    """SuperX API service for trader stats"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.superx_api_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def get_trader_stats(self, address: str) -> dict:
        client = await self._get_client()
        resp = await client.get(
            f"{self.base_url}/v1/wallets/{address.lower()}",
        )
        resp.raise_for_status()
        return resp.json()


# Singleton instances
_hl_service: HyperliquidService | None = None
_superx_service: SuperXService | None = None


def get_hyperliquid_service() -> HyperliquidService:
    global _hl_service
    if _hl_service is None:
        _hl_service = HyperliquidService()
    return _hl_service


def get_superx_service() -> SuperXService:
    global _superx_service
    if _superx_service is None:
        _superx_service = SuperXService()
    return _superx_service
