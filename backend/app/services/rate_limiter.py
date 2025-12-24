import asyncio
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RequestWeight:
    """Request weight configuration based on CCXT's Hyperliquid implementation"""

    l2Book: int = 2
    allMids: int = 2
    clearinghouseState: int = 2
    spotClearinghouseState: int = 2
    orderStatus: int = 2
    exchangeStatus: int = 2
    candleSnapshot: int = 4
    userFills: int = 20
    userFillsByTime: int = 20
    portfolio: int = 20
    meta: int = 20
    metaAndAssetCtxs: int = 20
    default: int = 20


class RateLimiter:
    """
    Rate limiter with weighted requests for Hyperliquid API.
    Based on CCXT implementation: 1200 requests/minute with different weights per endpoint.
    """

    def __init__(
        self,
        max_weight_per_minute: int = 1200,
        min_interval_ms: int = 20,  # 50 req/s max (optimized for speed)
    ):
        self.max_weight_per_minute = max_weight_per_minute
        self.min_interval_ms = min_interval_ms
        self.weights = RequestWeight()

        self._weight_used: int = 0
        self._window_start: float = time.time()
        self._last_request: float = 0
        self._lock = asyncio.Lock()

    def get_weight(self, request_type: str) -> int:
        """Get weight for request type"""
        return getattr(self.weights, request_type, self.weights.default)

    async def acquire(self, request_type: str = "default") -> None:
        """Acquire permission to make a request, waiting if necessary"""
        weight = self.get_weight(request_type)

        async with self._lock:
            now = time.time()

            # Reset window if more than 60 seconds passed
            if now - self._window_start >= 60:
                self._weight_used = 0
                self._window_start = now
                logger.debug("Rate limiter window reset")

            # Check if we have capacity
            if self._weight_used + weight > self.max_weight_per_minute:
                # Wait for window reset
                wait_time = 60 - (now - self._window_start)
                if wait_time > 0:
                    logger.warning(
                        "Rate limit near capacity (%d/%d), waiting %.1fs",
                        self._weight_used,
                        self.max_weight_per_minute,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    self._weight_used = 0
                    self._window_start = time.time()

            # Enforce minimum interval between requests
            time_since_last = (time.time() - self._last_request) * 1000
            if time_since_last < self.min_interval_ms:
                wait_ms = self.min_interval_ms - time_since_last
                await asyncio.sleep(wait_ms / 1000)

            # Update counters
            self._weight_used += weight
            self._last_request = time.time()

            logger.debug(
                "Request acquired: type=%s, weight=%d, total=%d/%d",
                request_type,
                weight,
                self._weight_used,
                self.max_weight_per_minute,
            )

    @property
    def remaining_weight(self) -> int:
        """Get remaining weight in current window"""
        now = time.time()
        if now - self._window_start >= 60:
            return self.max_weight_per_minute
        return max(0, self.max_weight_per_minute - self._weight_used)

    @property
    def usage_pct(self) -> float:
        """Get usage percentage"""
        return (self._weight_used / self.max_weight_per_minute) * 100


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
