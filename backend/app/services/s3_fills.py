"""
S3-based historical fills fetcher for Hyperliquid.

Bypasses the 10K fills limit by directly accessing multiple S3 data sources:
- s3://hl-mainnet-node-data/node_trades (from 2025-03-22, contains both sides)
- s3://hl-mainnet-node-data/node_fills (from 2025-05-25, API format)
- s3://hl-mainnet-node-data/node_fills_by_block (from 2025-07-27, current format)

Data is organized by date/hour: {prefix}/hourly/{YYYYMMDD}/{HH}.lz4

NOTE: Hyperliquid S3 bucket uses "requester pays" model.
You need AWS credentials and will be charged for data transfer (~$0.09/GB).
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import boto3
from botocore.config import Config

from app.config import get_settings

logger = logging.getLogger(__name__)

# Thread pool for S3 operations (boto3 is not async-native)
_executor = ThreadPoolExecutor(max_workers=8)

# S3 bucket info
BUCKET_NAME = "hl-mainnet-node-data"

# Data source configurations with availability dates
DATA_SOURCES = {
    "node_fills_by_block": {
        "prefix": "node_fills_by_block/hourly",
        "start_date": datetime(2025, 7, 27, tzinfo=UTC),
        "priority": 1,  # Highest priority (newest format)
        "format": "fills",
    },
    "node_fills": {
        "prefix": "node_fills/hourly",
        "start_date": datetime(2025, 5, 25, tzinfo=UTC),
        "priority": 2,
        "format": "fills",
    },
    "node_trades": {
        "prefix": "node_trades/hourly",
        "start_date": datetime(2025, 3, 22, tzinfo=UTC),
        "priority": 3,  # Lowest priority but oldest data
        "format": "trades",
    },
}


def _parse_iso_time(time_str: str) -> int:
    """Parse ISO format time string to milliseconds timestamp"""
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, AttributeError):
        return 0


def _convert_trade_to_fill(trade: dict, side_info: dict, is_buyer: bool) -> dict:
    """Convert node_trades format to fills format for a specific side"""
    time_ms = _parse_iso_time(trade.get("time", ""))

    return {
        "coin": trade.get("coin"),
        "px": trade.get("px"),
        "sz": trade.get("sz"),
        "side": "B" if is_buyer else "A",
        "time": time_ms,
        "hash": trade.get("hash"),
        "oid": side_info.get("oid"),
        "user": side_info.get("user"),
        "startPosition": side_info.get("start_pos"),
        "cloid": side_info.get("cloid"),
        "twapId": side_info.get("twap_id"),
        # Mark as converted from trades for debugging
        "_source": "node_trades",
    }


class S3FillsService:
    """Fetches historical fills from Hyperliquid S3 archive with multi-source support"""

    def __init__(self):
        settings = get_settings()
        self._enabled = bool(settings.aws_access_key_id and settings.aws_secret_access_key)

        if self._enabled:
            self._s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
                config=Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    connect_timeout=10,
                    read_timeout=60,
                ),
            )
            logger.info("S3FillsService initialized with AWS credentials")
        else:
            self._s3 = None
            logger.warning(
                "AWS credentials not configured. S3 historical fills disabled. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"
            )

        self._lz4_available = self._check_lz4()

    def _check_lz4(self) -> bool:
        try:
            import lz4.frame

            return True
        except ImportError:
            logger.warning("lz4 not installed, S3 compressed data cannot be read")
            return False

    def _get_best_source_for_date(self, dt: datetime) -> str | None:
        """Determine the best data source for a given date"""
        available = []
        for name, config in DATA_SOURCES.items():
            if dt >= config["start_date"]:
                available.append((config["priority"], name))

        if not available:
            return None

        available.sort()
        return available[0][1]

    def _check_file_exists(self, key: str) -> bool:
        """Check if S3 file exists"""
        if not self._enabled or not self._s3:
            return False
        try:
            self._s3.head_object(
                Bucket=BUCKET_NAME,
                Key=key,
                RequestPayer="requester",
            )
            return True
        except Exception:
            return False

    def _download_file(self, key: str) -> bytes | None:
        """Download and decompress S3 file"""
        if not self._enabled or not self._s3:
            return None

        try:
            response = self._s3.get_object(
                Bucket=BUCKET_NAME,
                Key=key,
                RequestPayer="requester",
            )
            content = response["Body"].read()

            if key.endswith(".lz4"):
                if not self._lz4_available:
                    return None
                import lz4.frame

                content = lz4.frame.decompress(content)

            return content
        except Exception as e:
            logger.debug("Failed to download S3 file %s: %s", key, e)
            return None

    def _parse_fills_file(self, content: bytes, target_address: str) -> list[dict]:
        """Parse node_fills or node_fills_by_block format"""
        fills = []
        target_lower = target_address.lower()

        for line in content.decode("utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    # Single fill or batched block format
                    if "events" in data:
                        # Batched format: {local_time, block_time, block_number, events}
                        for event in data.get("events", []):
                            if isinstance(event, dict):
                                user = event.get("user", "").lower()
                                if user == target_lower:
                                    fills.append(event)
                    else:
                        user = data.get("user", "").lower()
                        if user == target_lower:
                            fills.append(data)
                elif isinstance(data, list):
                    for fill in data:
                        if isinstance(fill, dict):
                            user = fill.get("user", "").lower()
                            if user == target_lower:
                                fills.append(fill)
            except json.JSONDecodeError:
                continue

        return fills

    def _parse_trades_file(self, content: bytes, target_address: str) -> list[dict]:
        """Parse node_trades format and convert to fills"""
        fills = []
        target_lower = target_address.lower()

        for line in content.decode("utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    trades_to_process = [data]
                elif isinstance(data, list):
                    trades_to_process = data
                else:
                    continue

                for trade in trades_to_process:
                    if not isinstance(trade, dict):
                        continue

                    side_info_list = trade.get("side_info", [])
                    if len(side_info_list) >= 2:
                        # side_info[0] = maker/seller, side_info[1] = taker/buyer
                        # Check if either side matches target address
                        for i, side_info in enumerate(side_info_list[:2]):
                            if isinstance(side_info, dict):
                                user = side_info.get("user", "").lower()
                                if user == target_lower:
                                    is_buyer = i == 1
                                    fill = _convert_trade_to_fill(trade, side_info, is_buyer)
                                    fills.append(fill)
            except json.JSONDecodeError:
                continue

        return fills

    def _fetch_hour_data_sync(
        self,
        date_str: str,
        hour: int,
        target_address: str,
    ) -> list[dict]:
        """Fetch fills for a specific hour, trying multiple sources"""
        dt = datetime.strptime(date_str, "%Y%m%d").replace(hour=hour, tzinfo=UTC)

        # Try sources in priority order
        sources_to_try = []
        for name, config in DATA_SOURCES.items():
            if dt >= config["start_date"]:
                sources_to_try.append((config["priority"], name, config))

        sources_to_try.sort()

        for _, source_name, config in sources_to_try:
            key = f"{config['prefix']}/{date_str}/{hour}.lz4"

            content = self._download_file(key)
            if content is None:
                continue

            if config["format"] == "fills":
                fills = self._parse_fills_file(content, target_address)
            else:
                fills = self._parse_trades_file(content, target_address)

            if fills:
                logger.debug(
                    "Found %d fills from %s for %s/%d", len(fills), source_name, date_str, hour
                )
                return fills

        return []

    async def fetch_fills_range(
        self,
        address: str,
        start_time: int,
        end_time: int | None = None,
    ) -> list[dict]:
        """
        Fetch fills for an address within a time range from S3.

        Automatically selects the best available data source for each time period:
        - node_fills_by_block: 2025-07-27+
        - node_fills: 2025-05-25+
        - node_trades: 2025-03-22+

        Args:
            address: Wallet address
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds (default: now)

        Returns:
            List of fill records matching the address
        """
        if not self._enabled:
            logger.warning("S3 fills disabled - AWS credentials not configured")
            return []

        if end_time is None:
            end_time = int(datetime.now(UTC).timestamp() * 1000)

        start_dt = datetime.fromtimestamp(start_time / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_time / 1000, tz=UTC)

        # Earliest available data
        earliest_available = min(c["start_date"] for c in DATA_SOURCES.values())
        if start_dt < earliest_available:
            logger.info(
                "Requested start %s is before earliest S3 data %s, adjusting",
                start_dt.isoformat(),
                earliest_available.isoformat(),
            )
            start_dt = earliest_available

        # Generate hours to query
        hours_to_query: list[tuple[str, int]] = []
        current = start_dt.replace(minute=0, second=0, microsecond=0)
        while current <= end_dt:
            date_str = current.strftime("%Y%m%d")
            hours_to_query.append((date_str, current.hour))
            current += timedelta(hours=1)

        logger.info(
            "S3 fills query: address=%s, range=%s to %s, hours=%d",
            address[:10],
            start_dt.isoformat(),
            end_dt.isoformat(),
            len(hours_to_query),
        )

        # Fetch in parallel
        loop = asyncio.get_event_loop()
        all_fills: list[dict] = []

        batch_size = 24
        for i in range(0, len(hours_to_query), batch_size):
            batch = hours_to_query[i : i + batch_size]

            tasks = [
                loop.run_in_executor(
                    _executor,
                    self._fetch_hour_data_sync,
                    date_str,
                    hour,
                    address,
                )
                for date_str, hour in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_fills.extend(result)
                elif isinstance(result, Exception):
                    logger.debug("S3 fetch error: %s", result)

        # Filter by exact time range and deduplicate
        filtered = []
        seen: set[str] = set()
        for fill in all_fills:
            fill_time = fill.get("time")
            if fill_time is None:
                continue
            try:
                t = int(fill_time)
            except (TypeError, ValueError):
                continue

            if start_time <= t <= end_time:
                dedupe_key = fill.get("hash") or f"{t}:{fill.get('coin')}:{fill.get('oid')}"
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    filtered.append(fill)

        filtered.sort(key=lambda f: int(f.get("time", 0)))

        logger.info("S3 fills result: found %d fills for %s", len(filtered), address[:10])
        return filtered

    async def check_availability(self) -> dict:
        """Check S3 data source availability"""
        if not self._enabled:
            return {"enabled": False, "sources": {}}

        result = {"enabled": True, "sources": {}}
        loop = asyncio.get_event_loop()

        for name, config in DATA_SOURCES.items():
            # Check yesterday's data
            yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y%m%d")
            key = f"{config['prefix']}/{yesterday}/12.lz4"

            exists = await loop.run_in_executor(
                _executor,
                self._check_file_exists,
                key,
            )

            result["sources"][name] = {
                "available": exists,
                "start_date": config["start_date"].strftime("%Y-%m-%d"),
                "format": config["format"],
            }

        return result

    def get_data_coverage(self) -> dict:
        """Get information about data coverage"""
        return {
            "earliest_date": "2025-03-22",
            "sources": {
                name: {
                    "start_date": config["start_date"].strftime("%Y-%m-%d"),
                    "format": config["format"],
                }
                for name, config in DATA_SOURCES.items()
            },
            "note": "Data before 2025-03-22 requires running a Hyperliquid node",
        }

    @property
    def enabled(self) -> bool:
        return self._enabled


# Singleton
_s3_service: S3FillsService | None = None


def get_s3_fills_service() -> S3FillsService:
    global _s3_service
    if _s3_service is None:
        _s3_service = S3FillsService()
    return _s3_service
