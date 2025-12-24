"""
Fills cache service for storing and retrieving historical fills data.

Features:
- Local SQLite cache to avoid repeated S3 downloads
- Incremental updates (only fetch new data)
- Smart data source selection (cache → S3 → API)
- Background sync support for watchlist addresses
"""

import logging
from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.database import DataSyncLog, FillsCache, SessionLocal, Watchlist
from app.services.hyperliquid import get_hyperliquid_service
from app.services.s3_fills import get_s3_fills_service

logger = logging.getLogger(__name__)

# S3 data earliest available date
S3_EARLIEST_DATE = datetime(2025, 3, 22, tzinfo=UTC)
S3_EARLIEST_MS = int(S3_EARLIEST_DATE.timestamp() * 1000)


def _fill_to_cache_record(fill: dict, address: str) -> dict:
    """Convert a fill dict to FillsCache fields"""
    fill_time = fill.get("time")
    if isinstance(fill_time, str):
        try:
            dt = datetime.fromisoformat(fill_time.replace("Z", "+00:00"))
            fill_time = int(dt.timestamp() * 1000)
        except ValueError:
            fill_time = 0

    fill_hash = (
        fill.get("hash") or fill.get("tid") or f"{fill_time}:{fill.get('coin')}:{fill.get('oid')}"
    )

    return {
        "address": address.lower(),
        "coin": fill.get("coin", ""),
        "fill_time": int(fill_time) if fill_time else 0,
        "fill_hash": fill_hash,
        "side": fill.get("side", ""),
        "px": str(fill.get("px", "")),
        "sz": str(fill.get("sz", "")),
        "oid": fill.get("oid"),
        "start_position": str(fill.get("startPosition", "")) if fill.get("startPosition") else None,
        "closed_pnl": str(fill.get("closedPnl", "")) if fill.get("closedPnl") else None,
        "fee": str(fill.get("fee", "")) if fill.get("fee") else None,
        "source": fill.get("_source", "api"),
        "raw_data": fill,
    }


def _cache_record_to_fill(record: FillsCache) -> dict:
    """Convert FillsCache record back to fill dict"""
    if record.raw_data:
        return record.raw_data

    return {
        "coin": record.coin,
        "time": record.fill_time,
        "hash": record.fill_hash,
        "side": record.side,
        "px": record.px,
        "sz": record.sz,
        "oid": record.oid,
        "startPosition": record.start_position,
        "closedPnl": record.closed_pnl,
        "fee": record.fee,
    }


class FillsCacheService:
    """Service for managing fills cache with incremental updates"""

    def __init__(self):
        self._s3_service = get_s3_fills_service()

    def _get_db(self) -> Session:
        return SessionLocal()

    def get_cached_fills(
        self,
        address: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict]:
        """Get fills from local cache"""
        db = self._get_db()
        try:
            query = db.query(FillsCache).filter(FillsCache.address == address.lower())

            if start_time:
                query = query.filter(FillsCache.fill_time >= start_time)
            if end_time:
                query = query.filter(FillsCache.fill_time <= end_time)

            query = query.order_by(FillsCache.fill_time.asc())
            records = query.all()

            return [_cache_record_to_fill(r) for r in records]
        finally:
            db.close()

    def get_cache_coverage(self, address: str) -> dict:
        """Get information about cached data for an address"""
        db = self._get_db()
        try:
            result = (
                db.query(
                    func.count(FillsCache.id).label("count"),
                    func.min(FillsCache.fill_time).label("earliest"),
                    func.max(FillsCache.fill_time).label("latest"),
                )
                .filter(FillsCache.address == address.lower())
                .first()
            )

            return {
                "address": address,
                "total_fills": result.count if result else 0,
                "earliest_time": result.earliest if result else None,
                "latest_time": result.latest if result else None,
                "earliest_date": datetime.fromtimestamp(result.earliest / 1000, tz=UTC).isoformat()
                if result and result.earliest
                else None,
                "latest_date": datetime.fromtimestamp(result.latest / 1000, tz=UTC).isoformat()
                if result and result.latest
                else None,
            }
        finally:
            db.close()

    def save_fills_to_cache(self, address: str, fills: list[dict]) -> int:
        """Save fills to cache, skip duplicates"""
        if not fills:
            return 0

        db = self._get_db()
        added = 0
        try:
            for fill in fills:
                record_data = _fill_to_cache_record(fill, address)
                existing = (
                    db.query(FillsCache)
                    .filter(FillsCache.fill_hash == record_data["fill_hash"])
                    .first()
                )

                if not existing:
                    db.add(FillsCache(**record_data))
                    added += 1

            db.commit()
            logger.info("Saved %d new fills to cache for %s", added, address[:10])
            return added
        except Exception as e:
            db.rollback()
            logger.error("Failed to save fills to cache: %s", e)
            return 0
        finally:
            db.close()

    async def fetch_and_cache_fills(
        self,
        address: str,
        start_time: int,
        end_time: int | None = None,
        use_api_fallback: bool = True,
    ) -> list[dict]:
        """
        Fetch fills from S3/API and cache them locally.

        Strategy:
        1. Check what's already in cache
        2. Fetch missing ranges from S3 (2025-03-22+)
        3. Optionally fetch from API for data before S3 coverage
        """
        if end_time is None:
            end_time = int(datetime.now(UTC).timestamp() * 1000)

        address_lower = address.lower()

        # Check existing cache coverage
        coverage = self.get_cache_coverage(address)
        logger.info(
            "Cache coverage for %s: %d fills, %s to %s",
            address[:10],
            coverage["total_fills"],
            coverage.get("earliest_date", "N/A"),
            coverage.get("latest_date", "N/A"),
        )

        # Determine what needs to be fetched
        fills_to_add = []

        # Fetch from S3 for ranges >= 2025-03-22
        s3_start = max(start_time, S3_EARLIEST_MS)
        if s3_start < end_time:
            # Check if we need to fetch new data from S3
            latest_cached = coverage["latest_time"] or 0

            if end_time > latest_cached:
                # Fetch new data since last cache
                fetch_start = max(s3_start, latest_cached + 1) if latest_cached else s3_start
                logger.info(
                    "Fetching S3 data from %s",
                    datetime.fromtimestamp(fetch_start / 1000, tz=UTC).isoformat(),
                )

                s3_fills = await self._s3_service.fetch_fills_range(address, fetch_start, end_time)
                fills_to_add.extend(s3_fills)
                logger.info("Fetched %d fills from S3", len(s3_fills))

        # Optionally fetch from API for data before S3 coverage
        if use_api_fallback and start_time < S3_EARLIEST_MS:
            earliest_cached = coverage["earliest_time"]
            api_end = min(end_time, S3_EARLIEST_MS)

            # Only fetch from API if we don't have cached data for this period
            if not earliest_cached or earliest_cached > start_time:
                logger.info("Fetching API data for period before S3 coverage")
                try:
                    hl_service = get_hyperliquid_service()
                    api_fills = await hl_service.get_user_fills_windowed(
                        address, start_time=start_time, end_time=api_end, aggregate=False
                    )
                    for fill in api_fills:
                        fill["_source"] = "api"
                    fills_to_add.extend(api_fills)
                    logger.info("Fetched %d fills from API", len(api_fills))
                except Exception as e:
                    logger.warning("API fill fetch failed: %s", e)

        # Save new fills to cache
        if fills_to_add:
            self.save_fills_to_cache(address, fills_to_add)

        # Return all cached fills for the requested range
        return self.get_cached_fills(address, start_time, end_time)

    async def sync_address(self, address: str, full_sync: bool = False) -> dict:
        """
        Sync fills for an address (for watchlist background updates).

        Args:
            address: Wallet address to sync
            full_sync: If True, fetch all available history; otherwise incremental

        Returns:
            Sync result with stats
        """
        db = self._get_db()
        now_ms = int(datetime.now(UTC).timestamp() * 1000)

        # Create sync log
        sync_log = DataSyncLog(
            address=address.lower(),
            sync_type="full" if full_sync else "incremental",
            time_range_end=now_ms,
        )
        db.add(sync_log)
        db.commit()
        sync_id = sync_log.id

        try:
            coverage = self.get_cache_coverage(address)

            if full_sync or not coverage["total_fills"]:
                # Full sync - fetch all available data
                start_time = S3_EARLIEST_MS
            else:
                # Incremental - only fetch new data
                start_time = (coverage["latest_time"] or S3_EARLIEST_MS) + 1

            sync_log.time_range_start = start_time

            # Fetch and cache
            fills = await self.fetch_and_cache_fills(
                address, start_time, now_ms, use_api_fallback=full_sync
            )

            # Update sync log
            new_coverage = self.get_cache_coverage(address)
            fills_added = new_coverage["total_fills"] - coverage["total_fills"]

            sync_log.fills_added = max(0, fills_added)
            sync_log.completed_at = datetime.utcnow()
            sync_log.status = "completed"
            sync_log.source = "s3" if start_time >= S3_EARLIEST_MS else "mixed"
            db.commit()

            # Update watchlist entry if exists
            watchlist_entry = (
                db.query(Watchlist).filter(Watchlist.address == address.lower()).first()
            )
            if watchlist_entry:
                watchlist_entry.last_updated = datetime.utcnow()
                watchlist_entry.last_fill_time = new_coverage["latest_time"]
                watchlist_entry.total_cached_fills = new_coverage["total_fills"]
                watchlist_entry.data_coverage_start = new_coverage["earliest_time"]
                watchlist_entry.data_coverage_end = new_coverage["latest_time"]
                db.commit()

            return {
                "address": address,
                "sync_type": sync_log.sync_type,
                "fills_added": fills_added,
                "total_fills": new_coverage["total_fills"],
                "coverage": new_coverage,
                "status": "completed",
            }

        except Exception as e:
            sync_log.status = "failed"
            sync_log.error_message = str(e)
            sync_log.completed_at = datetime.utcnow()
            db.commit()
            logger.error("Sync failed for %s: %s", address, e)
            return {
                "address": address,
                "status": "failed",
                "error": str(e),
            }
        finally:
            db.close()

    def clear_cache(self, address: str | None = None) -> int:
        """Clear cache for an address or all addresses"""
        db = self._get_db()
        try:
            if address:
                deleted = (
                    db.query(FillsCache).filter(FillsCache.address == address.lower()).delete()
                )
            else:
                deleted = db.query(FillsCache).delete()
            db.commit()
            logger.info("Cleared %d cached fills", deleted)
            return deleted
        finally:
            db.close()


# Singleton
_fills_cache_service: FillsCacheService | None = None


def get_fills_cache_service() -> FillsCacheService:
    global _fills_cache_service
    if _fills_cache_service is None:
        _fills_cache_service = FillsCacheService()
    return _fills_cache_service
