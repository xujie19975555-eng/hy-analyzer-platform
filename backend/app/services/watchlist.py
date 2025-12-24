"""
Watchlist service for managing tracked traders and background data sync.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import asc, desc
from sqlalchemy.orm import Session

from app.models.database import DataSyncLog, SessionLocal, TraderAnalysis, Watchlist
from app.services.fills_cache import get_fills_cache_service

logger = logging.getLogger(__name__)

# Background sync task handle
_sync_task: asyncio.Task | None = None
_sync_running = False


class WatchlistService:
    """Service for managing watchlist and background data sync"""

    def _get_db(self) -> Session:
        return SessionLocal()

    def add_to_watchlist(
        self,
        address: str,
        alias: str | None = None,
        notes: str | None = None,
        priority: int = 0,
        auto_update: bool = True,
    ) -> dict:
        """Add an address to the watchlist"""
        db = self._get_db()
        try:
            existing = db.query(Watchlist).filter(Watchlist.address == address.lower()).first()

            if existing:
                existing.alias = alias or existing.alias
                existing.notes = notes or existing.notes
                existing.priority = priority
                existing.auto_update = auto_update
                db.commit()
                return {
                    "address": existing.address,
                    "alias": existing.alias,
                    "status": "updated",
                }

            entry = Watchlist(
                address=address.lower(),
                alias=alias,
                notes=notes,
                priority=priority,
                auto_update=auto_update,
            )
            db.add(entry)
            db.commit()

            return {
                "address": entry.address,
                "alias": entry.alias,
                "status": "added",
            }
        finally:
            db.close()

    def remove_from_watchlist(self, address: str) -> bool:
        """Remove an address from the watchlist"""
        db = self._get_db()
        try:
            deleted = db.query(Watchlist).filter(Watchlist.address == address.lower()).delete()
            db.commit()
            return deleted > 0
        finally:
            db.close()

    def get_watchlist(
        self,
        sort_by: Literal["priority", "added_at", "last_updated", "alias"] = "priority",
        order: Literal["asc", "desc"] = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get all watchlist entries with analysis data"""
        db = self._get_db()
        try:
            sort_column = {
                "priority": Watchlist.priority,
                "added_at": Watchlist.added_at,
                "last_updated": Watchlist.last_updated,
                "alias": Watchlist.alias,
            }.get(sort_by, Watchlist.priority)

            order_func = desc if order == "desc" else asc

            entries = (
                db.query(Watchlist)
                .order_by(order_func(sort_column))
                .offset(offset)
                .limit(limit)
                .all()
            )

            results = []
            for entry in entries:
                analysis = (
                    db.query(TraderAnalysis).filter(TraderAnalysis.address == entry.address).first()
                )

                result = {
                    "address": entry.address,
                    "alias": entry.alias,
                    "notes": entry.notes,
                    "priority": entry.priority,
                    "auto_update": entry.auto_update,
                    "added_at": entry.added_at.isoformat() if entry.added_at else None,
                    "last_updated": entry.last_updated.isoformat() if entry.last_updated else None,
                    "total_cached_fills": entry.total_cached_fills,
                    "data_coverage": {
                        "start": entry.data_coverage_start,
                        "end": entry.data_coverage_end,
                        "start_date": datetime.fromtimestamp(
                            entry.data_coverage_start / 1000, tz=UTC
                        ).isoformat()
                        if entry.data_coverage_start
                        else None,
                        "end_date": datetime.fromtimestamp(
                            entry.data_coverage_end / 1000, tz=UTC
                        ).isoformat()
                        if entry.data_coverage_end
                        else None,
                    }
                    if entry.data_coverage_start
                    else None,
                    "analysis": None,
                }

                if analysis:
                    result["analysis"] = {
                        "account_value": analysis.account_value,
                        "roe_30d": analysis.roe_30d,
                        "ai_score": analysis.ai_score,
                        "ai_recommendation": analysis.ai_recommendation,
                        "analyzed_at": analysis.analyzed_at.isoformat()
                        if analysis.analyzed_at
                        else None,
                    }

                results.append(result)

            return results
        finally:
            db.close()

    def get_watchlist_entry(self, address: str) -> dict | None:
        """Get a single watchlist entry"""
        db = self._get_db()
        try:
            entry = db.query(Watchlist).filter(Watchlist.address == address.lower()).first()

            if not entry:
                return None

            return {
                "address": entry.address,
                "alias": entry.alias,
                "notes": entry.notes,
                "priority": entry.priority,
                "auto_update": entry.auto_update,
                "added_at": entry.added_at.isoformat() if entry.added_at else None,
                "last_updated": entry.last_updated.isoformat() if entry.last_updated else None,
                "total_cached_fills": entry.total_cached_fills,
                "data_coverage_start": entry.data_coverage_start,
                "data_coverage_end": entry.data_coverage_end,
            }
        finally:
            db.close()

    def is_in_watchlist(self, address: str) -> bool:
        """Check if address is in watchlist"""
        db = self._get_db()
        try:
            return (
                db.query(Watchlist).filter(Watchlist.address == address.lower()).first() is not None
            )
        finally:
            db.close()

    def get_addresses_for_sync(self) -> list[str]:
        """Get addresses that need background sync"""
        db = self._get_db()
        try:
            entries = (
                db.query(Watchlist)
                .filter(Watchlist.auto_update == True)
                .order_by(desc(Watchlist.priority))
                .all()
            )
            return [e.address for e in entries]
        finally:
            db.close()

    def get_sync_history(
        self,
        address: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get sync history logs"""
        db = self._get_db()
        try:
            query = db.query(DataSyncLog)
            if address:
                query = query.filter(DataSyncLog.address == address.lower())

            logs = query.order_by(desc(DataSyncLog.started_at)).limit(limit).all()

            return [
                {
                    "id": log.id,
                    "address": log.address,
                    "sync_type": log.sync_type,
                    "started_at": log.started_at.isoformat() if log.started_at else None,
                    "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                    "fills_added": log.fills_added,
                    "status": log.status,
                    "source": log.source,
                    "error_message": log.error_message,
                }
                for log in logs
            ]
        finally:
            db.close()

    def get_watchlist_count(self) -> int:
        """Get total count of watchlist entries"""
        db = self._get_db()
        try:
            return db.query(Watchlist).count()
        finally:
            db.close()


async def run_background_sync(interval_minutes: int = 60):
    """Background task to sync watchlist addresses periodically"""
    global _sync_running
    _sync_running = True

    watchlist_service = WatchlistService()
    fills_cache = get_fills_cache_service()

    logger.info("Starting background sync task (interval: %d minutes)", interval_minutes)

    while _sync_running:
        try:
            addresses = watchlist_service.get_addresses_for_sync()
            logger.info("Background sync: %d addresses to update", len(addresses))

            for address in addresses:
                if not _sync_running:
                    break

                try:
                    result = await fills_cache.sync_address(address, full_sync=False)
                    logger.info(
                        "Synced %s: %d new fills (total: %d)",
                        address[:10],
                        result.get("fills_added", 0),
                        result.get("total_fills", 0),
                    )
                except Exception as e:
                    logger.error("Sync failed for %s: %s", address[:10], e)

                # Small delay between addresses to avoid overloading S3
                await asyncio.sleep(2)

        except Exception as e:
            logger.error("Background sync error: %s", e)

        # Wait for next sync interval
        await asyncio.sleep(interval_minutes * 60)

    logger.info("Background sync task stopped")


def start_background_sync(interval_minutes: int = 60):
    """Start the background sync task"""
    global _sync_task
    if _sync_task is None or _sync_task.done():
        _sync_task = asyncio.create_task(run_background_sync(interval_minutes))
        logger.info("Background sync task started")
    return _sync_task


def stop_background_sync():
    """Stop the background sync task"""
    global _sync_running, _sync_task
    _sync_running = False
    if _sync_task and not _sync_task.done():
        _sync_task.cancel()
        logger.info("Background sync task stopped")


def is_background_sync_running() -> bool:
    """Check if background sync is running"""
    return _sync_running and _sync_task is not None and not _sync_task.done()


# Singleton
_watchlist_service: WatchlistService | None = None


def get_watchlist_service() -> WatchlistService:
    global _watchlist_service
    if _watchlist_service is None:
        _watchlist_service = WatchlistService()
    return _watchlist_service
