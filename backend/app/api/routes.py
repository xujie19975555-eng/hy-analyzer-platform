import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.schemas import (
    AnalysisStatus,
    BackgroundSyncStatus,
    BacktestRequest,
    BacktestResult,
    CacheCoverage,
    FullAnalysisRequest,
    FullAnalysisResult,
    SyncLogItem,
    SyncResult,
    TraderHistoryItem,
    TraderStats,
    WalletAddress,
    WatchlistAddRequest,
    WatchlistItem,
    WatchlistUpdateRequest,
)
from app.services.analyzer import compute_trade_metrics_timeframes, parse_superx_stats
from app.services.backtest import period_to_time_range_ms, run_backtest
from app.services.fills_cache import get_fills_cache_service
from app.services.full_analysis import (
    delete_analysis,
    get_analysis_history,
    get_analysis_status,
    get_cached_analysis,
    is_analysis_running,
    is_cache_valid,
    run_full_analysis,
)
from app.services.hyperliquid import (
    HyperliquidService,
    SuperXService,
    get_hyperliquid_service,
    get_superx_service,
)
from app.services.watchlist import (
    get_watchlist_service,
    is_background_sync_running,
    start_background_sync,
    stop_background_sync,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["traders"])


@router.get("/traders/{address}/stats", response_model=TraderStats)
async def get_trader_stats(
    address: str,
    superx: SuperXService = Depends(get_superx_service),
    hl: HyperliquidService = Depends(get_hyperliquid_service),
):
    """Get trader statistics from SuperX + enriched with Hyperliquid data"""
    try:
        wallet = WalletAddress(address=address)
        data = await superx.get_trader_stats(wallet.address)
        base_stats = parse_superx_stats(wallet.address, data)

        account_value = await hl.get_account_value(wallet.address)

        start_ms, end_ms = period_to_time_range_ms("90d")
        fills_90d = await hl.get_user_fills_windowed(
            wallet.address, start_time=start_ms, end_time=end_ms, aggregate=False
        )
        metrics = compute_trade_metrics_timeframes(list(fills_90d), now_ms=end_ms)

        return base_stats.model_copy(update={"account_value": account_value, **metrics})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to fetch stats for %s", address)
        raise HTTPException(status_code=500, detail="Failed to fetch trader stats")


@router.get("/traders/{address}/portfolio")
async def get_trader_portfolio(
    address: str, hl: HyperliquidService = Depends(get_hyperliquid_service)
):
    """Get trader portfolio data from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        return await hl.get_portfolio(wallet.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to fetch portfolio for %s", address)
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio")


@router.get("/traders/{address}/state")
async def get_trader_state(address: str, hl: HyperliquidService = Depends(get_hyperliquid_service)):
    """Get current trader state (positions, margin) from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        return await hl.get_user_state(wallet.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to fetch state for %s", address)
        raise HTTPException(status_code=500, detail="Failed to fetch state")


@router.get("/traders/{address}/trades")
async def get_trader_trades(
    address: str, hl: HyperliquidService = Depends(get_hyperliquid_service)
):
    """Get trader's recent trades from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        fills = await hl.get_user_fills(wallet.address)
        return {"address": wallet.address, "total_count": len(fills), "trades": fills}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Failed to fetch trades for %s", address)
        raise HTTPException(status_code=500, detail="Failed to fetch trades")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hy-analyzer-api"}


@router.get("/version")
async def get_version():
    """Get API version"""
    return {"version": "0.1.0", "api": "v1", "name": "HY Analyzer Platform"}


@router.post("/traders/{address}/backtest", response_model=BacktestResult)
async def backtest_trader(
    address: str,
    request: BacktestRequest,
    hl: HyperliquidService = Depends(get_hyperliquid_service),
):
    """
    Backtest a trader by scaling realized PnL from historical fills.

    - **capital**: Your custom starting capital (USDT)
    - **period**: Time period (7d/30d/90d/180d/all)

    Trades with scaled notional < 11 USDT are skipped.
    """
    try:
        wallet = WalletAddress(address=address)
        account_value = await hl.get_account_value(wallet.address)
        if account_value is None or account_value <= 0:
            raise HTTPException(
                status_code=502, detail="Failed to get account_value from Hyperliquid"
            )

        start_ms, end_ms = period_to_time_range_ms(request.period)
        fills = await hl.get_user_fills_windowed(
            wallet.address,
            start_time=start_ms,
            end_time=end_ms,
            aggregate=False,
        )

        return run_backtest(
            address=wallet.address,
            period=request.period,
            capital=float(request.capital),
            account_value=float(account_value),
            fills=fills,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to backtest for %s", address)
        raise HTTPException(status_code=500, detail="Failed to run backtest")


@router.post("/analyze/{address}", response_model=FullAnalysisResult)
async def analyze_trader(
    address: str,
    request: FullAnalysisRequest = FullAnalysisRequest(),
):
    """
    Run complete trader analysis including stats, backtest, and AI evaluation.

    This endpoint:
    1. Checks if valid cached analysis exists (default: 1 hour validity)
    2. Returns cached result if valid and force_refresh=False
    3. Otherwise fetches fresh data from SuperX and Hyperliquid
    4. Runs backtests for 7d/30d/90d periods
    5. Generates AI evaluation for copy-trading
    6. Saves results to database

    Parameters:
    - **capital**: Starting capital for backtest (USDT)
    - **force_refresh**: If True, ignore cache and re-analyze
    """
    try:
        wallet = WalletAddress(address=address)

        # Check cache first (unless force_refresh)
        if not request.force_refresh and is_cache_valid(wallet.address):
            cached = get_cached_analysis(wallet.address)
            if cached:
                logger.info("Returning cached analysis for %s", wallet.address)
                return cached

        if await is_analysis_running():
            raise HTTPException(status_code=429, detail="Another analysis is running. Please wait.")
        return await run_full_analysis(wallet.address, request.capital)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Full analysis failed for %s", address)
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/analyze/{address}/status", response_model=AnalysisStatus)
async def get_trader_analysis_status(address: str):
    """Get the current analysis status for a trader"""
    try:
        wallet = WalletAddress(address=address)
        status = await get_analysis_status(wallet.address)
        if not status:
            cached = get_cached_analysis(wallet.address)
            if cached:
                return AnalysisStatus(
                    address=wallet.address, status="completed", progress=100, current_step="已完成"
                )
            return AnalysisStatus(
                address=wallet.address, status="queued", progress=0, current_step="未开始"
            )
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analyze/{address}/result", response_model=FullAnalysisResult)
async def get_trader_analysis_result(address: str):
    """Get cached analysis result for a trader"""
    try:
        wallet = WalletAddress(address=address)
        result = get_cached_analysis(wallet.address)
        if not result:
            raise HTTPException(status_code=404, detail="No analysis found for this trader")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise


@router.get("/history", response_model=list[TraderHistoryItem])
async def get_traders_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: Literal["ai_score", "analyzed_at", "roe_30d", "account_value"] = Query("ai_score"),
    order: Literal["asc", "desc"] = Query("desc"),
    min_score: float | None = Query(None, ge=0, le=100),
    recommendation: str | None = Query(None),
):
    """
    Get list of previously analyzed traders.

    - **sort_by**: Sort field (default: ai_score for best traders first)
    - **order**: Sort order (default: desc)
    - **min_score**: Minimum AI score filter
    - **recommendation**: Filter by recommendation (strong_follow, follow, neutral, avoid, strong_avoid)
    """
    return get_analysis_history(
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        order=order,
        min_score=min_score,
        recommendation=recommendation,
    )


@router.get("/analysis/running")
async def check_analysis_running():
    """Check if any analysis is currently running"""
    return {"running": await is_analysis_running()}


@router.delete("/history/{address}")
async def delete_trader_analysis(address: str):
    """Delete a trader's analysis from history"""
    try:
        wallet = WalletAddress(address=address)
        deleted = delete_analysis(wallet.address)
        if not deleted:
            raise HTTPException(status_code=404, detail="Analysis not found for this address")
        return {"address": wallet.address, "status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# === Watchlist Endpoints ===


@router.get("/watchlist", response_model=list[WatchlistItem])
async def get_watchlist(
    sort_by: Literal["priority", "added_at", "last_updated", "alias"] = Query("priority"),
    order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Get all watchlist entries with analysis data.

    - **sort_by**: Sort field (default: priority)
    - **order**: Sort order (default: desc for highest priority first)
    """
    service = get_watchlist_service()
    return service.get_watchlist(sort_by=sort_by, order=order, limit=limit, offset=offset)


@router.post("/watchlist", response_model=dict)
async def add_to_watchlist(request: WatchlistAddRequest):
    """
    Add an address to the watchlist.

    Once added, fills data will be automatically synced in the background.
    """
    service = get_watchlist_service()
    result = service.add_to_watchlist(
        address=request.address,
        alias=request.alias,
        notes=request.notes,
        priority=request.priority,
        auto_update=request.auto_update,
    )

    # Trigger initial sync
    if result["status"] == "added":
        fills_cache = get_fills_cache_service()
        try:
            sync_result = await fills_cache.sync_address(request.address, full_sync=True)
            result["sync"] = sync_result
        except Exception as e:
            logger.warning("Initial sync failed for %s: %s", request.address, e)
            result["sync_error"] = str(e)

    return result


@router.get("/watchlist/{address}")
async def get_watchlist_entry(address: str):
    """Get a single watchlist entry"""
    try:
        wallet = WalletAddress(address=address)
        service = get_watchlist_service()
        entry = service.get_watchlist_entry(wallet.address)
        if not entry:
            raise HTTPException(status_code=404, detail="Address not in watchlist")
        return entry
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/watchlist/{address}")
async def update_watchlist_entry(address: str, request: WatchlistUpdateRequest):
    """Update a watchlist entry"""
    try:
        wallet = WalletAddress(address=address)
        service = get_watchlist_service()

        if not service.is_in_watchlist(wallet.address):
            raise HTTPException(status_code=404, detail="Address not in watchlist")

        update_data = request.model_dump(exclude_none=True)
        if update_data:
            result = service.add_to_watchlist(
                address=wallet.address,
                alias=update_data.get("alias"),
                notes=update_data.get("notes"),
                priority=update_data.get("priority", 0),
                auto_update=update_data.get("auto_update", True),
            )
            return result

        return {"address": wallet.address, "status": "unchanged"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/watchlist/{address}")
async def remove_from_watchlist(address: str):
    """Remove an address from the watchlist"""
    try:
        wallet = WalletAddress(address=address)
        service = get_watchlist_service()
        removed = service.remove_from_watchlist(wallet.address)
        if not removed:
            raise HTTPException(status_code=404, detail="Address not in watchlist")
        return {"address": wallet.address, "status": "removed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/watchlist/{address}/sync", response_model=SyncResult)
async def sync_watchlist_address(
    address: str,
    full: bool = Query(False, description="Full sync (fetch all history) vs incremental"),
):
    """
    Manually trigger data sync for a watchlist address.

    - **full**: If True, fetches all available history; otherwise only new data
    """
    try:
        wallet = WalletAddress(address=address)
        fills_cache = get_fills_cache_service()
        result = await fills_cache.sync_address(wallet.address, full_sync=full)
        return SyncResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Sync failed for %s", address)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


# === Cache Endpoints ===


@router.get("/cache/{address}/coverage", response_model=CacheCoverage)
async def get_cache_coverage(address: str):
    """Get cache coverage information for an address"""
    try:
        wallet = WalletAddress(address=address)
        fills_cache = get_fills_cache_service()
        return fills_cache.get_cache_coverage(wallet.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/cache/{address}/fills")
async def get_cached_fills(
    address: str,
    start_time: int | None = Query(None, description="Start timestamp in ms"),
    end_time: int | None = Query(None, description="End timestamp in ms"),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Get cached fills for an address"""
    try:
        wallet = WalletAddress(address=address)
        fills_cache = get_fills_cache_service()
        fills = fills_cache.get_cached_fills(wallet.address, start_time, end_time)
        return {
            "address": wallet.address,
            "total": len(fills),
            "fills": fills[:limit],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/cache/{address}")
async def clear_address_cache(address: str):
    """Clear cached fills for an address"""
    try:
        wallet = WalletAddress(address=address)
        fills_cache = get_fills_cache_service()
        deleted = fills_cache.clear_cache(wallet.address)
        return {"address": wallet.address, "deleted": deleted}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# === Sync Management Endpoints ===


@router.get("/sync/status", response_model=BackgroundSyncStatus)
async def get_sync_status():
    """Get background sync status"""
    service = get_watchlist_service()
    addresses = service.get_addresses_for_sync()
    return BackgroundSyncStatus(
        running=is_background_sync_running(),
        watchlist_count=service.get_watchlist_count(),
        next_sync_addresses=addresses[:10],
    )


@router.post("/sync/start")
async def start_sync(
    interval: int = Query(60, ge=5, le=1440, description="Sync interval in minutes"),
):
    """Start background sync task"""
    start_background_sync(interval)
    return {"status": "started", "interval_minutes": interval}


@router.post("/sync/stop")
async def stop_sync():
    """Stop background sync task"""
    stop_background_sync()
    return {"status": "stopped"}


@router.get("/sync/logs", response_model=list[SyncLogItem])
async def get_sync_logs(
    address: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get sync history logs"""
    service = get_watchlist_service()
    addr = None
    if address:
        try:
            wallet = WalletAddress(address=address)
            addr = wallet.address
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return service.get_sync_history(address=addr, limit=limit)
