import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

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
    reset_analysis_status,
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
    return {"version": "0.2.0", "api": "v1", "name": "HY Analyzer Platform"}


@router.post("/traders/{address}/backtest", response_model=BacktestResult)
async def backtest_trader(
    address: str,
    request: BacktestRequest,
    hl: HyperliquidService = Depends(get_hyperliquid_service),
):
    """
    Backtest a trader by scaling realized PnL from historical fills.
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
    """
    try:
        wallet = WalletAddress(address=address)

        if not request.force_refresh and is_cache_valid(wallet.address):
            cached = get_cached_analysis(wallet.address)
            if cached:
                logger.info("Returning cached analysis for %s", wallet.address)
                return cached

        if await is_analysis_running():
            raise HTTPException(status_code=429, detail="Another analysis is running. Please wait.")
        return await run_full_analysis(
            wallet.address, request.capital, use_full_history=request.use_full_history
        )
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


@router.get("/analyze/{address}/stream")
async def stream_analysis_status(address: str):
    """
    Stream analysis status updates using Server-Sent Events (SSE).

    Use this endpoint for real-time progress updates during analysis.
    Connect with EventSource in the frontend:

    ```javascript
    const es = new EventSource('/api/v1/analyze/0x.../stream');
    es.onmessage = (e) => {
        const status = JSON.parse(e.data);
        console.log(status.progress, status.current_step);
        if (status.status === 'completed' || status.status === 'failed') {
            es.close();
        }
    };
    ```
    """
    try:
        wallet = WalletAddress(address=address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def event_generator():
        last_progress = -1
        retry_count = 0
        max_retries = 600  # 10 minutes max (600 * 1 second)

        while retry_count < max_retries:
            status = await get_analysis_status(wallet.address)

            if status:
                # Only send if progress changed
                if status.progress != last_progress or status.status in ("completed", "failed"):
                    last_progress = status.progress
                    data = {
                        "address": status.address,
                        "status": status.status,
                        "progress": status.progress,
                        "current_step": status.current_step,
                        "error": status.error,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    if status.status in ("completed", "failed"):
                        break
            else:
                # Check if analysis is completed in cache
                cached = get_cached_analysis(wallet.address)
                if cached:
                    data = {
                        "address": wallet.address,
                        "status": "completed",
                        "progress": 100,
                        "current_step": "已完成",
                        "error": None,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    break

            await asyncio.sleep(1)
            retry_count += 1

        # Send final message if timed out
        if retry_count >= max_retries:
            data = {
                "address": wallet.address,
                "status": "timeout",
                "progress": 0,
                "current_step": "分析超时",
                "error": "Analysis timed out after 10 minutes",
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
    """Get list of previously analyzed traders."""
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


@router.post("/analysis/reset")
async def reset_analysis():
    """Force reset stuck analysis status"""
    await reset_analysis_status()
    return {"status": "reset", "running": await is_analysis_running()}


@router.get("/analysis/check/{address}")
async def check_address_analyzed(address: str):
    """Check if an address has been analyzed before."""
    try:
        wallet = WalletAddress(address=address)
        cached = get_cached_analysis(wallet.address)
        if cached:
            return {
                "found": True,
                "address": cached.address,
                "analyzed_at": cached.analyzed_at,
                "account_value": cached.account_value,
                "ai_score": cached.ai_evaluation.score if cached.ai_evaluation else None,
                "ai_recommendation": cached.ai_evaluation.recommendation if cached.ai_evaluation else None,
                "roe_30d": cached.roe_30d,
                "trading_days": cached.trading_days,
                "first_trade_date": cached.first_trade_date,
            }
        return {"found": False, "address": wallet.address}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    """Get all watchlist entries with analysis data."""
    service = get_watchlist_service()
    return service.get_watchlist(sort_by=sort_by, order=order, limit=limit, offset=offset)


@router.post("/watchlist", response_model=dict)
async def add_to_watchlist(request: WatchlistAddRequest):
    """Add an address to the watchlist."""
    service = get_watchlist_service()
    result = service.add_to_watchlist(
        address=request.address,
        alias=request.alias,
        notes=request.notes,
        priority=request.priority,
        auto_update=request.auto_update,
    )

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
    """Manually trigger data sync for a watchlist address."""
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
