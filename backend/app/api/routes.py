from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.models.schemas import (
    WalletAddress, TraderStats, TraderPnLHistory,
    TraderTradesResponse, PerformanceSummary, APIError
)
from app.services.hyperliquid import (
    HyperliquidService, SuperXService,
    get_hyperliquid_service, get_superx_service
)
from app.services.analyzer import parse_superx_stats, analyze_trades

router = APIRouter(prefix="/api/v1", tags=["traders"])


@router.get("/traders/{address}/stats", response_model=TraderStats)
async def get_trader_stats(
    address: str,
    superx: SuperXService = Depends(get_superx_service)
):
    """Get trader statistics from SuperX"""
    try:
        wallet = WalletAddress(address=address)
        data = await superx.get_trader_stats(wallet.address)
        return parse_superx_stats(wallet.address, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/traders/{address}/portfolio")
async def get_trader_portfolio(
    address: str,
    hl: HyperliquidService = Depends(get_hyperliquid_service)
):
    """Get trader portfolio data from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        return await hl.get_portfolio(wallet.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio: {str(e)}")


@router.get("/traders/{address}/state")
async def get_trader_state(
    address: str,
    hl: HyperliquidService = Depends(get_hyperliquid_service)
):
    """Get current trader state (positions, margin) from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        return await hl.get_user_state(wallet.address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch state: {str(e)}")


@router.get("/traders/{address}/trades")
async def get_trader_trades(
    address: str,
    hl: HyperliquidService = Depends(get_hyperliquid_service)
):
    """Get trader's recent trades from Hyperliquid"""
    try:
        wallet = WalletAddress(address=address)
        fills = await hl.get_user_fills(wallet.address)
        return {
            "address": wallet.address,
            "total_count": len(fills),
            "trades": fills
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trades: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "hy-analyzer-api"}
