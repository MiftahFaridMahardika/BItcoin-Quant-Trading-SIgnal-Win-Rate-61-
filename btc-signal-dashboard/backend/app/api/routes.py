from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models.signal import Signal
from app.models.candle import Candle

router = APIRouter()


@router.get("/signals")
async def get_signals(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    signal_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Signal).order_by(desc(Signal.timestamp))
    if signal_type:
        query = query.where(Signal.signal_type == signal_type)
    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    signals = result.scalars().all()
    return {"signals": [s.to_dict() for s in signals], "total": len(signals)}


@router.get("/signals/latest")
async def get_latest_signal(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Signal).order_by(desc(Signal.timestamp)).limit(1)
    )
    signal = result.scalar_one_or_none()
    if not signal:
        raise HTTPException(status_code=404, detail="No signals found")
    return signal.to_dict()


@router.get("/candles")
async def get_candles(
    limit: int = Query(200, ge=1, le=1000),
    timeframe: str = Query("1m"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Candle)
        .where(Candle.timeframe == timeframe)
        .order_by(desc(Candle.timestamp))
        .limit(limit)
    )
    candles = result.scalars().all()
    # Return in chronological order
    return {"candles": [c.to_dict() for c in reversed(candles)]}


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    since = datetime.utcnow() - timedelta(hours=24)
    result = await db.execute(
        select(Signal).where(Signal.timestamp >= since)
    )
    signals_24h = result.scalars().all()

    buy_count = sum(1 for s in signals_24h if s.signal_type == "BUY")
    sell_count = sum(1 for s in signals_24h if s.signal_type == "SELL")
    avg_confidence = (
        sum(s.confidence for s in signals_24h) / len(signals_24h)
        if signals_24h else 0.0
    )

    return {
        "signals_24h": len(signals_24h),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "avg_confidence": round(avg_confidence, 4),
    }
