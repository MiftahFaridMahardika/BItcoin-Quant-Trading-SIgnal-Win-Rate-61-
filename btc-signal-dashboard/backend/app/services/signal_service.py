from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from app.models.signal import Signal


class SignalService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_recent(self, limit: int = 50) -> list[Signal]:
        result = await self.db.execute(
            select(Signal).order_by(desc(Signal.timestamp)).limit(limit)
        )
        return result.scalars().all()

    async def get_by_type(self, signal_type: str, limit: int = 50) -> list[Signal]:
        result = await self.db.execute(
            select(Signal)
            .where(Signal.signal_type == signal_type)
            .order_by(desc(Signal.timestamp))
            .limit(limit)
        )
        return result.scalars().all()

    async def get_win_rate(self, hours: int = 24) -> float:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self.db.execute(
            select(func.count(Signal.id)).where(Signal.timestamp >= since)
        )
        total = result.scalar() or 0
        if total == 0:
            return 0.0
        # Placeholder: in real system, join with trades table
        return 0.0

    async def get_performance_stats(self) -> dict:
        since_24h = datetime.utcnow() - timedelta(hours=24)
        since_7d = datetime.utcnow() - timedelta(days=7)

        result_24h = await self.db.execute(
            select(func.count(Signal.id), func.avg(Signal.confidence))
            .where(Signal.timestamp >= since_24h)
        )
        row_24h = result_24h.first()

        result_7d = await self.db.execute(
            select(func.count(Signal.id))
            .where(Signal.timestamp >= since_7d)
        )
        count_7d = result_7d.scalar() or 0

        return {
            "signals_24h": row_24h[0] or 0,
            "avg_confidence_24h": round(float(row_24h[1] or 0), 4),
            "signals_7d": count_7d,
        }
