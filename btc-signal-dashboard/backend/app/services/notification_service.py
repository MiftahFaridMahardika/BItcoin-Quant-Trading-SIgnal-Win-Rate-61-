import logging
from app.api.websocket import broadcast

logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications for important signal events."""

    async def notify_signal(self, signal: dict):
        """Broadcast a new signal notification."""
        if signal.get("signal") in ("BUY", "SELL"):
            confidence = signal.get("confidence", 0)
            if confidence >= 0.6:
                await broadcast({
                    "type": "alert",
                    "severity": "high" if confidence >= 0.8 else "medium",
                    "signal": signal["signal"],
                    "price": signal["price"],
                    "confidence": confidence,
                    "message": (
                        f"Strong {signal['signal']} signal detected! "
                        f"Confidence: {confidence:.1%} | "
                        f"Price: ${signal['price']:,.2f}"
                    ),
                })
                logger.info(f"Alert sent for {signal['signal']} signal")

    async def notify_regime_change(self, old_regime: str, new_regime: str, price: float):
        """Broadcast regime change notification."""
        await broadcast({
            "type": "regime_change",
            "from": old_regime,
            "to": new_regime,
            "price": price,
            "message": f"Market regime changed: {old_regime} → {new_regime}",
        })
        logger.info(f"Regime change: {old_regime} → {new_regime}")
