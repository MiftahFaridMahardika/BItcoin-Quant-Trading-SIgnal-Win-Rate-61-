"""
Integration test: BinanceDataFetcher + RealTimeSignalGenerator + all engines.

Run from backend/ directory:
    python test_signal_generator.py

Takes ~15s to bootstrap, then waits for a 4h candle close (~up to 4h wait).
For quick integration check, we manually trigger the pipeline on loaded data.
"""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

sys.path.insert(0, ".")

from app.core.data_fetcher import BinanceDataFetcher
from app.core.signal_generator import RealTimeSignalGenerator


async def run_integration_test():
    """Bootstrap live data then immediately run signal pipeline on loaded candles."""

    logger.info("=" * 65)
    logger.info("BTC Signal Generator — Integration Test")
    logger.info("=" * 65)

    # ── Step 1: Bootstrap data ────────────────────────────────────────────────
    fetcher = BinanceDataFetcher("btcusdt")
    generator = RealTimeSignalGenerator(
        fetcher,
        primary_timeframe="4h",
        risk_per_trade=1_000.0,
    )

    received_signals: list = []

    async def on_signal(signal):
        received_signals.append(signal)
        logger.info(f"  CALLBACK received signal id={signal.id}")

    generator.on_new_signal = on_signal

    # Start generator (wires callbacks into fetcher)
    await generator.start()

    # Start fetcher in background
    fetcher_task = asyncio.create_task(fetcher.start())

    # Wait for bootstrap (up to 30s)
    logger.info("Waiting for bootstrap...")
    for _ in range(80):
        await asyncio.sleep(0.5)
        bars_4h = len(fetcher.candles["4h"])
        if bars_4h >= 400:
            break

    bars_4h = len(fetcher.candles["4h"])
    bars_1m = len(fetcher.candles["1m"])
    logger.info(f"Bootstrap complete: 4h={bars_4h}  1m={bars_1m}")

    if bars_4h < 250:
        logger.error(f"Not enough 4h bars ({bars_4h}) — check internet access")
        await fetcher.stop()
        fetcher_task.cancel()
        return False

    # ── Step 2: Manually trigger signal pipeline on loaded 4h data ────────────
    logger.info("\nManually triggering signal pipeline on loaded 4h history...")

    history_4h = fetcher.get_candles("4h")
    last_candle = history_4h[-1]

    # Simulate a candle-close callback
    await generator._on_candle_close("4h", last_candle, history_4h)

    # Give async tasks a moment to complete
    await asyncio.sleep(2)

    # ── Step 3: Report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 65)

    summary = generator.get_market_summary()
    logger.info(f"  Symbol       : {summary['symbol']}")
    logger.info(f"  Price        : ${summary['price']:,.2f}  ({summary['change_24h']:+.2f}%)")
    logger.info(f"  Regime       : {summary['regime']}")
    logger.info(f"  Signal count : {summary['signal_count']}")

    current = generator.get_current_signal()
    if current:
        logger.info(f"\n  Signal       : {current['signal']}")
        logger.info(f"  Score        : {current['score']:+d}/{current['max_score']}")
        logger.info(f"  Confidence   : {current['confidence']:.1%}")
        logger.info(f"  Vol regime   : {current['vol_regime']}")
        logger.info(f"  Direction    : {current['direction']}")
        if current.get("stop_loss"):
            logger.info(f"  Entry        : ${current['entry']:,.2f}")
            logger.info(f"  Stop Loss    : ${current['stop_loss']:,.2f}")
            logger.info(f"  TP1          : ${current['take_profit_1']:,.2f}")
            logger.info(f"  TP2          : ${current['take_profit_2']:,.2f}")
            logger.info(f"  TP3          : ${current['take_profit_3']:,.2f}")
            logger.info(f"  R/R Ratio    : {current['risk_reward']:.2f}x")
            logger.info(f"  Size (BTC)   : {current['position_size_btc']:.6f}")
        if current.get("reasons"):
            logger.info(f"\n  Reasons:")
            for r in current["reasons"]:
                logger.info(f"    • {r}")
        ind = current.get("indicators", {})
        if ind:
            logger.info(f"\n  Key indicators:")
            for k in ["rsi", "macd_hist", "zscore_20", "ema_stack_signal",
                      "supertrend_dir", "vol_ratio", "atr_14"]:
                v = ind.get(k)
                if v is not None:
                    logger.info(f"    {k:<20s}: {v:.4f}")
    else:
        logger.warning("  No signal generated yet")

    passed = current is not None
    logger.info(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 65)

    await fetcher.stop()
    fetcher_task.cancel()
    return passed


if __name__ == "__main__":
    result = asyncio.run(run_integration_test())
    sys.exit(0 if result else 1)
