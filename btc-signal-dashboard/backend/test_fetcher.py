"""
Test script for BinanceDataFetcher.
Run from the backend/ directory:
    pip install websockets aiohttp pandas
    python test_fetcher.py
"""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

# Allow running without installing the app package
sys.path.insert(0, ".")
from app.core.data_fetcher import BinanceDataFetcher


# ── Counters ────────────────────────────────────────────
candle_count = {"1m": 0, "5m": 0, "1h": 0}
price_updates = 0
TEST_DURATION = 90          # seconds to run before auto-stop
CANDLE_CLOSE_TARGET = 2     # stop after this many 1m closes (smoke test)


async def on_candle_close(tf: str, candle: dict, history: list):
    candle_count[tf] = candle_count.get(tf, 0) + 1
    logger.info(
        f"CANDLE_CLOSE [{tf:>3}] "
        f"O={candle['open']:,.2f}  H={candle['high']:,.2f}  "
        f"L={candle['low']:,.2f}  C={candle['close']:,.2f}  "
        f"V={candle['volume']:.2f}  history={len(history)}"
    )


async def on_price_update(info: dict):
    global price_updates
    price_updates += 1
    if price_updates % 10 == 1:          # print every 10th update
        logger.info(
            f"PRICE  ${info['price']:,.2f}  "
            f"24h {info['change_24h']:+.2f}%  "
            f"Vol {info['volume_24h']:,.0f}"
        )


async def on_connection_change(status: str, error):
    if status == "connected":
        logger.info("CONNECTION  status=connected")
    else:
        logger.warning(f"CONNECTION  status={status}  error={error}")


async def run_test():
    fetcher = BinanceDataFetcher("btcusdt")
    fetcher.on_candle_close       = on_candle_close
    fetcher.on_price_update       = on_price_update
    fetcher.on_connection_change  = on_connection_change

    # Run in background so we can inspect state after a timeout
    task = asyncio.create_task(fetcher.start())

    # Wait until bootstrapped (up to 30 s)
    for _ in range(60):
        await asyncio.sleep(0.5)
        if fetcher.is_ready:
            break

    if fetcher.is_ready:
        info = fetcher.get_market_info()
        df1m = fetcher.get_dataframe("1m")
        df1h = fetcher.get_dataframe("1h")

        logger.info("=" * 60)
        logger.info("BOOTSTRAP RESULT")
        logger.info(f"  Symbol  : {info['symbol']}")
        logger.info(f"  Price   : ${info['price']:,.2f}")
        logger.info(f"  Change  : {info['change_24h']:+.2f}%")
        logger.info(f"  1m rows : {len(df1m)}")
        logger.info(f"  1h rows : {len(df1h)}")
        if not df1m.empty:
            logger.info(f"  1m head :\n{df1m.tail(3)}")
        logger.info("=" * 60)
        logger.info(f"Listening for live candles for {TEST_DURATION}s ...")
    else:
        logger.error("Bootstrap timed out — check internet / Binance access")
        await fetcher.stop()
        task.cancel()
        return

    # Run until timeout
    try:
        await asyncio.wait_for(task, timeout=TEST_DURATION)
    except asyncio.TimeoutError:
        pass

    await fetcher.stop()
    task.cancel()

    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info(f"  Price updates received : {price_updates}")
    for tf, count in candle_count.items():
        logger.info(f"  Candles closed [{tf:>3}]  : {count}")
    logger.info("  PASS" if fetcher.is_ready else "  FAIL (not ready)")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
