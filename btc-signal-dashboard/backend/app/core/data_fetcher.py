"""
Binance Real-Time Data Fetcher
Connects to Binance WebSocket for live OHLCV data.
"""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional
from collections import deque
import aiohttp
import websockets

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Real-time data fetcher dari Binance WebSocket.

    Features:
    - Multi-timeframe streaming (1m, 5m, 15m, 1h, 4h, 1d)
    - Auto-reconnect on disconnect (exponential backoff)
    - Historical data bootstrap via REST API
    - Callback system untuk signal generation
    """

    WS_URL = "wss://stream.binance.com:9443/ws"
    REST_URL = "https://api.binance.com/api/v3"

    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.ws = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Candle storage per timeframe
        self.candles: Dict[str, deque] = {
            '1m':  deque(maxlen=1000),
            '5m':  deque(maxlen=500),
            '15m': deque(maxlen=500),
            '1h':  deque(maxlen=500),
            '4h':  deque(maxlen=600),   # need 500+ for FeatureEngine (EMA200 + warmup)
            '1d':  deque(maxlen=300),
        }

        # Current (incomplete) candles
        self.current_candles: Dict[str, Dict] = {}

        # Latest market data
        self.current_price: float = 0.0
        self.price_change_24h: float = 0.0
        self.volume_24h: float = 0.0

        # Callbacks
        self.on_candle_close: Optional[Callable] = None
        self.on_price_update: Optional[Callable] = None
        self.on_connection_change: Optional[Callable] = None
        self.on_bootstrap_complete: Optional[Callable] = None

    # ──────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────

    async def start(self):
        """Start WebSocket connection with auto-reconnect."""
        self.is_running = True

        while self.is_running:
            try:
                await self._connect()
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error(f"WebSocket error (attempt {self.reconnect_attempts}): {e}")

                if self.on_connection_change:
                    await self.on_connection_change("disconnected", str(e))

                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached. Stopping.")
                    break

                wait = min(30, 2 ** self.reconnect_attempts)
                logger.info(f"Reconnecting in {wait}s...")
                await asyncio.sleep(wait)

    async def stop(self):
        """Stop WebSocket connection gracefully."""
        self.is_running = False
        if self.ws:
            await self.ws.close()
        logger.info("BinanceDataFetcher stopped")

    def get_candles(self, timeframe: str, limit: int = None) -> List[Dict]:
        """Return historical closed candles for a timeframe."""
        candles = list(self.candles.get(timeframe, []))
        if limit:
            candles = candles[-limit:]
        return candles

    def get_dataframe(self, timeframe: str = "1m", limit: int = None):
        """Return candles as a pandas DataFrame (index = timestamp)."""
        import pandas as pd

        candles = self.get_candles(timeframe, limit)
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_market_info(self) -> Dict:
        """Return current market snapshot."""
        return {
            'symbol': self.symbol.upper(),
            'price': self.current_price,
            'change_24h': self.price_change_24h,
            'volume_24h': self.volume_24h,
        }

    @property
    def is_ready(self) -> bool:
        """True once enough 1m candles are loaded for signal generation."""
        return len(self.candles['1m']) >= 100

    # ──────────────────────────────────────────────
    # WebSocket connection
    # ──────────────────────────────────────────────

    async def _connect(self):
        """Open combined Binance WebSocket stream."""
        streams = [
            f"{self.symbol}@kline_1m",
            f"{self.symbol}@kline_5m",
            f"{self.symbol}@kline_15m",
            f"{self.symbol}@kline_1h",
            f"{self.symbol}@kline_4h",
            f"{self.symbol}@kline_1d",
            f"{self.symbol}@ticker",
        ]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        logger.info(f"Connecting to Binance WebSocket ({self.symbol.upper()})...")

        async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
            self.ws = ws
            self.reconnect_attempts = 0

            if self.on_connection_change:
                await self.on_connection_change("connected", None)

            logger.info("WebSocket connected successfully")

            # Bootstrap historical data before processing live messages
            await self._load_historical_data()
            if self.on_bootstrap_complete:
                await self.on_bootstrap_complete()

            async for message in ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")

    # ──────────────────────────────────────────────
    # Historical bootstrap
    # ──────────────────────────────────────────────

    async def _load_historical_data(self):
        """Bootstrap candle history via Binance REST API."""
        logger.info("Loading historical candles...")

        timeframes = [
            ('1m',  500),
            ('5m',  300),
            ('15m', 200),
            ('1h',  200),
            ('4h',  501),   # 500 closed + 1 open → 500 closed bars for engines
            ('1d',  200),
        ]

        async with aiohttp.ClientSession() as session:
            for tf, limit in timeframes:
                try:
                    params = {
                        "symbol": self.symbol.upper(),
                        "interval": tf,
                        "limit": limit,
                    }
                    async with session.get(f"{self.REST_URL}/klines", params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # All but the last candle are closed
                            for row in data[:-1]:
                                self.candles[tf].append(self._parse_rest_candle(row))
                            # Last candle is still open
                            if data:
                                self.current_candles[tf] = self._parse_rest_candle(data[-1])
                            logger.info(f"  {tf}: {len(data) - 1} candles loaded")
                        else:
                            logger.error(f"  {tf}: REST error {resp.status}")
                except Exception as e:
                    logger.error(f"  {tf}: {e}")

            # Load 24h ticker for price / change data
            try:
                params = {"symbol": self.symbol.upper()}
                async with session.get(f"{self.REST_URL}/ticker/24hr", params=params) as resp:
                    if resp.status == 200:
                        t = await resp.json()
                        self.current_price    = float(t['lastPrice'])
                        self.price_change_24h = float(t['priceChangePercent'])
                        self.volume_24h       = float(t['volume'])
                        logger.info(f"  Price: ${self.current_price:,.2f} ({self.price_change_24h:+.2f}%)")
            except Exception as e:
                logger.error(f"  ticker: {e}")

    def _parse_rest_candle(self, row: list) -> Dict:
        return {
            'timestamp': row[0],
            'open':      float(row[1]),
            'high':      float(row[2]),
            'low':       float(row[3]),
            'close':     float(row[4]),
            'volume':    float(row[5]),
            'close_time': row[6],
            'trades':    row[8],
        }

    # ──────────────────────────────────────────────
    # Message handlers
    # ──────────────────────────────────────────────

    async def _handle_message(self, data: Dict):
        if 'stream' not in data:
            return
        stream  = data['stream']
        payload = data['data']

        if '@kline_' in stream:
            await self._handle_kline(payload)
        elif '@ticker' in stream:
            await self._handle_ticker(payload)

    async def _handle_kline(self, data: Dict):
        """Process a kline (candlestick) event."""
        k         = data['k']
        tf        = k['i']
        is_closed = k['x']

        candle = {
            'timestamp':  k['t'],
            'open':       float(k['o']),
            'high':       float(k['h']),
            'low':        float(k['l']),
            'close':      float(k['c']),
            'volume':     float(k['v']),
            'close_time': k['T'],
            'trades':     k['n'],
        }

        self.current_candles[tf] = candle
        self.current_price = candle['close']

        if is_closed:
            self.candles[tf].append(candle)
            logger.info(f"Candle closed [{tf}] @ ${candle['close']:,.2f}")

            if self.on_candle_close:
                await self.on_candle_close(tf, candle, list(self.candles[tf]))

    async def _handle_ticker(self, data: Dict):
        """Process a 24h rolling ticker event."""
        self.current_price    = float(data['c'])
        self.price_change_24h = float(data['P'])
        self.volume_24h       = float(data['v'])

        if self.on_price_update:
            await self.on_price_update({
                'price':      self.current_price,
                'change_24h': self.price_change_24h,
                'volume_24h': self.volume_24h,
                'high_24h':   float(data['h']),
                'low_24h':    float(data['l']),
            })
