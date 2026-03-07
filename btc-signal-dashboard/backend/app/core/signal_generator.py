"""
Real-Time Signal Generator
Generates trading signals from live market data using the quant engines.

Flow per candle-close event:
  1. Rename columns lowercase → Title Case (engine requirement)
  2. compute_all_features()   — 50+ technical indicators
  3. RegimeDetector.predict() — BULL / BEAR / SIDEWAYS / HIGH_VOL
  4. calculate_signal_score() — weighted 6-layer score (-19…+19)
  5. calculate_sl_tp()        — ATR-based SL/TP levels
  6. Broadcast to WebSocket clients & persist to DB
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.core.data_fetcher import BinanceDataFetcher

logger = logging.getLogger(__name__)

# Path to risk config — relative to this file's package root
_RISK_CONFIG = Path(__file__).parent.parent.parent / "configs" / "risk_config.yaml"


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class SignalOutput:
    """Complete signal output broadcasted to clients."""
    # Identity
    id: str
    timestamp: str
    candle_time: int
    timeframe: str

    # Market snapshot
    price: float
    atr: float

    # Signal
    signal: str        # STRONG_LONG | LONG | SKIP | SHORT | STRONG_SHORT
    direction: Optional[str]   # LONG | SHORT | None
    score: int
    max_score: int
    confidence: float

    # Regime
    regime: str        # BULL | BEAR | SIDEWAYS | HIGH_VOL
    vol_regime: str    # LOW | NORMAL | HIGH | EXTREME  (from SignalEngine)

    # Trade levels (set when direction is not None)
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    risk_reward: Optional[float] = None
    position_size_btc: Optional[float] = None

    # Diagnostics
    reasons: List[str] = field(default_factory=list)
    indicators: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Generator ─────────────────────────────────────────────────────────────────

class RealTimeSignalGenerator:
    """
    Generates trading signals in real-time from BinanceDataFetcher candle events.

    Primary timeframe (default '4h') drives signal generation.
    1m candles from the fetcher are kept for price display only.
    """

    # Signals that warrant calculating SL/TP levels
    ACTIONABLE = {"STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"}

    def __init__(
        self,
        data_fetcher: BinanceDataFetcher,
        primary_timeframe: str = "4h",
        risk_per_trade: float = 1_000.0,
    ):
        self.data_fetcher     = data_fetcher
        self.primary_tf       = primary_timeframe
        self.risk_per_trade   = risk_per_trade

        # Engines — lazy-init so import errors surface cleanly
        self._feature_engine  = None
        self._signal_engine   = None
        self._risk_engine     = None
        self._regime_detector = None
        self._regime_fitted   = False

        # State
        self.current_signal: Optional[SignalOutput] = None
        self.current_regime: str = "UNKNOWN"
        self.signal_history: List[SignalOutput] = []
        self._signal_seq: int = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks (set by external code if needed)
        self.on_new_signal:     Optional[Callable] = None
        self.on_price_update:   Optional[Callable] = None
        self.on_regime_change:  Optional[Callable] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._init_engines()

        # Wire fetcher callbacks
        self.data_fetcher.on_candle_close       = self._on_candle_close
        self.data_fetcher.on_price_update       = self._on_price_update
        self.data_fetcher.on_bootstrap_complete = self._on_bootstrap_complete

        logger.info(f"RealTimeSignalGenerator started (primary_tf={self.primary_tf})")

    async def stop(self):
        self._running = False
        logger.info("RealTimeSignalGenerator stopped")

    # ── Engine initialisation ──────────────────────────────────────────────────

    def _init_engines(self):
        """Import and instantiate engines. Errors are logged, not raised."""
        try:
            from app.core.engines.feature_engine import FeatureEngine
            self._feature_engine = FeatureEngine()
            logger.info("FeatureEngine loaded")
        except Exception as e:
            logger.error(f"FeatureEngine init failed: {e}")

        try:
            from app.core.engines.signal_engine import SignalEngine
            self._signal_engine = SignalEngine({})
            logger.info("SignalEngine loaded")
        except Exception as e:
            logger.error(f"SignalEngine init failed: {e}")

        try:
            from app.core.engines.risk_engine import RiskEngine
            config_path = str(_RISK_CONFIG)
            self._risk_engine = RiskEngine(config_path=config_path)
            logger.info(f"RiskEngine loaded (config={config_path})")
        except Exception as e:
            logger.error(f"RiskEngine init failed: {e}")

        try:
            from app.core.engines.regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector()
            logger.info("RegimeDetector loaded (unfitted — will fit on first signal)")
        except Exception as e:
            logger.error(f"RegimeDetector init failed: {e}")

    def _bootstrap_regime(self):
        """Fit regime detector on startup using existing bootstrap data."""
        if not self._regime_detector or not self._feature_engine:
            return
        try:
            history = self.data_fetcher.get_candles(self.primary_tf)
            if len(history) < 250:
                logger.warning(f"Not enough bootstrap candles for regime fit: {len(history)}")
                return
            df = self._build_df(history)
            df = self._feature_engine.compute_all_features(df)
            start = df.index[0].strftime("%Y-%m-%d")
            end   = df.index[-1].strftime("%Y-%m-%d")
            self._regime_detector.fit(df, start, end)
            self._regime_fitted = True
            series = self._regime_detector.predict(df)
            self.current_regime = str(series.iloc[-1])
            logger.info(f"RegimeDetector bootstrapped: current_regime={self.current_regime}")
        except Exception as e:
            logger.warning(f"Regime bootstrap failed: {e}")

    # ── Fetcher callbacks ──────────────────────────────────────────────────────

    async def _on_bootstrap_complete(self):
        """Called by data_fetcher once historical data is loaded. Fit regime immediately."""
        logger.info("Bootstrap complete — fitting RegimeDetector...")
        await asyncio.get_event_loop().run_in_executor(None, self._bootstrap_regime)
        if self.current_regime != "UNKNOWN" and self.on_regime_change:
            await self.on_regime_change(self.current_regime, "UNKNOWN")

    async def _on_candle_close(self, timeframe: str, candle: Dict, history: List[Dict]):
        """Called by BinanceDataFetcher on every closed candle."""
        if timeframe != self.primary_tf:
            return  # only generate signals on the primary timeframe

        logger.info(
            f"Candle close [{timeframe}] @ ${candle['close']:,.2f} "
            f"| history={len(history)}"
        )
        try:
            signal = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_signal_sync, timeframe, candle, history
            )
            if signal:
                await self._emit(signal)
        except Exception as e:
            logger.error(f"Signal pipeline error: {e}", exc_info=True)

    async def _on_price_update(self, price_data: Dict):
        if self.on_price_update:
            await self.on_price_update(price_data)

    # ── Signal pipeline (runs in thread pool to avoid blocking event loop) ─────

    def _generate_signal_sync(
        self, timeframe: str, candle: Dict, history: List[Dict]
    ) -> Optional[SignalOutput]:
        """Synchronous signal generation — safe to run in executor."""

        if len(history) < 210:
            logger.warning(f"Not enough history: {len(history)} bars (need 210+ for EMA200)")
            return None

        # 1. Build DataFrame with Title-Case columns (engine requirement)
        df = self._build_df(history)

        # 2. Feature engineering
        if not self._feature_engine:
            logger.warning("FeatureEngine not available — skipping")
            return None
        df = self._feature_engine.compute_all_features(df)

        # 3. Regime detection
        regime, regime_changed = self._detect_regime(df)

        # 4. Signal scoring (last bar = index -1)
        if not self._signal_engine:
            logger.warning("SignalEngine not available — skipping")
            return None
        result = self._signal_engine.calculate_signal_score(df, len(df) - 1)

        # 5. Extract ATR (FeatureEngine produces 'atr_14')
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else candle["close"] * 0.02

        # 6. Build output
        self._signal_seq += 1
        sig_id = f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._signal_seq:04d}"

        direction: Optional[str] = None
        if result.signal in self.ACTIONABLE:
            direction = "LONG" if result.score > 0 else "SHORT"

        output = SignalOutput(
            id=sig_id,
            timestamp=datetime.utcnow().isoformat(),
            candle_time=candle["timestamp"],
            timeframe=timeframe,
            price=candle["close"],
            atr=atr,
            signal=result.signal,
            direction=direction,
            score=result.score,
            max_score=result.max_score,
            confidence=result.confidence,
            regime=regime,
            vol_regime=result.regime,  # LOW/NORMAL/HIGH/EXTREME from SignalEngine
            reasons=result.reasons[:6],
            indicators=self._extract_indicators(df),
        )

        # 7. SL/TP levels when actionable
        if direction and self._risk_engine:
            try:
                sl_tp = self._risk_engine.calculate_sl_tp(
                    entry_price=candle["close"],
                    direction=direction,
                    atr=atr,
                    profile="normal",
                )
                pos = self._risk_engine.calculate_position_size(
                    entry_price=candle["close"],
                    stop_loss=sl_tp.stop_loss,
                    risk_amount=self.risk_per_trade,
                    leverage=1.0,
                )
                output.entry            = candle["close"]
                output.stop_loss        = sl_tp.stop_loss
                output.take_profit_1    = sl_tp.take_profit_1
                output.take_profit_2    = sl_tp.take_profit_2
                output.take_profit_3    = sl_tp.take_profit_3
                output.risk_reward      = sl_tp.risk_reward_ratio
                output.position_size_btc = pos.quantity
            except Exception as e:
                logger.warning(f"SL/TP calculation failed: {e}")

        if regime_changed and self.on_regime_change:
            # schedule the coroutine callback
            asyncio.run_coroutine_threadsafe(
                self.on_regime_change(regime, self.current_regime),
                asyncio.get_event_loop(),
            )

        return output

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_df(self, history: List[Dict]) -> pd.DataFrame:
        """Convert candle history to DataFrame with Title-Case OHLCV columns."""
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        # Rename to what FeatureEngine / SignalEngine expect
        df = df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    def _detect_regime(self, df: pd.DataFrame):
        """
        Try to detect regime.  On the first call, fits the HMM if possible.
        Returns (regime_label, changed: bool).
        """
        if not self._regime_detector:
            return "UNKNOWN", False

        try:
            # Fit if not yet fitted (retries on every call until success)
            if not self._regime_fitted and len(df) >= 250:
                start = df.index[0].strftime("%Y-%m-%d")
                end   = df.index[-1].strftime("%Y-%m-%d")
                self._regime_detector.fit(df, start, end)
                self._regime_fitted = True
                logger.info("RegimeDetector fitted on candle close")

            if self._regime_fitted:
                series = self._regime_detector.predict(df)
                regime = str(series.iloc[-1])
            else:
                regime = self.current_regime if self.current_regime != "UNKNOWN" else "UNKNOWN"

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            regime = self.current_regime if self.current_regime != "UNKNOWN" else "UNKNOWN"

        changed = regime != self.current_regime
        self.current_regime = regime
        return regime, changed

    def _extract_indicators(self, df: pd.DataFrame) -> Dict:
        """Pull the most relevant indicator values for dashboard display."""
        last = df.iloc[-1]
        cols = [
            # Trend
            "ema_21", "ema_55", "ema_200", "hma_55", "supertrend_dir",
            "ema_stack_signal", "hma_signal", "ichi_signal",
            # Momentum
            "rsi", "macd_line", "macd_hist", "zscore_20",
            "stoch_k", "stoch_d", "williams_r",
            # Volatility
            "atr_14", "atr_pct", "atr_ratio", "vol_regime",
            "bb_pct_b", "bb_bandwidth",
            # Volume
            "obv_signal", "vwap_signal", "cmf_20", "vol_ratio",
            # Price action
            "trend_structure",
        ]
        result = {}
        for col in cols:
            if col in df.columns:
                v = last[col]
                result[col] = float(v) if pd.notna(v) else None
        return result

    # ── Broadcast & persist ────────────────────────────────────────────────────

    async def _emit(self, signal: SignalOutput):
        """Store, broadcast and persist a new signal."""
        self.current_signal = signal
        self.signal_history.append(signal)
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        logger.info(
            f"SIGNAL  {signal.signal:<14s}  "
            f"score={signal.score:+d}/{signal.max_score}  "
            f"conf={signal.confidence:.0%}  "
            f"regime={signal.regime}  "
            f"price=${signal.price:,.2f}"
        )

        # Broadcast via callback (wired by main.py → broadcast_to_all)
        if self.on_new_signal:
            try:
                await self.on_new_signal(signal)
            except Exception as e:
                logger.error(f"Broadcast callback failed: {e}")

        # DB persistence
        await self._save_signal(signal)

    async def _save_signal(self, signal: SignalOutput):
        try:
            from app.database import AsyncSessionLocal
            from app.models.signal import Signal

            async with AsyncSessionLocal() as session:
                row = Signal(
                    timestamp=datetime.utcnow(),
                    price=signal.price,
                    signal_type=signal.signal,
                    confidence=signal.confidence,
                    regime=signal.regime,
                    entry=signal.entry,
                    stop_loss=signal.stop_loss,
                    take_profit_1=signal.take_profit_1,
                    take_profit_2=signal.take_profit_2,
                    take_profit_3=signal.take_profit_3,
                    risk_reward=signal.risk_reward,
                    scores={"score": signal.score, "max_score": signal.max_score,
                            "individual": signal.indicators},
                )
                session.add(row)
                await session.commit()
        except Exception as e:
            logger.error(f"DB save failed: {e}")

    # ── Public accessors ───────────────────────────────────────────────────────

    def get_current_signal(self) -> Optional[Dict]:
        return self.current_signal.to_dict() if self.current_signal else None

    def get_signal_history(self, limit: int = 20) -> List[Dict]:
        tail = self.signal_history[-limit:] if limit else self.signal_history
        return [s.to_dict() for s in reversed(tail)]

    def get_market_summary(self) -> Dict:
        return {
            **self.data_fetcher.get_market_info(),
            "regime": self.current_regime,
            "current_signal": self.get_current_signal(),
            "signal_count": self._signal_seq,
        }
