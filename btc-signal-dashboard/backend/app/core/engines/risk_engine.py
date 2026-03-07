"""
Risk Engine - Hedge Fund Grade Risk Management

Core Components:
1. Position Sizing (Kelly Criterion)
2. Dynamic SL/TP (ATR-based)
3. Drawdown Scaling
4. Risk Limits
5. Performance Metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    risk_amount: float
    position_notional: float
    margin_required: float
    quantity: float
    method: str


@dataclass
class SLTPLevels:
    """Stop Loss and Take Profit levels."""
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    sl_distance: float
    sl_percentage: float
    risk_reward_ratio: float


class RiskEngine:
    """
    Comprehensive Risk Management Engine.
    """

    def __init__(self, config_path: str = "configs/risk_config.yaml"):
        """Initialize with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Account state
        self.initial_capital = 100_000
        self.account_balance = self.initial_capital
        self.peak_balance = self.initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0

        # Open positions tracking
        self.open_positions: List[Dict] = []

        # Risk limits
        self.max_risk_per_trade = self.config['position_sizing']['max_risk_per_trade']
        self.max_daily_loss = self.config['risk_limits']['max_daily_loss']
        self.max_total_drawdown = self.config['risk_limits']['max_total_drawdown']
        self.max_concurrent = self.config['risk_limits']['max_concurrent_trades']

    # ══════════════════════════════════════════════════════════
    # POSITION SIZING
    # ══════════════════════════════════════════════════════════

    def kelly_criterion(self,
                        win_rate: float,
                        avg_win_r: float,
                        avg_loss_r: float = 1.0,
                        kelly_fraction: float = 0.5) -> Dict:
        """
        Kelly Criterion Position Sizing

        Formula:
        f = (win_rate × avg_win - loss_rate × avg_loss) / avg_win

        Half-Kelly (safer):
        f_half = f × kelly_fraction

        Example:
        - Win rate: 58%
        - Avg win: 2.3R
        - Avg loss: 1.0R

        Kelly Full = (0.58 × 2.3 - 0.42 × 1.0) / 2.3 = 39.7%
        Kelly Half = 39.7% × 0.5 = 19.85%
        """
        loss_rate = 1 - win_rate

        # Full Kelly
        kelly_full = (win_rate * avg_win_r - loss_rate * avg_loss_r) / avg_win_r
        kelly_full = max(0.0, kelly_full)

        # Half Kelly (default)
        kelly_half = kelly_full * kelly_fraction

        # Cap at max risk
        kelly_capped = min(kelly_half, self.max_risk_per_trade)

        # Dollar risk
        dollar_risk = self.account_balance * kelly_capped

        return {
            "kelly_full_pct": kelly_full * 100,
            "kelly_half_pct": kelly_half * 100,
            "kelly_capped_pct": kelly_capped * 100,
            "dollar_risk": dollar_risk,
            "explanation": f"Risk ${dollar_risk:,.2f} ({kelly_capped:.2%}) per trade",
        }

    def fixed_fractional(self, risk_pct: float = None) -> float:
        """Fixed Fractional: Risk fixed % of account."""
        risk_pct = risk_pct or self.max_risk_per_trade
        return self.account_balance * risk_pct

    def volatility_adjusted_sizing(self,
                                   atr: float,
                                   price: float,
                                   base_risk_pct: float = None) -> Dict:
        """
        Adjust position size based on volatility.

        High volatility → smaller position
        Low volatility  → larger position
        """
        base_risk_pct = base_risk_pct or self.max_risk_per_trade
        volatility_pct = atr / price

        # Normalize: 2% ATR/Price treated as "normal"
        vol_multiplier = 0.02 / volatility_pct if volatility_pct > 0 else 1.0
        vol_multiplier = float(np.clip(vol_multiplier, 0.5, 2.0))

        adjusted_risk_pct = min(base_risk_pct * vol_multiplier, self.max_risk_per_trade)

        return {
            "volatility_pct": volatility_pct * 100,
            "vol_multiplier": vol_multiplier,
            "adjusted_risk_pct": adjusted_risk_pct * 100,
            "dollar_risk": self.account_balance * adjusted_risk_pct,
        }

    # ══════════════════════════════════════════════════════════
    # DYNAMIC POSITION SIZING (NEW)
    # ══════════════════════════════════════════════════════════

    def calculate_dynamic_kelly(
        self,
        recent_trades: List[Dict],
        market_bias:   str  = "NEUTRAL",
        regime:        str  = "NORMAL",
        base_fraction: float = 0.5,       # half-Kelly baseline
    ) -> float:
        """
        Adjust Kelly fraction using recent win-rate and market conditions.

        Blend: 60% base half-Kelly + 40% recent-performance Kelly
        Then multiply by market-bias and regime factors.

        Parameters
        ----------
        recent_trades : list of dicts with 'pnl' and 'pnl_r' keys
        market_bias   : 'STRONG_BULL' | 'BULL' | 'NEUTRAL' | 'BEAR' | 'STRONG_BEAR'
        regime        : 'BULL' | 'BEAR' | 'SIDEWAYS' | 'NORMAL' | 'HIGH_VOL'
        base_fraction : baseline half-Kelly fraction (default 0.5)

        Returns
        -------
        float — adjusted Kelly fraction (0.05 … 0.25)
        """
        # ── Base half-Kelly ──────────────────────────────────
        # Historical parameters (calibrated for this system)
        base_win_rate = 0.50
        base_avg_win  = 1.5
        base_avg_loss = 1.0
        full_kelly = (
            base_win_rate * base_avg_win
            - (1 - base_win_rate) * base_avg_loss
        ) / max(base_avg_win, 1e-9)
        base_kelly = max(0.0, full_kelly) * base_fraction

        # ── Recent-performance Kelly ─────────────────────────
        window = 20
        if len(recent_trades) >= window:
            tail = recent_trades[-window:]
            wins   = [t for t in tail if t.get("pnl", 0) > 0]
            losses = [t for t in tail if t.get("pnl", 0) < 0]
            wr     = len(wins) / window

            avg_win_r  = float(np.mean([t.get("pnl_r", 0) for t in wins])) if wins else base_avg_win
            avg_loss_r = float(np.mean([abs(t.get("pnl_r", 0)) for t in losses])) if losses else base_avg_loss
            avg_win_r  = max(avg_win_r, 0.1)
            avg_loss_r = max(avg_loss_r, 0.1)

            recent_full_kelly = (wr * avg_win_r - (1 - wr) * avg_loss_r) / avg_win_r
            recent_kelly      = max(0.0, recent_full_kelly) * base_fraction

            # Blend: 60% historical, 40% recent
            adjusted = 0.60 * base_kelly + 0.40 * recent_kelly
        elif len(recent_trades) >= 5:
            # Have some data but not enough: conservative blend
            adjusted = base_kelly * 0.75
        else:
            # Cold start: very conservative
            adjusted = base_kelly * 0.50

        # ── Market-bias multiplier ───────────────────────────
        bias_mults = {
            "STRONG_BULL": 1.25,
            "BULL":        1.10,
            "NEUTRAL":     0.85,
            "BEAR":        0.80,
            "STRONG_BEAR": 0.70,
        }
        adjusted *= bias_mults.get(market_bias, 0.85)

        # ── Regime multiplier ────────────────────────────────
        if regime.upper() in ("HIGH_VOL", "EXTREME"):
            adjusted *= 0.40   # extreme risk-off
        elif regime.upper() in ("HIGH",):
            adjusted *= 0.75

        # ── Drawdown scaling from existing engine ────────────
        dd_mult, _ = self.get_drawdown_multiplier()
        adjusted  *= dd_mult

        # ── Hard caps: 5% min fraction, 25% max ─────────────
        return float(np.clip(adjusted, 0.05, 0.25))

    # ─────────────────────────────────────────────────────────

    def get_streak_multiplier(self, recent_trades: List[Dict]) -> float:
        """
        Scale position size up on winning streaks, down on losing streaks.

        Streak   │ Wins  │ Multiplier
        ─────────┼───────┼──────────
        ≥ 5 wins │  ✓    │ × 1.30
        ≥ 3 wins │       │ × 1.15
        Normal   │       │ × 1.00
        ≥ 3 loss │       │ × 0.75
        ≥ 5 loss │       │ × 0.55
        """
        if len(recent_trades) < 3:
            return 1.0

        # Count consecutive wins/losses from the most recent trade backwards
        tail = recent_trades[-10:]
        streak    = 0
        direction = None

        for trade in reversed(tail):
            pnl = trade.get("pnl", 0)
            is_win = pnl > 0

            if direction is None:
                direction = "win" if is_win else "loss"
                streak    = 1
            elif (is_win and direction == "win") or (not is_win and direction == "loss"):
                streak += 1
            else:
                break

        if direction == "win":
            if streak >= 5:   return 1.30
            if streak >= 3:   return 1.15
            return 1.05       # single/two wins: slight bump
        else:
            if streak >= 5:   return 0.55
            if streak >= 3:   return 0.75
            return 0.85       # 1-2 losses: mild reduction

    # ─────────────────────────────────────────────────────────

    def volatility_adjusted_size_ratio(
        self,
        current_atr:     float,
        historical_atr:  float,
    ) -> float:
        """
        Return a multiplier based on the ratio of current ATR to long-run ATR.

        vol_ratio > 2.0 → × 0.50 (very high vol → half size)
        vol_ratio > 1.5 → × 0.70
        vol_ratio 0.5–1.5 → × 1.00 (normal)
        vol_ratio < 0.5 → × 1.20 (low vol → bigger size)
        """
        if historical_atr <= 0:
            return 1.0
        vol_ratio = current_atr / historical_atr

        if vol_ratio > 2.0:   return 0.50
        if vol_ratio > 1.5:   return 0.70
        if vol_ratio < 0.50:  return 1.20
        return 1.00

    # ─────────────────────────────────────────────────────────

    def calculate_optimal_position_size(
        self,
        entry_price:      float,
        stop_loss:        float,
        recent_trades:    List[Dict],
        market_bias:      str   = "NEUTRAL",
        regime:           str   = "NORMAL",
        current_atr:      float = 0.0,
        historical_atr:   float = 0.0,
        signal_quality:   float = 70.0,   # 0-100 quality score
        leverage:         float = 1.0,
        base_risk_pct:    float = 0.02,   # 2% base risk
    ) -> Dict:
        """
        Unified optimal position sizing combining all 4 adjustments.

        Final risk % = base_risk_pct
                       × kelly_mult       (recent perf + market bias)
                       × streak_mult      (win/loss streak momentum)
                       × vol_mult         (current vs historical ATR)
                       × quality_factor   (signal quality 0.5–1.0 band)

        Capped between 0.5% and 5% of account balance.

        Returns dict with:
          risk_amount      : dollar risk this trade
          risk_pct         : final risk as % of account
          position_size    : PositionSize object
          kelly_mult       : float
          streak_mult      : float
          vol_mult         : float
          quality_factor   : float
          breakdown        : human-readable string
        """
        # ── Component multipliers ────────────────────────────
        kelly_mult   = self.calculate_dynamic_kelly(
            recent_trades, market_bias, regime
        )
        streak_mult  = self.get_streak_multiplier(recent_trades)
        vol_mult     = self.volatility_adjusted_size_ratio(current_atr, historical_atr)

        # Signal quality: map 0-100 → 0.5 … 1.0 band
        q            = max(0.0, min(signal_quality, 100.0))
        quality_factor = 0.50 + 0.50 * (q / 100.0)

        # ── Final risk percentage ────────────────────────────
        final_risk_pct = (
            base_risk_pct
            * kelly_mult
            * streak_mult
            * vol_mult
            * quality_factor
        )

        # Hard caps: 0.5% floor, 5% ceiling
        final_risk_pct = float(np.clip(final_risk_pct, 0.005, 0.05))

        # ── Dollar risk & position ───────────────────────────
        risk_amount   = self.account_balance * final_risk_pct
        position_size = self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_amount=risk_amount,
            leverage=leverage,
        )

        breakdown = (
            f"base {base_risk_pct:.1%}"
            f" × kelly {kelly_mult:.3f}"
            f" × streak {streak_mult:.2f}"
            f" × vol {vol_mult:.2f}"
            f" × quality {quality_factor:.2f}"
            f" = {final_risk_pct:.2%}"
        )

        return {
            "risk_amount":    risk_amount,
            "risk_pct":       final_risk_pct * 100,
            "position_size":  position_size,
            "kelly_mult":     kelly_mult,
            "streak_mult":    streak_mult,
            "vol_mult":       vol_mult,
            "quality_factor": quality_factor,
            "breakdown":      breakdown,
        }

    # ══════════════════════════════════════════════════════════
    # STOP LOSS & TAKE PROFIT
    # ══════════════════════════════════════════════════════════

    def calculate_sl_tp(self,
                        entry_price: float,
                        direction: str,
                        atr: float,
                        profile: str = "normal") -> SLTPLevels:
        """
        ATR-based dynamic SL/TP calculation.

        Profiles:
        - conservative: SL=1.0×ATR, TP1=1.5×ATR, TP2=2.5×ATR, TP3=4.0×ATR
        - normal:       SL=1.5×ATR, TP1=2.0×ATR, TP2=3.5×ATR, TP3=5.0×ATR
        - aggressive:   SL=2.0×ATR, TP1=2.5×ATR, TP2=4.0×ATR, TP3=6.0×ATR

        Partial Exit Strategy:
        - TP1: Exit 33%
        - TP2: Exit 33%
        - TP3: Exit 34% (runner)
        """
        multipliers = {
            "conservative": {"sl": 1.0, "tp1": 1.5, "tp2": 2.5, "tp3": 4.0},
            "normal":       {"sl": 1.5, "tp1": 2.0, "tp2": 3.5, "tp3": 5.0},
            "aggressive":   {"sl": 2.0, "tp1": 2.5, "tp2": 4.0, "tp3": 6.0},
        }

        m = multipliers.get(profile, multipliers["normal"])

        if direction == "LONG":
            sl  = entry_price - atr * m["sl"]
            tp1 = entry_price + atr * m["tp1"]
            tp2 = entry_price + atr * m["tp2"]
            tp3 = entry_price + atr * m["tp3"]
        else:  # SHORT
            sl  = entry_price + atr * m["sl"]
            tp1 = entry_price - atr * m["tp1"]
            tp2 = entry_price - atr * m["tp2"]
            tp3 = entry_price - atr * m["tp3"]

        sl_distance = abs(entry_price - sl)
        sl_pct = sl_distance / entry_price
        rr_ratio = abs(tp2 - entry_price) / sl_distance if sl_distance > 0 else 0.0

        return SLTPLevels(
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            sl_distance=sl_distance,
            sl_percentage=sl_pct * 100,
            risk_reward_ratio=rr_ratio,
        )

    # ══════════════════════════════════════════════════════════
    # ADAPTIVE SL / TP MULTIPLIERS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def get_adaptive_sl_tp_multipliers(
        regime: str = "NORMAL",
        volatility_percentile: float = 50.0,
        trend_strength: float = 0.5,
    ) -> Optional[Dict]:
        """
        Adaptive SL/TP multipliers based on regime and market conditions.

        Returns None for HIGH_VOL / EXTREME regimes (caller should skip trade).

        Compared to fixed defaults (SL=1.5, TP3=5.0):
          - Tighter base TPs (1.5 / 2.5 / 4.0) make partials more reachable.
          - Wider SL in trending regimes reduces premature stop-outs.
          - Volatility percentile and trend strength fine-tune both sides.

        Parameters
        ----------
        regime               : "BULL" | "BEAR" | "SIDEWAYS" | "NORMAL" |
                               "HIGH" | "LOW" | "HIGH_VOL" | "EXTREME"
        volatility_percentile: 0–100, e.g. 50 = median vol, 80 = high vol
        trend_strength       : 0–1, proxy for signal quality / confidence
        """
        regime_up = (regime or "NORMAL").upper()

        if regime_up in ("HIGH_VOL", "EXTREME"):
            return None          # skip trade entirely

        # Base multipliers — tighter TPs than the old 2.0/3.5/5.0 for
        # easier partial-exit achievement
        base_sl  = 1.5
        base_tp1 = 1.5
        base_tp2 = 2.5
        base_tp3 = 4.0

        if regime_up in ("BULL", "HIGH"):
            # Trending up: wider SL to survive pullbacks, moderate TP targets
            sl_mult = base_sl * 1.3   # 1.95× ATR — room for volatile upswings
            tp_mult = 1.1             # Less aggressive than 1.2 to keep TP reachable
        elif regime_up == "BEAR":
            # Trending down: same wider SL, moderate TP
            sl_mult = base_sl * 1.3
            tp_mult = 1.1
        elif regime_up in ("SIDEWAYS", "LOW"):
            # Ranging: tighter SL but keep TP1 reachable in the range
            sl_mult = base_sl * 0.8   # 1.2× ATR — tight SL in range
            tp_mult = 1.0             # TP1=1.5×ATR, reachable within the range
        else:                          # NORMAL / unknown
            sl_mult = base_sl
            tp_mult = 1.0

        # Volatility adjustment
        if volatility_percentile > 80:
            sl_mult *= 1.2            # Wider SL when intrabar volatility is high
        elif volatility_percentile < 20:
            sl_mult *= 0.9            # Tighter in quiet markets

        # Let winners run more in strong trends
        if trend_strength > 0.7:
            tp_mult *= 1.3

        # Enforce minimum TP1 = 1.5× ATR so partials are always reachable
        tp1_raw = base_tp1 * tp_mult
        tp2_raw = base_tp2 * tp_mult
        tp3_raw = base_tp3 * tp_mult
        tp1_raw = max(tp1_raw, 1.5)   # Never closer than 1.5× ATR

        return {
            "sl":  round(sl_mult, 4),
            "tp1": round(tp1_raw, 4),
            "tp2": round(tp2_raw, 4),
            "tp3": round(tp3_raw, 4),
        }

    def calculate_adaptive_sl_tp(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        regime: str = "NORMAL",
        volatility_percentile: float = 50.0,
        trend_strength: float = 0.5,
    ) -> Optional["SLTPLevels"]:
        """
        Compute SL/TP levels using adaptive multipliers.

        Returns None if the regime dictates skipping the trade.
        """
        mults = self.get_adaptive_sl_tp_multipliers(
            regime, volatility_percentile, trend_strength
        )
        if mults is None:
            return None

        sign = 1 if direction == "LONG" else -1
        sl  = entry_price - sign * atr * mults["sl"]
        tp1 = entry_price + sign * atr * mults["tp1"]
        tp2 = entry_price + sign * atr * mults["tp2"]
        tp3 = entry_price + sign * atr * mults["tp3"]

        sl_dist = abs(entry_price - sl)
        sl_pct  = sl_dist / entry_price
        rr      = abs(tp2 - entry_price) / sl_dist if sl_dist > 0 else 0.0

        return SLTPLevels(
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            sl_distance=sl_dist,
            sl_percentage=sl_pct * 100,
            risk_reward_ratio=rr,
        )

    # ══════════════════════════════════════════════════════════
    # POSITION SIZE CALCULATION
    # ══════════════════════════════════════════════════════════

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss: float,
                                risk_amount: float,
                                leverage: float = 1.0) -> PositionSize:
        """
        Calculate exact position size.

        Formula:
        SL_distance_pct   = |Entry - SL| / Entry
        Position_notional = Risk_amount / SL_distance_pct
        Margin            = Position_notional / Leverage
        Quantity          = Position_notional / Entry
        """
        sl_distance = abs(entry_price - stop_loss)
        sl_pct = sl_distance / entry_price

        if sl_pct == 0:
            return PositionSize(0.0, 0.0, 0.0, 0.0, "error")

        position_notional = risk_amount / sl_pct
        margin_required = position_notional / leverage
        quantity = position_notional / entry_price

        return PositionSize(
            risk_amount=risk_amount,
            position_notional=position_notional,
            margin_required=margin_required,
            quantity=quantity,
            method="risk_based",
        )

    # ══════════════════════════════════════════════════════════
    # DRAWDOWN SCALING
    # ══════════════════════════════════════════════════════════

    def get_drawdown_multiplier(self) -> Tuple[float, str]:
        """
        Get position multiplier based on current drawdown.

        Scaling Rules:
        - DD < 5%:  100% normal size
        - DD 5-10%:  75% of normal
        - DD 10-15%: 50% of normal
        - DD 15-20%: 25% of normal
        - DD > 20%:   0% (STOP TRADING)
        """
        dd = self.current_drawdown
        levels = self.config['drawdown_scaling']['levels']

        for level in reversed(levels):
            if dd >= level['threshold']:
                if level['multiplier'] == 0:
                    return 0.0, "STOP_TRADING"
                return level['multiplier'], f"DD={dd:.1%}, Size={level['multiplier']:.0%}"

        return 1.0, "Normal"

    def apply_drawdown_scaling(self, base_risk: float) -> Dict:
        """Apply drawdown scaling to risk amount."""
        multiplier, status = self.get_drawdown_multiplier()
        adjusted_risk = base_risk * multiplier

        return {
            "base_risk": base_risk,
            "multiplier": multiplier,
            "adjusted_risk": adjusted_risk,
            "current_drawdown": self.current_drawdown,
            "status": status,
        }

    # ══════════════════════════════════════════════════════════
    # RISK CHECKS
    # ══════════════════════════════════════════════════════════

    def check_can_trade(self) -> Dict:
        """Check all risk limits before allowing a new trade."""
        checks: Dict = {}
        can_trade = True

        # Daily loss check
        daily_loss_pct = abs(self.daily_pnl) / self.account_balance if self.account_balance > 0 else 0
        checks['daily_loss'] = {
            'limit': self.max_daily_loss,
            'current': daily_loss_pct,
            'passed': daily_loss_pct < self.max_daily_loss,
        }
        if not checks['daily_loss']['passed']:
            can_trade = False

        # Max drawdown check
        checks['max_drawdown'] = {
            'limit': self.max_total_drawdown,
            'current': self.current_drawdown,
            'passed': self.current_drawdown < self.max_total_drawdown,
        }
        if not checks['max_drawdown']['passed']:
            can_trade = False

        # Concurrent positions check
        checks['concurrent'] = {
            'limit': self.max_concurrent,
            'current': len(self.open_positions),
            'passed': len(self.open_positions) < self.max_concurrent,
        }
        if not checks['concurrent']['passed']:
            can_trade = False

        return {
            'can_trade': can_trade,
            'checks': checks,
        }

    # ══════════════════════════════════════════════════════════
    # PERFORMANCE METRICS
    # ══════════════════════════════════════════════════════════

    def calculate_sharpe_ratio(self,
                               returns: pd.Series,
                               risk_free_rate: float = 0.05,
                               periods_per_year: int = 252) -> float:
        """
        Sharpe Ratio = (Return - Rf) / StdDev × √periods

        Target: > 1.5
        """
        excess = returns - (risk_free_rate / periods_per_year)
        std = excess.std()
        if std == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * excess.mean() / std)

    def calculate_sortino_ratio(self,
                                returns: pd.Series,
                                risk_free_rate: float = 0.05,
                                periods_per_year: int = 252) -> float:
        """
        Sortino = (Return - Rf) / Downside_StdDev × √periods

        Better for asymmetric returns.
        """
        excess = returns - (risk_free_rate / periods_per_year)
        downside = returns[returns < 0].std()
        if downside == 0:
            return float('inf')
        return float(np.sqrt(periods_per_year) * excess.mean() / downside)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict:
        """Calculate max drawdown from equity curve."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max

        max_dd = drawdowns.min()
        max_dd_idx = drawdowns.idxmin()

        return {
            "max_drawdown": abs(float(max_dd)),
            "max_dd_date": max_dd_idx,
            "current_drawdown": abs(float(drawdowns.iloc[-1])),
        }

    def calculate_calmar_ratio(self,
                               returns: pd.Series,
                               equity_curve: pd.Series) -> float:
        """
        Calmar = Annual_Return / Max_Drawdown

        Target: > 1.0
        """
        annual_return = float(returns.mean() * 252)
        max_dd = self.calculate_max_drawdown(equity_curve)["max_drawdown"]

        if max_dd == 0:
            return float('inf')
        return annual_return / max_dd

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Profit Factor = Total_Wins / Total_Losses

        Target: > 1.5
        """
        wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

        if losses == 0:
            return float('inf')
        return wins / losses

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """
        Expectancy = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss)

        In R terms.  Target: > 0.3R
        """
        if not trades:
            return 0.0

        wins   = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]

        win_rate  = len(wins) / len(trades)
        loss_rate = 1 - win_rate

        avg_win_r  = float(np.mean([t['pnl_r'] for t in wins]))  if wins   else 0.0
        avg_loss_r = float(np.mean([abs(t['pnl_r']) for t in losses])) if losses else 0.0

        return (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    def full_performance_report(self,
                                trades: List[Dict],
                                equity_curve: pd.Series) -> Dict:
        """
        Aggregate all performance metrics into one dict.
        """
        returns = equity_curve.pct_change().dropna()

        return {
            "sharpe":          self.calculate_sharpe_ratio(returns),
            "sortino":         self.calculate_sortino_ratio(returns),
            "calmar":          self.calculate_calmar_ratio(returns, equity_curve),
            "max_drawdown":    self.calculate_max_drawdown(equity_curve)["max_drawdown"],
            "profit_factor":   self.calculate_profit_factor(trades),
            "expectancy_r":    self.calculate_expectancy(trades),
            "total_trades":    len(trades),
            "win_rate":        len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
        }

    # ══════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def update_balance(self, pnl: float):
        """Update account balance after trade."""
        self.account_balance += pnl
        self.daily_pnl += pnl

        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        self.current_drawdown = (
            (self.peak_balance - self.account_balance) / self.peak_balance
            if self.peak_balance > 0 else 0.0
        )

    def reset_daily(self):
        """Reset daily PnL (call at start of new day)."""
        self.daily_pnl = 0.0

    def reset_all(self, initial_capital: float = None):
        """Reset all state."""
        capital = initial_capital or self.initial_capital
        self.account_balance = capital
        self.initial_capital = capital
        self.peak_balance = capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.open_positions = []
