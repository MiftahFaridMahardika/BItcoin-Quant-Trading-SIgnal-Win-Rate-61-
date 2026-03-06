# Technical Documentation

## BTC Quant Trading System — Deep Dive

---

## Table of Contents
1. [Signal Generation Methodology](#1-signal-generation-methodology)
2. [Feature Engineering Pipeline](#2-feature-engineering-pipeline)
3. [Risk Management Framework](#3-risk-management-framework)
4. [Machine Learning Architecture](#4-machine-learning-architecture)
5. [Backtesting Infrastructure](#5-backtesting-infrastructure)
6. [Performance Analysis](#6-performance-analysis)
7. [System Design Decisions](#7-system-design-decisions)

---

## 1. Signal Generation Methodology

### 1.1 Multi-Layer Scoring System

The signal engine implements a **weighted voting mechanism** across 11 signal checkers, organized into 5 analytical layers. This design mirrors institutional quant strategies that aggregate multiple alpha factors.

#### Weight Allocation Philosophy

```
Total Score Range: -19 to +19
Classification Thresholds:
  STRONG_LONG  : score ≥ +8
  LONG         : score ≥ +4
  SKIP         : -4 < score < +4
  SHORT        : score ≤ -4
  STRONG_SHORT : score ≤ -8
```

The weight distribution reflects the relative importance of each signal category:

| Layer | Weight | Rationale |
|-------|--------|-----------|
| Trend (L1) | 7/19 (37%) | Trend is the primary alpha source in crypto |
| Momentum (L2) | 5/19 (26%) | Secondary confirmation, mean reversion detection |
| Volume (L4) | 3/19 (16%) | Validation of price movements |
| Volatility (L3) | 2/19 (11%) | Regime identification, not direction |
| Price Action (L5) | 2/19 (11%) | Microstructure signals |

### 1.2 Individual Signal Checkers

#### Trend Layer (L1)

**EMA Structure Checker** (Weight: 3)
```python
# Triple EMA stack: EMA21, EMA55, EMA200
Conditions:
  Bullish (+1): EMA21 > EMA55 > EMA200 (uptrend alignment)
  Bearish (-1): EMA21 < EMA55 < EMA200 (downtrend alignment)
  Neutral (0):  Mixed alignment

Crossover Signals:
  Golden Cross (+1): EMA21 crosses above EMA55
  Death Cross (-1): EMA21 crosses below EMA55
```

**HMA Direction** (Weight: 2)
- Hull Moving Average for reduced lag
- Slope calculation over 3-bar window
- Direction persistence filter

**Supertrend** (Weight: 2)
- ATR-based trend following
- Configurable multiplier (default: 3.0)
- State persistence to reduce whipsaws

#### Momentum Layer (L2)

**RSI Divergence** (Weight: 2)
```python
Bullish Divergence (+1):
  - Price makes lower low
  - RSI makes higher low
  - RSI < 40 (oversold zone)

Bearish Divergence (-1):
  - Price makes higher high
  - RSI makes lower high
  - RSI > 60 (overbought zone)
```

**MACD Histogram** (Weight: 2)
- Histogram direction and magnitude
- Zero-line crossovers
- Momentum acceleration/deceleration

**Z-Score** (Weight: 1)
- Mean reversion signal
- Z > +2: Overbought (-1)
- Z < -2: Oversold (+1)

### 1.3 Volatility Regime Filter

The volatility regime acts as a **blocking filter**, not a scoring component:

| Regime | ATR Ratio | Action |
|--------|-----------|--------|
| LOW | < 0.7 | Normal trading |
| NORMAL | 0.7 – 1.3 | Normal trading |
| HIGH | 1.3 – 2.0 | Reduce position size |
| EXTREME | > 2.0 | **BLOCK all signals** |

This prevents trading during:
- Flash crashes
- Extreme volatility events
- Low liquidity periods

---

## 2. Feature Engineering Pipeline

### 2.1 Causal Computation Guarantee

All features are computed **causally** — row N only uses data from rows ≤ N. This eliminates look-ahead bias by design.

```python
# Example: EMA calculation
ema_21 = close.ewm(span=21, adjust=False).mean()
# adjust=False ensures causal (recursive) computation
```

### 2.2 Feature Categories

#### L1: Trend Features (12 indicators)
```python
EMA System:
  - ema_21, ema_55, ema_200
  - ema_21_55_cross ({-1, 0, +1})
  - ema_stack_signal (alignment score)

HMA System:
  - hma_16, hma_64
  - hma_slope, hma_signal

Supertrend:
  - supertrend_line, supertrend_dir ({-1, +1})

Ichimoku Cloud:
  - tenkan_sen, kijun_sen, senkou_span_a/b
  - ichi_signal ({-1, 0, +1})
```

#### L2: Momentum Features (20 indicators)
```python
RSI Variants:
  - rsi_14, rsi_divergence_bull/bear

MACD:
  - macd_line, signal_line, histogram
  - macd_cross ({-1, 0, +1})

Stochastic:
  - stoch_k, stoch_d, stoch_cross

Rate of Change:
  - roc_10, roc_30 (percentage returns)

Z-Score:
  - zscore_20 (mean reversion)

Williams %R, CCI, Ultimate Oscillator
```

#### L3: Volatility Features (14 indicators)
```python
ATR System:
  - atr_14, atr_pct, atr_ratio
  - vol_regime ({0, 1, 2, 3})

Bollinger Bands:
  - bb_upper, bb_lower, bb_middle
  - bb_pct_b (position within bands)
  - bb_bandwidth (volatility measure)

Keltner Channels:
  - kc_upper, kc_lower, kc_middle

Historical Volatility:
  - hist_vol_20 (annualized)
```

#### L4: Volume Features (10 indicators)
```python
Volume Analysis:
  - volume_sma_20, vol_ratio
  - vol_trend ({-1, 0, +1})

MFI (Money Flow Index):
  - mfi_14 (volume-weighted RSI)

CMF (Chaikin Money Flow):
  - cmf_20 (-1 to +1 range)

OBV (On-Balance Volume):
  - obv, obv_sma, obv_signal

VWAP:
  - vwap, vwap_signal
```

#### L5: Price Action Features (10 indicators)
```python
Returns:
  - ret_1, ret_6, ret_24, ret_168 (1h, 6h, 1d, 1w)

Price Structure:
  - trend_structure ({-1, 0, +1})
  - support_resistance proximity

Candlestick Patterns:
  - doji, hammer, engulfing detection
```

### 2.3 Feature Selection for ML

Not all features are ML-safe. Raw prices, cumulative indicators (OBV), and pre-computed signals are excluded to prevent:
- Scale sensitivity
- Data leakage
- Redundant information

**32 ML-Safe Features Selected:**
- Normalized indicators (RSI, Stochastic, Williams %R)
- Ordinal signals ({-1, 0, +1})
- Ratios and percentages
- Bounded measures (0-1 or -1 to +1)

---

## 3. Risk Management Framework

### 3.1 Kelly Criterion Implementation

The Kelly Criterion determines optimal position sizing based on historical edge:

```
Full Kelly Formula:
  f = (W × R - L) / R
  
Where:
  W = Win rate (probability of winning)
  L = Loss rate (1 - W)
  R = Average win / Average loss (reward/risk ratio)

Half-Kelly (Safer):
  f_half = f × 0.5
```

**Example Calculation:**
```
Win Rate: 58%
Avg Win: 2.3R
Avg Loss: 1.0R

Full Kelly = (0.58 × 2.3 - 0.42 × 1.0) / 2.3 = 39.7%
Half-Kelly = 39.7% × 0.5 = 19.85%
```

The system caps exposure at 30% of portfolio per position.

### 3.2 Dynamic Stop-Loss & Take-Profit

#### ATR-Based Calculation
```python
atr = average_true_range(period=14)

# Stop Loss
sl_distance = atr × sl_multiplier  # default: 1.5

# Take Profit Levels (tiered exit)
tp1_distance = sl_distance × tp1_multiplier  # 2.0×
tp2_distance = sl_distance × tp2_multiplier  # 3.5×
tp3_distance = sl_distance × tp3_multiplier  # 5.0×
```

#### Tiered Exit Strategy
| Level | Distance | Exit % | R:R Ratio |
|-------|----------|--------|-----------|
| TP1 | 2.0× SL | 33% | 2:1 |
| TP2 | 3.5× SL | 33% | 3.5:1 |
| TP3 | 5.0× SL | 34% | 5:1 |

**Expected Value per Trade:**
```
EV = (0.33 × 2R) + (0.33 × 3.5R) + (0.34 × 5R) - (1.0 × 1R)
EV = 0.66R + 1.16R + 1.70R - 1.0R
EV = +2.52R (positive expectancy)
```

### 3.3 Drawdown Protection

```yaml
Drawdown Scaling Levels:
  - threshold: 0.05  (5% DD)  → multiplier: 1.00 (Full size)
  - threshold: 0.10  (10% DD) → multiplier: 0.75 (75% size)
  - threshold: 0.15  (15% DD) → multiplier: 0.50 (50% size)
  - threshold: 0.20  (20% DD) → multiplier: 0.25 (25% size)
  - threshold: 0.25  (25% DD) → multiplier: 0.00 (Trading halted)
```

This creates a **convex risk profile** — reducing exposure when the system is performing poorly and increasing it when performing well.

### 3.4 Circuit Breakers

- **Consecutive Losses**: Trading halted after 5 consecutive losses
- **Daily Loss Limit**: Trading halted after 5% daily drawdown
- **Cool-down Period**: 24-hour pause before resuming

---

## 4. Machine Learning Architecture

### 4.1 Model Ensemble

The system uses a **heterogeneous ensemble** of four models with different inductive biases:

| Model | Strength | Weakness |
|-------|----------|----------|
| XGBoost | Best for tabular, handles non-linearity | Can overfit |
| LightGBM | Fast training, leaf-wise growth | Sensitive to outliers |
| Random Forest | Stable, resistant to overfit | Can underfit |
| MLP | Captures complex patterns | Needs more data |

### 4.2 Training Methodology

#### Chronological Split (No Shuffle)
```
Train: 2017-01-01 to 2020-12-31 (4 years)
Val:   2021-01-01 to 2021-12-31 (1 year)
Test:  2022-01-01 to 2024-12-31 (3 years)
```

Shuffling would introduce **look-ahead bias** — the model would learn patterns from the future to predict the past.

#### Label Generation
```python
# Forward returns
target_return = (future_close - current_close) / current_close

# Classification labels
if target_return > +threshold:  label = +1  (BUY)
elif target_return < -threshold: label = -1  (SELL)
else:                           label = 0   (HOLD)
```

Threshold is typically set to the 75th percentile of absolute returns to create balanced classes.

### 4.3 Hyperparameter Optimization

**Optuna** Bayesian optimization is used for hyperparameter tuning:

```python
# XGBoost search space
max_depth: [3, 4, 5, 6, 7]
learning_rate: [0.01, 0.05, 0.1]
n_estimators: [100, 200, 300]
subsample: [0.8, 0.9, 1.0]
colsample_bytree: [0.8, 0.9, 1.0]

# Optimization objective
maximize: F1-score (validation set)
trials: 100
```

### 4.4 Ensemble Voting

**Soft Voting** (probability-based):
```python
# Individual model probabilities
prob_xgb = model_xgb.predict_proba(X)
prob_lgb = model_lgb.predict_proba(X)
prob_rf = model_rf.predict_proba(X)
prob_mlp = model_mlp.predict_proba(X)

# Weighted average
final_prob = (
    0.35 × prob_xgb +
    0.30 × prob_lgb +
    0.20 × prob_rf +
    0.15 × prob_mlp
)

# Final prediction
prediction = argmax(final_prob)
confidence = max(final_prob)
```

---

## 5. Backtesting Infrastructure

### 5.1 State Machine Architecture

The execution engine uses a **finite state machine** to manage trade lifecycle:

```
States:
  IDLE → SCANNING → SIGNAL_FOUND → PENDING_ENTRY → IN_TRADE → CLOSING
   ↑                                                          ↓
   └──────────────────────────────────────────────────────────┘

Transitions:
  SCANNING → SIGNAL_FOUND: Signal score crosses threshold
  SIGNAL_FOUND → PENDING_ENTRY: Entry filter validation
  PENDING_ENTRY → IN_TRADE: Price enters entry zone
  IN_TRADE → CLOSING: SL, TP, or exit condition hit
  CLOSING → SCANNING: Position closed, resume scanning
```

### 5.2 Walk-Forward Optimization

WFO addresses the **data mining bias** problem in strategy optimization:

```
Window 1: Train[2017-2018] → Test[2019]
Window 2: Train[2018-2019] → Test[2020]
Window 3: Train[2019-2020] → Test[2021]
...
Window 6: Train[2022-2023] → Test[2024]
```

**Benefits:**
- Parameters are always optimized on past data
- Tests on truly unseen future data
- Measures strategy degradation over time

### 5.3 Monte Carlo Simulation

MC simulation tests the **statistical significance** of backtest results:

```python
# Generate 1,000 equity curve permutations
for i in range(1000):
    # Shuffle trade sequence (not trade outcomes)
    shuffled_trades = random.sample(original_trades, len(trades))
    
    # Rebuild equity curve
    mc_equity = build_equity_curve(shuffled_trades)
    
    # Record metrics
    mc_returns.append(final_return(mc_equity))
    mc_drawdowns.append(max_drawdown(mc_equity))
```

**Analysis Outputs:**
- 95% confidence interval for final equity
- Probability of exceeding max drawdown
- Risk of ruin estimation

---

## 6. Performance Analysis

### 6.1 Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sharpe Ratio | (R_p - R_f) / σ_p | Risk-adjusted return |
| Sortino Ratio | (R_p - R_f) / σ_d | Downside risk focus |
| Calmar Ratio | CAGR / Max DD | Return vs worst drawdown |
| Profit Factor | Gross Profit / Gross Loss | Payout ratio |
| Expectancy | (Win% × Avg Win) - (Loss% × Avg Loss) | Expected value per trade |

### 6.2 Consistency Score

The consistency score measures **robustness** across walk-forward windows:

```python
# Coefficient of Variation (CV)
cv = std(metric) / mean(metric)

# Consistency (0-1 scale)
consistency = 1 - min(cv, 1.0)
```

Higher consistency indicates the strategy is less sensitive to:
- Market regime changes
- Parameter variations
- Time periods

---

## 7. System Design Decisions

### 7.1 Why 4-Hour Timeframe?

The primary timeframe is 4H because:
1. **Noise Reduction**: Filters 1-minute noise while maintaining responsiveness
2. **Sleep-Friendly**: 6 candles per day vs 1,440 (1-min)
3. **Trend Capture**: Captures multi-day moves without excessive churn
4. **Institutional Alignment**: Many funds use 4H for swing trading

### 7.2 Why 71 Indicators?

More indicators ≠ better. The 71 indicators are carefully selected to:
1. **Capture different market phenomena**: trend, momentum, volatility, volume
2. **Provide redundancy**: Multiple confirming signals reduce false positives
3. **Enable ML feature selection**: Rich feature space for model training

### 7.3 Why Kelly Criterion?

Kelly Criterion provides **optimal growth** under geometric Brownian motion assumptions. In practice, Half-Kelly is used because:
1. Real returns have fatter tails than normal
2. Parameter estimation error exists
3. Psychological drawdown tolerance

### 7.4 Why Ensemble ML?

Single models fail in different ways. Ensemble provides:
1. **Error diversification**: Different models make different mistakes
2. **Stability**: Smoother equity curves
3. **Confidence calibration**: Probability estimates are more reliable

---

## 8. Known Limitations

1. **Execution Assumption**: Assumes market orders fill at close price
2. **Slippage Model**: Fixed slippage may underestimate in volatile periods
3. **Liquidity Assumption**: Assumes position size doesn't move the market
4. **Regime Changes**: Strategy may degrade in unprecedented market conditions

---

## 9. Future Enhancements

| Priority | Enhancement | Impact |
|----------|-------------|--------|
| High | Live data integration | Production readiness |
| High | Realistic fill simulation | Accuracy improvement |
| Medium | Multi-timeframe fusion | Signal quality |
| Medium | Alternative data (on-chain) | Alpha generation |
| Low | Reinforcement learning | Adaptive behavior |

---

*Document Version: 1.0*
*Last Updated: March 2026*
