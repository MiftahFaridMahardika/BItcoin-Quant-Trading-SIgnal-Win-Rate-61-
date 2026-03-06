# BTC Backtest Performance Report
## Period: 2023-01-01 → 2024-12-31 (Out-of-Sample)

### Scorecard: 1/9 targets met

### Capital
| Metric | Value |
|--------|-------|
| Initial Capital | $100,000.00 |
| Final Equity | $95,109.10 |
| Total P&L | $-4,890.90 |
| Total Return | -4.89% |
| CAGR | -2.48% |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | 41 |
| Winning | 12 |
| Losing | 29 |
| Win Rate | 29.3% |
| Avg Win | $3,673.50 (4.53R) |
| Avg Loss | $1,688.72 (1.04R) |
| Max Consec Losses | 8 |

### Risk Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe Ratio | -1.22 | >1.5 | ❌ |
| Sortino Ratio | -0.47 | >2.0 | ❌ |
| Calmar Ratio | -0.16 | >1.0 | ❌ |
| Max Drawdown | 15.3% | <15% | ❌ |
| Profit Factor | 0.90 | >1.5 | ❌ |
| Expectancy | 0.59R | >0.3R | ✅ |

### Strategy vs Buy & Hold
| Metric | Strategy | Buy & Hold | Diff |
|--------|----------|------------|------|
| Total Return | -4.89% | 459.38% | -464.27% |
| CAGR | -2.48% | 136.65% | -139.13% |
| Max Drawdown | 15.3% | 30.1% | -14.9% |
| Sharpe | -1.22 | 0.38 | -1.61 |

### Charts
- **Equity + DD + B&H**: `charts/equity_curve.png`
- **Monthly heatmap**: `charts/monthly_returns.png`
- **PnL histogram**: `charts/trade_distribution.png`
- **DD timeline**: `charts/drawdown_analysis.png`
- **Direction breakdown**: `charts/signal_performance.png`
- **Rolling WR + Cum PnL**: `charts/rolling_metrics.png`
