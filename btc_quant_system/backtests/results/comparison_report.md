# OPTIMIZATION COMPARISON REPORT
> Generated: 2026-03-06 15:08:11  
> Period: 2019-2024 (6 years) | Initial Capital: $100,000 per year  

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZATION COMPARISON REPORT                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  METRIC              │ BASELINE    │ OPTIMIZED   │ CHANGE      │ STATUS  ║
║  ────────────────────┼─────────────┼─────────────┼─────────────┼──────── ║
║  Total Return        │ +29.54%     │   +109.41% │     +79.87% │  ✅     ║
║  Win Rate            │ 30.2%       │     61.27% │     +31.07% │  ✅     ║
║  Max Drawdown        │ 13.84%      │     39.49% │     +25.65% │  ❌     ║
║  Sharpe Ratio        │ 0.944       │     -0.355 │      -1.30 │  ❌     ║
║  Profit Factor       │ N/A         │      1.867 │          — │  —      ║
║  vs Buy & Hold       │ -194%       │  -2461.59% │   -2267.59% │  ❌     ║
║  SL Exit Rate        │ 89%         │     22.54% │     -66.46% │  ✅     ║
║  Avg Win (R)         │ 3.2R        │      1.00R │      -2.20R │  ❌     ║
║  Avg Loss (R)        │ 1.04R       │      1.37R │      +0.33R │  ❌     ║
║  Total Trades        │ 116         │        142 │     +26.00 │  ✅     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Year-by-Year Breakdown

| Year | Period | Trades | Win Rate | PnL | Return | Max DD | Sharpe | Notes |
|------|--------|--------|----------|-----|--------|--------|--------|-------|
| 2019 | Bull recovery | 4 | 50.0% | $+7,691 | +7.69% | 6.36% | -1.06 | — |
| 2020 | COVID crash + recovery | 22 | 63.6% | $+17,992 | +17.99% | 8.75% | -0.55 | — |
| 2021 | Peak bull | 31 | 54.8% | $+6,910 | +6.91% | 14.87% | -0.70 | — |
| 2022 | Bear market | 35 | 60.0% | $+43,584 | +43.58% | 9.75% | -0.09 | — |
| 2023 | Recovery | 23 | 60.9% | $-21,067 | -21.07% | 39.49% | -0.44 | — |
| 2024 | ETF bull | 27 | 70.4% | $+54,302 | +54.30% | 10.91% | 0.03 | — |

## Final Assessment

| Target | Goal | Result | Status |
|--------|------|--------|--------|
| Win Rate > 50% | > 50% | 61.27% | ✅ |
| Max DD < 20% | < 20% | 39.49% | ❌ |
| Beat Buy & Hold | > 0% vs B&H | -2461.59% | ❌ |
| Sharpe > 1.0 | > 1.0 | -0.35 | ❌ |
| Profit Factor > 1.5 | > 1.5 | 1.87 | ✅ |

---
*Config: min_quality_score=70, max_risk=3%, max_dd=20%, TrendAwareEngine, PullbackEntry, AdaptiveSLTP, PartialExits*