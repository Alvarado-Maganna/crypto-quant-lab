# Experiment: position sizing with a no-edge entry rule (notebook 11)

- **Date started:** 2026-11-25
- **Status:** concluded (descriptive simulation, no strategy, no edge by design)
- **Branch / PR:** `feat/11-position-sizing`

## 1. Hypothesis
No prior hypotheses were recorded. The notebook answers questions Q1–Q9, listed in its
introduction: look-ahead checks, max drawdown, final equity, ruin, planned vs realized stop
losses, losing streaks, the size chosen by volatility targeting, block vs i.i.d. resampling, and
realized exposure.

## 2. Data
- Binance spot BTC/USDT daily candles via ccxt (`src/data/ohlcv.py`), reusing notebook 10's cache
  in `data/10_regimes/`: 2017-08-17 → 2026-11-24 (UTC), 3,387 candles, no missing days.
- 3,386 day tuples `(open, high, low, close) / previous close`, resampled as whole days.
- No in-sample / out-of-sample split: nothing is fitted. The only full-sample quantities are the
  mean daily return (used to demean) and the median σ30 (used to scale σ_target).

## 3. Method
- 10,000 paths × 1,095 days, seed 11. Main: stationary block bootstrap (mean block 20 days);
  control: i.i.d. days. Raw and demeaned returns. Common random numbers (same paths and coin
  flips for every scheme).
- Entry: when flat, at each close, p = 0.20 to buy at the next open (30-day warm-up). Exit: 5% stop
  or close of day entry + 20. Stop fill: open if gapped, else stop × (1 − slippage).
- Schemes: all-in; fixed 1% risk (20% of equity); vol targeting min(1, σ_target / σ30) with
  σ_target = 0.20 × full-sample median σ30 (54.8% → 11.0%); buy & hold.
- Costs: 10 bps per side on notional; slippage 0 bps (main) and 100 bps (stress). Spot, no
  leverage, no funding.
- Look-ahead: σ30 and the whole simulator recomputed on truncated paths (k = 60, 365, 730): 0
  differences.
- Post-inspection addition: an exposure-matched comparison of schemes 3 and 2 (σ_target rescaled
  per case so both hold the same average exposure), because the pre-specified σ_target gives
  scheme 3 1.39 pp less exposure under i.i.d.

## 4. Benchmarks
- Buy & hold: same paths (block, raw): median final equity 22,672, P(ruin) 35.4%.
- Simple model: fixed 1% risk is the simple sizing rule that vol targeting must beat.
- Random selection: the entry rule itself is random.

## 5. Results (block bootstrap, raw returns, 0 bps, 10,000 paths)
| Metric | All-in | Fixed 1% risk | Vol targeting | Buy & hold |
|---|---|---|---|---|
| Max DD median / p99 | 60.0% / 89.7% | 16.1% / 35.0% | 15.7% / 34.4% | 65.4% / 93.5% |
| Final equity median | 14,157 | 11,550 | 11,699 | 22,672 |
| % paths below 10,000 | 36.3% | 25.2% | 23.7% | 26.3% |
| P(ruin) | 33.2% | 0.00% | 0.00% | 35.4% |
| Time-weighted exposure | 64.6% | 13.6% | 13.8% | 100% |

- Demeaned: P(ruin) all-in 64.9% vs fixed risk 0.03%; median final equity 5,634 vs 9,489.
- Stops (fixed risk): realized loss median 1.04% of equity (max 1.09%) at 0 bps, 1.23% at 100 bps;
  2 gap fills out of 522,111 stops (daily bars cannot see intraday moves).
- Losing streaks: median longest 11 trades vs median log2(n) 6.2 (loss rate 72.0%).
- Vol targeting vs fixed risk, paired mean Δ max DD: pre-specified −0.47 pp (block) / −1.35 pp
  (i.i.d., lower exposure); exposure-matched −0.65 pp (block) / +0.42 pp (i.i.d.).

## 6. Bias checklist
- [x] No look-ahead (truncation tests; σ30 at entry uses the 30 returns before the entry day)
- [ ] No survivorship bias: not applicable (one asset), but the asset is BTC, the survivor
- [x] Costs: fees and slippage included; daily bars understate real stop slippage
- [x] Variants counted: 4 schemes × 2 resamplings × 2 return versions × 2 slippages, all reported;
  1 post-inspection check, labelled

## 7. Conclusion
With the same no-edge entries, the size alone moves P(ruin) from about a third to zero. Volatility
targeting adds only a small drawdown reduction (about 0.6 pp at equal exposure), which goes away
when clustering is destroyed. Stop-loss results from daily bars are a lower bound. Next step:
model intraday stop fills (minute data, as in notebook 08) and add leverage/liquidation (notebook
04) before any sizing rule is used on perpetuals.
