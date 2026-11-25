# Experiment: volatility of BTC and ETH (notebook 09)

- **Date started:** 2026-11-24
- **Status:** concluded (descriptive study, no strategy)
- **Branch / PR:** `feat/09-volatility`

## 1. Hypothesis
No hypotheses were recorded. The notebook answers questions H1–H7, which are listed in its
introduction: current level vs history, ETH vs BTC, tails, clustering, the √t rule, weekend and
hour-of-week, and cost vs typical move.

## 2. Data
- Binance spot BTC/USDT and ETH/USDT, daily and hourly candles via ccxt (`src/data/ohlcv.py`).
- Open times 2018-01-01 00:00 → 2026-11-23 23:00 UTC (last complete day 2026-11-23).
- Coverage: daily 3,249 / 3,249 candles. Hourly 77,854 / 77,976 (122 missing hours in 27 runs, the
  same hours for both assets, 2018–2023). No duplicates. The daily close equals the 23:00 hourly
  close on every day that has one.

## 3. Method
- Simple returns, std with ddof=1, annualized with √365 (daily) and √(365·24) (hourly).
- Past-only σ: std of the 30 returns before day t; z_t = r_t / σ_{t-1}. A full-sample σ appears
  only in one reference table, labelled as look-ahead.
- Post-inspection choices: (a) no return across a missing hourly candle (NaN instead); (b) weekly
  √t check grouped by quarter (a month has only 4–5 weeks); (c) tail table adds the count expected
  under Student-t(29), the distribution of z for normal returns when σ is estimated from 30 days.
- Costs (Part 7) are carried over from notebook 07 (taker → taker, BTC, 1k–10k): spot 20.00 bps,
  perp 10.02 bps without funding.

## 4. Benchmarks
Not applicable (no strategy). The normal and t(29) distributions are the reference for the tails.

## 5. Results
See the Findings table and the key takeaways of `notebooks/09_volatility.ipynb`.

## 6. Bias checklist
- [x] No look-ahead: z and the "big day" label use only the 30 previous days. Full-sample σ appears
  only in a labelled reference row.
- [x] Survivorship: BTC and ETH were picked by the brief, not by their results. Both are still large
  assets today, so the conclusions describe survivors.
- [x] Costs: taken from notebook 07, not re-measured. Funding is not included.
- [x] Variants: one window (30 days) and one threshold (|z| > 2), fixed by the brief. Nothing was tuned.

## 7. Conclusion
Daily volatility clusters and has fat tails that a normal distribution, or a 30-day σ estimate,
does not explain. The √t rule holds for hourly → daily in the median month, and holds a little
less well for daily → weekly. Next: use past-only σ for position sizing in the risk module, and
check whether the clustering survives transaction costs before using it as a signal.
