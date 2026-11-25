# Experiment: liquidity by hour (notebook 08)

- **Date started:** 2026-11-24
- **Status:** concluded (descriptive study, no strategy)
- **Branch / PR:** `feat/08-liquidity-by-hour`

## 1. Hypothesis
No prior hypotheses were recorded. The notebook answers questions H1–H7, which are listed in its
introduction: hour-of-week depth, whether the hour of notebooks 06/07 was typical, spot vs perp,
book cost by hour, volume vs depth, the hack and 10/10 periods, and stress vs normal day.

## 2. Data
- Own collector: Binance spot BTC/USDT, ETH/USDT, PENGU/USDT and USD-M BTC/USDT:USDT, one snapshot
  per minute, 2026-11-21 23:46 → 2026-11-24 00:10 UTC.
- data.binance.vision: USD-M BTCUSDT `bookDepth` (2025-11-24 → 2026-11-23, 2025-10-06 → 10-15,
  2026-11-19), spot 1-minute klines (BTCUSDT, ETHUSDT 12 months; PENGUUSDT 2 days), USD-M BTCUSDT
  1-minute klines (2026-08-01 → 11-23).

## 3. Method
- `bookDepth` validated against the collector before use: ratio ≈ 1 per side, best time shift 0 h.
- Post-inspection cleaning: an exact value repeated ≥ 20 times in a UTC day is treated as frozen.
  Days with > 5% frozen values are dropped. The result is shown with and without the cleaning.
- Costs use notebook 07's functions and fees. Funding is not included.

## 4. Benchmarks
Not applicable (no strategy).

## 5. Results
See the Findings table and the key takeaways of `notebooks/08_liquidity_by_hour.ipynb`.

## 6. Bias checklist
- [x] No look-ahead: the BTC conversion uses only completed 1-minute candles, and the stress label
  uses only prior days.
- [x] Survivorship: not applicable.
- [x] Costs realistic, with the same fees as notebook 07. Impact is a lower bound.
- [x] Variants: the cleaning thresholds and exclusion windows have sensitivity tables in the notebook.

## 7. Conclusion
`bookDepth` is usable for the perpetual after cleaning. Depth has only a mild hour-of-day shape
(thinnest around 21 UTC). For BTC and ETH, fees dominate the round-trip cost at every hour up to
100k. Next: funding costs (M8) and a longer collector run that covers spot over several weeks.
