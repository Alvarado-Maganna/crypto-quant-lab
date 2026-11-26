# Experiment: BTC market regimes with past-only rules (notebook 10)

- **Date started:** 2026-11-25
- **Status:** concluded (descriptive study, no strategy)
- **Branch / PR:** `feat/10-regimes`

## 1. Hypothesis
No prior hypotheses were recorded. The notebook answers questions Q1–Q9, listed in its
introduction: look-ahead checks, time in regime, durations, next-day returns by regime, overlap
of rules, MA200 flips and their assumed cost, expanding vs rolling volatility labels, the 30-day
window step, and the current state.

## 2. Data
- Binance spot BTC/USDT daily candles via ccxt (`src/data/ohlcv.py`), cached in `data/10_regimes/`.
- Open times 2017-08-17 (Binance's first day) → 2026-11-24 (last complete UTC day): 3,387 candles,
  no missing days, no duplicates. On the 3,249 days shared with notebook 09's cache the closes are
  identical.
- No in-sample / out-of-sample split: nothing is fitted or tested as a strategy.

## 3. Method
- Rules (pre-specified in the brief, before results):
  - Trend: close vs MA200 of closes ≤ t (warm-up 200 closes); sensitivity version with a ±2% band
    (switch only on a cross of more than 2%).
  - Volatility: past-only σ30 from `src/volatility.py` (√365), terciles with (a) expanding
    thresholds (min 365 σ values) and (b) rolling 730-day thresholds.
  - Drawdown from the running max: near-ATH (≥ −10%), correction (−10% to −30%), deep bear (< −30%).
- Timing: labels known at the close of day t; returns attributed to a regime are next-day returns.
  Same-day returns appear only in a column labelled "mechanical (same-day)".
- Look-ahead: truncation test at 2019-12-31, 2021-12-31, 2023-12-31 and 2025-11-24 (asserted in
  code); a full-sample tercile version is shown only as a labelled look-ahead illustration.
- Choices made by me before results (the brief did not fix them):
  - With the ±2% band, the first label after the warm-up is the first clear cross.
  - Rolling thresholds need a full 730-day window; tercile ties go to "mid".
  - "Worst episode" = the label's episode with the lowest compounded next-day return.
  - An i.i.d. standard error of the mean is added to the next-day table (it understates noise).
  - 2025-10-10 was listed with the halvings in the brief, but it is not a halving. It is drawn as an
    event marker (same as notebook 09), and only 2020-05-11 and 2024-04-20 are labelled as halvings.
- Post-inspection (cosmetic only): lighter drawdown bands and the legend moved; 2017 (no labels)
  removed from the flips table.
- Changes after the reviewer (none changes a conclusion):
  - The same-day column is explained per rule. It is mechanical for trend and drawdown, which use
    close t, but not for volatility, whose σ goes up to t−1.
  - Per-rule baselines and a warning about comparing 13 label rows at once.
  - A rough Poisson check for the window step.
  - A limitation on threshold flicker (deep-bear median 1 day vs max 1,015 days).
  - Denominators made explicit in the takeaways.
  - The notebook 09 cache cross-check is skipped if that cache is missing.
  - Edge-case guards in `episodes` and `next_day_stats`, plus 4 more tests.
- Costs: the flip cost uses an ASSUMED 20 bps spot taker round trip from notebook 07.

## 4. Benchmarks
Not applicable (no strategy). Each rule's per-label next-day means are compared with that rule's
own baseline, the mean over all its labelled days. The baselines are 10.5 bps (trend), 12.7
(expanding vol), 11.7 (rolling vol) and 13.9 (drawdown), because the rules start on different dates.

## 5. Results
See the Findings table and the key takeaways of `notebooks/10_regimes.ipynb`.

## 6. Bias checklist
- [x] No look-ahead: truncation test passes for all 5 rules at 4 cutoffs; the full-sample version
  fails it (677 days at 2021-12-31) and is labelled "do not use".
- [x] Survivorship: BTC was chosen by the brief, not by its results; it is a survivor.
- [x] Costs: only the illustrative flip cost, assumed from notebook 07. Spread and slippage beyond the
  taker fee and funding are not modelled (no strategy here).
- [x] Variants: one parameter set fixed by the brief, plus one pre-specified sensitivity version
  (±2% band) and two pre-specified volatility versions. Nothing was tuned.

## 7. Conclusion
The rules are past-only, and they describe very different things. BTC was above its MA200 on about
half of the days, but more than 30% below its highest close on about two thirds of them. The plain
MA200 label is noisy, and the ±2% band halves the flips. Next-day returns differ little between
labels compared with their noise, while same-day returns differ a lot for mechanical reasons. The
volatility labels inherit the 30-day window's steps. Next: if a regime is ever used as a filter, it
must be tested out-of-sample with full costs, against buy & hold and a random filter.
