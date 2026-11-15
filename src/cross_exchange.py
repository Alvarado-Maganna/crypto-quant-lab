"""Compare the same pair (e.g. BTC/USDT) across exchanges, minute by minute.

Glossary:
- **bps** (basis points): 1 bps = 0.01% = 1/10,000. A 20 bps gap on a 100,000 USDT price is 200 USDT.
- **Difference vs a reference**: ``(P_ex / P_ref - 1) * 10,000``, in bps. Positive = ``ex`` is
  more expensive than the reference at that minute.
- **Run**: consecutive minutes (no missing minute in between) where |difference| > threshold.

Caveat repeated in the notebook: each ``close`` is the last trade of that minute on that
exchange, and the last trades happen at different seconds. So two closes are **not**
simultaneous prices and part of every measured gap is a timing artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 10_000


def bps_diff(price, reference):
    """Relative difference of ``price`` vs ``reference``, in bps (aligned on the index)."""
    return (price / reference - 1) * BPS


def common_minutes(frames, require_volume=True):
    """Index of minutes present in **every** frame (and, optionally, with volume > 0 in all).

    ``frames`` is ``{exchange: ohlcv_df}``. Nothing is filled: a minute missing anywhere is out.
    """
    index = None
    for df in frames.values():
        keep = df.index[df["volume"] > 0] if require_volume else df.index
        index = keep if index is None else index.intersection(keep)
    return index.sort_values()


def pairwise_diffs(closes, pairs):
    """Return a DataFrame with one column per pair ``"a vs b"`` = ``bps_diff(a, b)``."""
    return pd.DataFrame({f"{a} vs {b}": bps_diff(closes[a], closes[b]) for a, b in pairs})


def longest_run(above, step="1min"):
    """Length (in minutes) of the longest run of consecutive True values in ``above``.

    ``above`` is a boolean Series indexed by time. Two True values only belong to the same run
    if their timestamps are exactly one ``step`` apart: a missing minute breaks the run, so we
    never assume the gap stayed open while we could not see it.
    """
    if not above.any():
        return 0
    times = above.index[above.to_numpy()]
    new_run = pd.Series(times).diff() != pd.Timedelta(step)
    return int(new_run.cumsum().value_counts().max())


def summarize_diff(diff, threshold_bps=20.0):
    """Summary of one difference series (bps): the numbers in the notebook's per-pair table."""
    diff = diff.dropna()
    if diff.empty:
        raise ValueError("No common minutes to summarize")
    absolute = diff.abs()
    above = absolute > threshold_bps
    return {
        "minutes": len(diff),
        "median_bps": diff.median(),
        "p95_abs_bps": absolute.quantile(0.95),
        "max_abs_bps": absolute.max(),
        "max_at_utc": absolute.idxmax(),
        "pct_above": 100 * above.mean(),
        "minutes_above": int(above.sum()),
        "longest_run_min": longest_run(above),
    }


def coverage_report(df, start_utc, end_utc, info=None, step="1min"):
    """Coverage and quality of one exchange in one window ``[start, end)``.

    Counts expected minutes, present minutes, gaps (missing minutes), exchange-filled candles
    (volume == 0) and whether those repeat the previous close, plus total volume (base units).
    """
    from src.data.candles import mark_exchange_filled

    expected = pd.date_range(start_utc, end_utc, freq=step, inclusive="left", tz="UTC")
    marked = mark_exchange_filled(df)
    return {
        "expected": len(expected),
        "present": len(df),
        "gaps": len(expected.difference(df.index)),
        "filled_vol0": int(marked["filled_by_exchange"].sum()),
        "filled_flat_prev_close": int((marked["filled_by_exchange"]
                                       & marked["flat_at_prev_close"]).sum()),
        "duplicates_dropped": (info or {}).get("duplicates_dropped", np.nan),
        "volume_base": df["volume"].sum(),
    }


def runs_above(diff, threshold_bps=20.0, step="1min"):
    """List every run of consecutive minutes with |diff| > threshold.

    Returns a DataFrame with first/last minute (UTC), length in minutes, the largest |diff| in
    the run and its signed value at that minute. A missing minute ends a run.
    """
    diff = diff.dropna()
    above = diff.abs() > threshold_bps
    times = diff.index[above.to_numpy()]
    if len(times) == 0:
        return pd.DataFrame(columns=["first_utc", "last_utc", "minutes", "max_abs_bps",
                                     "signed_at_max_bps"])
    run_id = (pd.Series(times).diff() != pd.Timedelta(step)).cumsum().to_numpy()
    rows = []
    for _, block in pd.Series(times).groupby(run_id):
        values = diff.loc[block.to_numpy()]
        at_max = values.abs().idxmax()
        rows.append({"first_utc": block.iloc[0], "last_utc": block.iloc[-1],
                     "minutes": len(block), "max_abs_bps": abs(values[at_max]),
                     "signed_at_max_bps": values[at_max]})
    return pd.DataFrame(rows)
