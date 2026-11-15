"""Generic data-quality checks for intraday candles (any exchange, any step).

Input: a DataFrame indexed by candle open time (UTC) with columns open/high/low/close/volume.
Nothing here fills gaps: problems are either raised or reported.
"""

from __future__ import annotations

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def to_utc(ts):
    """Parse ``ts`` as a UTC Timestamp. Naive values are read as UTC; aware ones are converted."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def drop_exact_duplicates(df):
    """Sort by time and drop repeated candles, but only if the repeats are identical.

    If one timestamp appears with **different** values we cannot know which one is right,
    so we raise instead of guessing. Returns ``(clean_df, n_dropped)``.
    """
    df = df.sort_index(kind="stable")
    repeated = df[df.index.duplicated(keep=False)]
    if len(repeated):
        n_versions = repeated.reset_index().drop_duplicates().groupby(df.index.name or "index").size()
        conflicting = n_versions[n_versions > 1]
        if len(conflicting):
            raise ValueError(f"Conflicting candles for timestamps: {list(conflicting.index)}")
    clean = df[~df.index.duplicated(keep="first")]
    return clean, len(df) - len(clean)


def mark_exchange_filled(df):
    """Add ``filled_by_exchange``: True for candles with no trades (``volume == 0``).

    Such candles are synthetic: the exchange copies the previous close into O/H/L/C.
    They are kept (a quiet minute is real information) but must not be read as trades.
    Also adds ``flat_at_prev_close`` so the claim "they repeat the previous close" is checked,
    not assumed.
    """
    out = df.copy()
    out["filled_by_exchange"] = out["volume"] == 0
    flat = (out["open"] == out["high"]) & (out["high"] == out["low"]) & (out["low"] == out["close"])
    out["flat_at_prev_close"] = flat & (out["open"] == out["close"].shift())
    return out


def validate_candles(df, start_utc, end_utc, step="1min"):
    """Assert the index is unique, sorted, on the step grid; report missing candles.

    Raises ``ValueError`` on duplicates, wrong order, off-grid timestamps or prices <= 0.
    Missing candles are reported (not raised): the caller decides what they mean.
    """
    start = to_utc(start_utc)
    end = to_utc(end_utc)
    if not df.index.is_unique:
        raise ValueError("Duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted")
    if not (df.index == df.index.floor(step)).all():
        raise ValueError(f"Some timestamps are not on the {step} grid")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Found prices <= 0")
    expected = pd.date_range(start, end, freq=step)
    inside = df.index[(df.index >= start) & (df.index <= end)]
    missing = expected.difference(inside)
    return {
        "first_utc": inside.min() if len(inside) else pd.NaT,
        "last_utc": inside.max() if len(inside) else pd.NaT,
        "candles": len(inside),
        "expected": len(expected),
        "missing": len(missing),
        "missing_pct": round(100 * len(missing) / len(expected), 2),
        "outside_window": len(df) - len(inside),
        "missing_index": missing,
    }


def missing_runs(missing_index, step="1min"):
    """Group missing timestamps into consecutive runs: DataFrame with first, last, minutes."""
    s = pd.Series(missing_index)
    if s.empty:
        return pd.DataFrame(columns=["first", "last", "minutes"])
    run_id = (s.diff() != pd.Timedelta(step)).cumsum()
    runs = s.groupby(run_id).agg(["first", "last", "count"])
    return runs.rename(columns={"count": "minutes"}).reset_index(drop=True)
