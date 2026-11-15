"""Metrics that describe how far, and for how long, a stablecoin leaves its $1 peg.

Input series are indexed by candle **open time (UTC)** on a regular grid (default 1 minute).
Durations are measured in clock time (differences between timestamps), never by counting rows,
so a missing candle can never shorten or stretch a duration by accident.

Thresholds are fractions: ``u = 0.005`` means 0.5% (i.e. a price outside [0.995, 1.005]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PEG = 1.0


def max_deviation(price):
    """Largest drop below the peg: ``min(price) - 1`` and the timestamp where it happens.

    Pass ``close`` for the main measure or ``low`` for the "worst traded print".
    Returns ``(deviation, timestamp)``; deviation is negative when the price went below $1.
    """
    price = price.dropna()
    ts = price.idxmin()
    return price.loc[ts] - PEG, ts


def max_upside_deviation(price):
    """Largest move above the peg: ``max(price) - 1`` and its timestamp (for USDT-like moves)."""
    price = price.dropna()
    ts = price.idxmax()
    return price.loc[ts] - PEG, ts


def abs_deviation(close):
    """``|close - 1|``, rounded to 12 decimals.

    Without rounding, floating point makes ``1 - 0.995 = 0.0050000000000000044``, so a price
    exactly on the band edge would be counted as outside it.
    """
    return (close - PEG).abs().round(12)


def is_off_peg(close, u):
    """Boolean Series: True where ``|close - 1| > u``."""
    return abs_deviation(close) > u


def minutes_off_peg(close, u, step=pd.Timedelta(minutes=1)):
    """Clock minutes spent off-peg: each off-peg candle counts for its own duration (``step``).

    Missing candles count as zero (they were not observed); report coverage separately.
    """
    n_off = int(is_off_peg(close.dropna(), u).sum())
    return n_off * step / pd.Timedelta(minutes=1)


def off_peg_span(close, u, step=pd.Timedelta(minutes=1)):
    """Clock time from the first off-peg candle's open to the last off-peg candle's close.

    Returns ``(first, last_end, duration)``; ``(NaT, NaT, Timedelta(0))`` if never off-peg.
    """
    off = is_off_peg(close.dropna(), u)
    times = off.index[off]
    if len(times) == 0:
        return pd.NaT, pd.NaT, pd.Timedelta(0)
    return times[0], times[-1] + step, times[-1] + step - times[0]


def time_to_recovery(close, start, u, hold, traded=None, min_traded=30,
                     step=pd.Timedelta(minutes=1)):
    """Time from ``start`` (usually the trough) until the price is back in the band for good.

    Recovery time = the open time ``t`` of the **first** candle at or after ``start`` such that:

    1. the candle at ``t`` is a traded candle with ``|close - 1| < u``;
    2. every traded candle in the window ``[t, t + hold)`` has ``|close - 1| < u``;
    3. that window holds at least ``min_traded`` traded candles (enough evidence);
    4. the data actually covers the whole window (``t + hold <= last candle open + step``).

    Edge case: a close exactly at ``1 ± u`` is neither "off-peg" (``> u``) nor "in band" (``< u``),
    so it can break a recovery window. This follows the definitions fixed before the analysis.

    ``traded`` is a boolean Series (True = candle with real trades, volume > 0). If None,
    every candle counts as traded (this is the "exchange-filled candles included" sensitivity).
    Returns ``(recovery_time, duration)`` or ``(NaT, NaT)`` if it never recovers in the data.
    """
    close = close.dropna()
    if traded is None:
        traded = pd.Series(True, index=close.index)
    traded = traded.reindex(close.index).fillna(False).astype(bool)

    in_band = (abs_deviation(close) < u).to_numpy()
    is_traded = traded.to_numpy()
    times = close.index
    data_end = times[-1] + step
    # Running counts let us check each window in O(1).
    bad = np.concatenate([[0], np.cumsum(is_traded & ~in_band)])
    n_traded = np.concatenate([[0], np.cumsum(is_traded)])

    first = times.searchsorted(start)
    for i in range(first, len(times)):
        t = times[i]
        if t + hold > data_end:
            break
        if not (is_traded[i] and in_band[i]):
            continue
        j = times.searchsorted(t + hold)          # window is rows i .. j-1, i.e. [t, t + hold)
        if bad[j] - bad[i] == 0 and n_traded[j] - n_traded[i] >= min_traded:
            return t, t - start
    return pd.NaT, pd.NaT


def share_in_band(close, band):
    """Fraction of candles with ``|close - 1| <= band`` (e.g. band = 0.001 for +/-0.1%)."""
    close = close.dropna()
    return float((abs_deviation(close) <= band).mean())


def classify_wick(df, ts, window=pd.Timedelta(minutes=15)):
    """Classify the extreme print of the candle at ``ts`` as an isolated wick or a level change.

    The extreme is whichever of ``low``/``high`` is farther from $1. We compare two levels:
    ``before`` = median close over the ``window`` before the candle, and ``after`` = median close
    over the ``window`` after it. It is a **level change** if ``after`` is closer to the extreme
    than to ``before`` (the price moved there and stayed); otherwise an **isolated wick**.

    Using a median of the minutes before (not just the previous close) keeps a cluster of
    consecutive spikes from being mistaken for a new level (changed after inspecting the data).

    Warning: the label looks ``window`` into the future, so it is a hindsight description only.
    Using it as a real-time signal would be look-ahead bias.
    Returns a dict with the extreme, the body (|close - open|), the wick size, both levels and the label.
    """
    row = df.loc[ts]
    low_dist, high_dist = PEG - row["low"], row["high"] - PEG
    extreme = row["low"] if low_dist >= high_dist else row["high"]
    body_edge = min(row["open"], row["close"]) if extreme == row["low"] else max(row["open"], row["close"])
    one_min = pd.Timedelta(minutes=1)
    before_closes = df["close"].loc[ts - window: ts - one_min]
    after_closes = df["close"].loc[ts + one_min: ts + window]
    before = before_closes.median() if len(before_closes) else row["open"]
    after = after_closes.median() if len(after_closes) else row["close"]
    label = "level change" if abs(after - extreme) < abs(after - before) else "isolated wick"
    return {"extreme": extreme, "body": abs(row["close"] - row["open"]),
            "wick": abs(extreme - body_edge), "median_close_before": before,
            "median_close_after": after, "volume": row["volume"], "label": label}


def rolling_median_close(close, window=pd.Timedelta(minutes=15)):
    """Trailing rolling median of ``close`` over a clock-time window (default 15 minutes).

    The value at time t uses only candles in (t - window, t]: never the future (no look-ahead).
    A median ignores spikes that last less than half the window but follows a move that stays.
    Because it only looks back, it lags: turning points show up to ~half a window later.
    """
    return close.rolling(window).median()


def implied_cross(base_usd, quote_usd):
    """Implied base/quote price from two USD prices, e.g. USDC/USDT = (USDC/USD) / (USDT/USD).

    Only timestamps present in both series are used (inner join; nothing is filled).
    """
    both = pd.concat({"base": base_usd, "quote": quote_usd}, axis=1, join="inner").dropna()
    return both["base"] / both["quote"]
