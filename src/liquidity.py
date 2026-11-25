"""Liquidity by hour (notebook 08): collector runs, Binance ``bookDepth`` files, candles.

Conventions:
- Every time is UTC. Mexico City time is UTC-6 all year (no daylight saving since 2022), so
  ``hour_cdmx = (hour_utc - 6) mod 24``; it is a label only, never used to group.
- Collector snapshots are placed in time by ``local_receive_utc`` (the first moment we knew the
  book), as in notebook 07.
- Weekdays follow pandas: Monday = 0 ... Sunday = 6, on the **UTC** calendar day.

``bookDepth`` (see ``src/data/binance_vision.py``): one row per (timestamp, percentage). This
module never assumes its meaning silently: ``book_depth_wide`` only reshapes, and the checks
(cumulative, sign, units, time zone) are separate functions whose results the notebook shows.
After the checks, ``percentage < 0`` is read as bids and ``> 0`` as asks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.order_book import DEDUP_KEY
from src.execution_costs import market_order_legs, round_trips
from src.order_book import BANDS, snapshot_metrics

CDMX_OFFSET_H = -6
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TAKER_ONLY = {"taker -> taker": ("taker", "taker")}
LONG_ONLY = {"long": ("buy", "sell")}     # taker -> taker long == short (checked in notebook 07)


# --- time labels -----------------------------------------------------------------------------

def cdmx_hour(hour_utc):
    """Mexico City hour (UTC-6, no daylight saving) of a UTC hour (int or array)."""
    return (np.asarray(hour_utc) + CDMX_OFFSET_H) % 24


def add_time_labels(df, ts_col):
    """Copy of ``df`` with ``hour_utc``, ``hour_cdmx``, ``date_utc``, ``weekday`` (0 = Mon)
    and ``weekday_name`` taken from the UTC timestamp column ``ts_col``."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    out["hour_utc"] = ts.dt.hour
    out["hour_cdmx"] = cdmx_hour(out["hour_utc"])
    out["date_utc"] = ts.dt.floor("D")
    out["weekday"] = ts.dt.day_of_week
    out["weekday_name"] = out["weekday"].map(dict(enumerate(WEEKDAYS)))
    return out


def hour_label(hour_utc):
    """'14 UTC (08 CDMX)'."""
    return f"{int(hour_utc):02d} UTC ({int(cdmx_hour(hour_utc)):02d} CDMX)"


# --- collector run: per-snapshot metrics and round trips, one hour file at a time -------------

def process_collector_run(run_dir, markets, sizes_by_market, fees, bands=BANDS):
    """Spread/depth metrics and taker -> taker round trips of every snapshot of a collector run.

    Reads one hour file and one market at a time (a 48 h spot market is ~29M rows, too much
    for memory at once). A snapshot never spans two hour files: files are split by the hour of
    ``local_request_utc``. Returns ``(metrics, rt)``: ``metrics`` = ``snapshot_metrics`` plus
    ``local_receive_utc``; ``rt`` = ``round_trips`` restricted to taker -> taker, long (one row
    per snapshot and size), with ``fees`` as in notebook 07.
    """
    files = sorted(Path(run_dir).glob("*.parquet"))
    if any(Path(run_dir).glob("parts/*/*.parquet")):
        raise ValueError(f"{run_dir} has unmerged part files: run the collector's final merge")
    metrics, rts = [], []
    for f in files:
        for market in markets:
            levels = pd.read_parquet(f, filters=[("market", "==", market)])
            if levels.empty:
                continue
            levels = levels.drop_duplicates(DEDUP_KEY, keep="last")
            m = snapshot_metrics(levels, bands=bands)
            receive = levels.groupby("local_request_utc")["local_receive_utc"].first()
            m["local_receive_utc"] = m["local_request_utc"].map(receive)
            metrics.append(m)
            legs = market_order_legs(levels, sizes_by_market[market])
            rts.append(round_trips(legs, fees, combos=TAKER_ONLY, directions=LONG_ONLY))
    return pd.concat(metrics, ignore_index=True), pd.concat(rts, ignore_index=True)


def add_total_depth(metrics, bands=BANDS):
    """Add ``depth_<band>_total_usdt`` (bid + ask) and ``complete_<band>_both``."""
    out = metrics.copy()
    for tag in bands:
        out[f"depth_{tag}_total_usdt"] = out[f"depth_{tag}_bid_usdt"] + out[f"depth_{tag}_ask_usdt"]
        out[f"complete_{tag}_both"] = out[f"complete_{tag}_bid"] & out[f"complete_{tag}_ask"]
    return out


def quantile(q):
    """Named aggregation helper: ``quantile(0.1)`` -> function computing the 10th percentile."""
    def f(x):
        return x.quantile(q)
    f.__name__ = f"p{int(round(q * 100))}"
    return f


# --- bookDepth --------------------------------------------------------------------------------

def book_depth_wide(raw):
    """One row per ``timestamp_utc``; columns ``(field, percentage)`` for depth and notional."""
    return raw.pivot_table(index="timestamp_utc", columns="percentage",
                           values=["depth", "notional"], aggfunc="first").sort_index()


def check_cumulative(wide, field="notional"):
    """Share of timestamps where ``field`` grows (weakly) with |percentage| on each side.

    If the file is cumulative from the mid, the value at ±5% is the largest and every step
    outward adds to it: the share is 100%. If each row were the depth of its own band only
    (e.g. 1%-2%), the values would not need to be ordered. Returns a DataFrame per side.
    """
    w = wide[field]
    rows = {}
    for side, cols in (("bid (negative %)", sorted([c for c in w.columns if c < 0], reverse=True)),
                       ("ask (positive %)", sorted([c for c in w.columns if c > 0]))):
        values = w[cols].to_numpy()             # columns from nearest the mid to furthest
        steps = np.diff(values, axis=1)
        rows[side] = {"percentages (mid outward)": cols,
                      "timestamps": len(w),
                      "monotonic (% of timestamps)": 100 * (steps >= 0).all(axis=1).mean(),
                      "smallest step": float(np.nanmin(steps))}
    return pd.DataFrame(rows).T


def implied_price(wide):
    """``notional / depth`` per (timestamp, percentage): the average price of the depth if
    ``notional`` is in USDT and ``depth`` in BTC."""
    return wide["notional"] / wide["depth"]


def book_depth_side_totals(wide, pcts=(0.2, 1.0), field="notional"):
    """Per timestamp: ``<field>_<p>_bid`` (at -p), ``_ask`` (at +p) and ``_total`` (sum).

    A percentage missing from the file gives NaN, never a filled value: BTCUSDT files before
    2026-01-15 have no ±0.2% rows (only ±1 ... ±5), checked in notebook 08.
    """
    w = wide[field]
    out = pd.DataFrame(index=w.index)
    missing = pd.Series(np.nan, index=w.index)
    for p in pcts:
        tag = f"{p:g}pct"
        out[f"{tag}_bid"] = w[-p] if -p in w.columns else missing
        out[f"{tag}_ask"] = w[p] if p in w.columns else missing
        out[f"{tag}_total"] = out[f"{tag}_bid"] + out[f"{tag}_ask"]
    return out


def frozen_values(sides, min_count=20, columns=None):
    """True where a value is repeated **exactly** at least ``min_count`` times in its UTC day.

    A cumulative USDT depth with many decimals does not take exactly the same value twice in a
    real book (0 repeats on the three days validated in notebook 08), so a value seen 20 times
    (~10 minutes of rows) in one day is the file repeating an old number. Counting per day, not
    per consecutive run, also catches a value that alternates with another one or comes back
    between other rows. NaN never counts. ``columns``: default every ``*_bid`` / ``*_ask``.
    """
    columns = columns or [c for c in sides.columns if c.endswith(("_bid", "_ask"))]
    day = sides.index.floor("D")
    return pd.DataFrame({c: (sides.groupby([day, sides[c]])[c].transform("size") >= min_count)
                         .reindex(sides.index, fill_value=False) & sides[c].notna()
                         for c in columns}, index=sides.index)


def clean_book_depth(sides, min_count=20, max_day_share=0.05, pcts=(0.2, 1.0)):
    """Remove frozen bookDepth values (``frozen_values``) and the days they make unusable.

    Returns ``(clean, frozen, day_share, bad_days)``: ``clean`` has frozen sides set to NaN and
    totals recomputed (NaN if either side is NaN); ``day_share`` is the share of frozen values
    per UTC day and column (among rows with a value); ``bad_days`` lists the days where any
    column has more than ``max_day_share`` frozen. On those days one side can be frozen while
    the other is distorted too (seen in notebook 08), so the caller should drop the whole day
    from patterns. Nothing is filled in.
    """
    frozen = frozen_values(sides, min_count)
    clean = sides.copy()
    for col in frozen.columns:
        clean.loc[frozen[col], col] = np.nan
    for p in pcts:
        tag = f"{p:g}pct"
        clean[f"{tag}_total"] = clean[f"{tag}_bid"] + clean[f"{tag}_ask"]
    day = sides.index.floor("D")
    has_value = sides[frozen.columns].notna()
    day_share = frozen.groupby(day).sum() / has_value.groupby(day).sum().replace(0, np.nan)
    bad_days = day_share.index[(day_share > max_day_share).any(axis=1)]
    return clean, frozen, day_share, bad_days


def resample_hourly(series_frame, how="median"):
    """Hourly aggregate (UTC hour start) of a time-indexed frame; ``n`` = rows in the hour."""
    g = series_frame.resample("1h")
    out = getattr(g, how)()
    out["n"] = g.size()
    return out[out["n"] > 0]


def match_nearest(left, right, tolerance="30s", left_on="timestamp_utc",
                  right_on="local_receive_utc", suffixes=("_bd", "_col")):
    """For each row of ``left``, the nearest row of ``right`` in time, within ``tolerance``.

    Both inputs need their time columns sorted; rows without a match within the tolerance are
    dropped. ``gap_s`` = right time - left time, in seconds.
    """
    l = left.sort_values(left_on).copy()
    r = right.sort_values(right_on).copy()
    # same resolution on both keys (bookDepth parses to us, the collector stores ms)
    l[left_on] = l[left_on].astype("datetime64[ms, UTC]")
    r[right_on] = r[right_on].astype("datetime64[ms, UTC]")
    r["_right_time"] = r[right_on]
    out = pd.merge_asof(l, r, left_on=left_on, right_on=right_on, direction="nearest",
                        tolerance=pd.Timedelta(tolerance), suffixes=suffixes)
    out = out.dropna(subset=["_right_time"])
    out["gap_s"] = (out["_right_time"] - out[left_on]).dt.total_seconds()
    return out.drop(columns="_right_time")


def lagged_correlation(x, y, lags_h=range(-8, 9)):
    """Pearson correlation of ``x`` shifted by ``lag`` hours against ``y`` (both hourly Series
    indexed by UTC hour). ``lag = +k`` compares x at t - k with y at t, i.e. x's clock would
    be **k hours ahead** of y's if this lag fits best. Returns a DataFrame lag -> corr, n."""
    rows = []
    for lag in lags_h:
        shifted = x.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=lag)
        both = pd.concat([shifted.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
        rows.append({"lag_h": lag, "corr": both["x"].corr(both["y"]) if len(both) > 2 else np.nan,
                     "n": len(both)})
    return pd.DataFrame(rows).set_index("lag_h")


# --- patterns over 12 months -------------------------------------------------------------------

def week_heatmap(hourly, column, stat="median"):
    """Weekday (rows, Mon..Sun) x UTC hour (columns 0..23) table of ``stat`` of ``column``.

    ``hourly`` needs ``weekday`` and ``hour_utc`` (``add_time_labels``). ``stat``: 'median' or
    a quantile function such as ``quantile(0.1)``.
    """
    g = hourly.groupby(["weekday", "hour_utc"])[column]
    table = (g.median() if stat == "median" else g.agg(stat)).unstack("hour_utc")
    table.index = [WEEKDAYS[i] for i in table.index]
    return table


def in_window(hourly, weekdays, hours):
    """Boolean mask: UTC weekday in ``weekdays`` and UTC hour in ``hours``."""
    return hourly["weekday"].isin(weekdays) & hourly["hour_utc"].isin(hours)


BUSINESS = dict(weekdays=[0, 1, 2, 3, 4], hours=list(range(13, 21)))      # Mon-Fri 13:00-20:59
WEEKEND_NIGHT = dict(weekdays=[5, 6], hours=[21, 22, 23, 0, 1, 2])        # Sat-Sun 21:00-02:59


def window_month(dates, start_day=24):
    """Label of the "window month" of each UTC date: from day ``start_day`` of one calendar
    month to day ``start_day - 1`` of the next, e.g. '2025-11-24 to 2025-12-23'. A 12-month
    window that starts on the 24th splits into exactly 12 such blocks."""
    d = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None).dt.normalize()
    first = d.dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=start_day - 1)
    first = first.where(d >= first, first - pd.DateOffset(months=1))
    last = first + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    return (first.dt.strftime("%Y-%m-%d") + " to " + last.dt.strftime("%Y-%m-%d")).to_numpy()


def ratio_by_month(hourly, column, num=BUSINESS, den=WEEKEND_NIGHT, month_col=None):
    """Per month: median of ``column`` in the ``num`` window, in the ``den`` window, their ratio
    and the hours behind each. Months are calendar months unless ``month_col`` names a column
    with another label (e.g. ``window_month``)."""
    h = hourly.assign(month=hourly[month_col] if month_col else
                      hourly["date_utc"].dt.strftime("%Y-%m"))
    a = h[in_window(h, **num)].groupby("month")[column].agg(["median", "count"])
    b = h[in_window(h, **den)].groupby("month")[column].agg(["median", "count"])
    out = pd.DataFrame({"numerator median": a["median"], "numerator hours": a["count"],
                        "denominator median": b["median"], "denominator hours": b["count"]})
    out["ratio"] = out["numerator median"] / out["denominator median"]
    return out


def cell_percentile(table, row, col):
    """Percentile (0-100) of one cell among all the cells of ``table`` (share of cells <= it)."""
    values = table.to_numpy().ravel()
    values = values[~np.isnan(values)]
    return 100 * (values <= table.loc[row, col]).mean()


# --- stress label ------------------------------------------------------------------------------

def daily_from_minutes(candles):
    """Daily OHLCV (UTC days) from 1-minute candles; ``minutes`` = candles in the day."""
    g = candles.resample("1D")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "quote_volume": g["quote_volume"].sum(),
                        "minutes": g.size()})
    return out[out["minutes"] > 0]


def stress_table(daily, days, lookback=90, threshold_pct=90):
    """Return and range of each day in ``days`` and their percentile vs the ``lookback`` days
    before it (the day itself excluded). Return = close / previous close - 1; range =
    (high - low) / close. The return percentile uses |return| (a big fall is stress too).
    ``stress`` is True if either percentile is above ``threshold_pct``."""
    d = daily.copy()
    d["return"] = d["close"].pct_change()
    d["range"] = (d["high"] - d["low"]) / d["close"]
    rows = []
    for day in days:
        day = pd.Timestamp(day, tz="UTC") if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day)
        prior = d[(d.index < day) & (d.index >= day - pd.Timedelta(days=lookback))]
        r, rg = d.loc[day, "return"], d.loc[day, "range"]
        rows.append({"day": day.date(), "close": d.loc[day, "close"], "high": d.loc[day, "high"],
                     "low": d.loc[day, "low"], "return (%)": 100 * r, "range (%)": 100 * rg,
                     "|return| percentile": 100 * (prior["return"].abs() < abs(r)).mean(),
                     "range percentile": 100 * (prior["range"] < rg).mean(),
                     "prior p90 |return| (%)": 100 * prior["return"].abs().quantile(0.9),
                     "prior p90 range (%)": 100 * prior["range"].quantile(0.9),
                     "prior days": len(prior)})
    out = pd.DataFrame(rows).set_index("day")
    out["stress"] = (out["|return| percentile"] > threshold_pct) | \
        (out["range percentile"] > threshold_pct)
    return out
