"""Market-regime helpers for notebook 10 (trend, volatility terciles, drawdown, episodes).

Every label here is **past-only**: the label on day t uses only values on days <= t, so it is
known at the close of day t. Any return attributed to a regime must therefore be the **next**
day's return (close t -> close t+1); ``next_day_stats`` does that shift.

Inputs are pandas Series on a regular daily grid, indexed by candle open time (UTC). Labels are
strings, with NaN where a rule has no label yet (warm-up).

The daily returns and the past-only sigma are not rebuilt here: they come from
``src/volatility.py`` (``step_returns``, ``past_only_sigma``, ``annualize``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ABOVE, BELOW = "above MA", "below MA"
LOW, MID, HIGH = "low", "mid", "high"
NEAR_ATH, CORRECTION, DEEP_BEAR = "near-ATH", "correction", "deep bear"


# --- 1. Trend: close vs a simple moving average ------------------------------------------------

def simple_moving_average(close, window=200):
    """Mean of the last ``window`` closes **including** day t. NaN until ``window`` closes exist."""
    return close.rolling(window, min_periods=window).mean()


def ma_trend_label(close, window=200):
    """``above MA`` if close_t > MA_t, else ``below MA``; NaN during the warm-up."""
    ma = simple_moving_average(close, window)
    label = pd.Series(np.where(close > ma, ABOVE, BELOW), index=close.index, dtype=object)
    return label.where(ma.notna())


def ma_trend_band_label(close, window=200, band=0.02):
    """Trend label with a +-``band`` hysteresis ("switch only on a clear cross").

    The label changes to ``above MA`` only when close > MA * (1 + band), and to ``below MA``
    only when close < MA * (1 - band). Inside the band it keeps yesterday's label.
    Before the first clear cross after the warm-up there is no label yet (NaN).
    Path-dependent, but each step only looks at day t and the label of day t-1.
    """
    ma = simple_moving_average(close, window)
    out = []
    state = None
    for c, m in zip(close.to_numpy(), ma.to_numpy()):
        if np.isnan(m):
            out.append(None)
            continue
        if c > m * (1 + band):
            state = ABOVE
        elif c < m * (1 - band):
            state = BELOW
        out.append(state)
    return pd.Series(out, index=close.index, dtype=object).where(lambda s: s.notna())


# --- 2. Volatility terciles from past-only thresholds ------------------------------------------

def expanding_tercile_thresholds(x, min_periods=365):
    """Lower / upper tercile of all non-NaN values of ``x`` on days <= t.

    Needs ``min_periods`` non-NaN values; before that both thresholds are NaN.
    Returns a DataFrame with columns ``q1`` (1/3 quantile) and ``q2`` (2/3 quantile).
    """
    e = x.expanding(min_periods=min_periods)
    return pd.DataFrame({"q1": e.quantile(1 / 3), "q2": e.quantile(2 / 3)})


def rolling_tercile_thresholds(x, window=730):
    """Lower / upper tercile of the last ``window`` values of ``x`` (day t included).

    The window must be full of non-NaN values (``min_periods=window``).
    """
    r = x.rolling(window, min_periods=window)
    return pd.DataFrame({"q1": r.quantile(1 / 3), "q2": r.quantile(2 / 3)})


def tercile_label(x, thresholds):
    """``low`` if x < q1, ``high`` if x > q2, else ``mid``; NaN where x or a threshold is NaN."""
    q1, q2 = thresholds["q1"], thresholds["q2"]
    label = pd.Series(np.where(x < q1, LOW, np.where(x > q2, HIGH, MID)), index=x.index, dtype=object)
    return label.where(x.notna() & q1.notna() & q2.notna())


# --- 3. Drawdown from the running maximum ------------------------------------------------------

def drawdown_from_running_max(close):
    """close_t / max(close_s for s <= t) - 1. Zero on a new high, negative otherwise."""
    return close / close.cummax() - 1


def drawdown_label(dd, correction=-0.10, deep=-0.30):
    """``near-ATH`` for dd >= -10%, ``correction`` for -30% <= dd < -10%, ``deep bear`` below -30%."""
    label = pd.Series(np.where(dd >= correction, NEAR_ATH,
                               np.where(dd >= deep, CORRECTION, DEEP_BEAR)),
                      index=dd.index, dtype=object)
    return label.where(dd.notna())


# --- 4. Episodes, flips, next-day stats --------------------------------------------------------

def episodes(labels):
    """Runs of consecutive days with the same (non-NaN) label.

    A NaN day ends a run. Each run gets ``censored_first`` = it starts on the first labelled day
    (we do not see its real start) and ``censored_last`` = it ends on the last day of the data
    (it may still be going on). Only uncensored runs have a known full duration.
    """
    s = labels.dropna()
    cols = ["label", "start", "end", "n_days", "censored_first", "censored_last"]
    if s.empty:
        return pd.DataFrame(columns=cols)
    pos = pd.Series(np.arange(len(labels)), index=labels.index).loc[s.index]
    new_run = (s != s.shift(1)) | (pos.diff() != 1)
    run_id = new_run.cumsum()
    g = pd.DataFrame({"label": s, "date": s.index, "run": run_id}).groupby("run")
    out = g.agg(label=("label", "first"), start=("date", "first"), end=("date", "last"),
                n_days=("label", "size")).reset_index(drop=True)
    out["censored_first"] = out["start"] == s.index[0]
    out["censored_last"] = out["end"] == labels.index[-1]
    return out[cols]


def episode_summary(eps, short_days=5):
    """Per label: number of episodes, how many are censored and short, and the median / p90 /
    max duration of **complete** (uncensored) episodes only."""
    rows = []
    for label, g in eps.groupby("label"):
        complete = g.loc[~g["censored_first"] & ~g["censored_last"], "n_days"]
        rows.append({
            "label": label,
            "n episodes": len(g),
            "n censored (first)": int(g["censored_first"].sum()),
            "n censored (last)": int(g["censored_last"].sum()),
            f"n short (< {short_days} d)": int((g["n_days"] < short_days).sum()),
            "n complete": len(complete),
            "median days (complete)": complete.median() if len(complete) else np.nan,
            "p90 days (complete)": complete.quantile(0.9) if len(complete) else np.nan,
            "max days (complete)": complete.max() if len(complete) else np.nan,
        })
    return pd.DataFrame(rows).set_index("label")


def flips(labels):
    """True on each day whose label differs from the previous day's label (both non-NaN)."""
    prev = labels.shift(1)
    return labels.notna() & prev.notna() & (labels != prev)


def flips_per_year(labels):
    """Number of label changes per calendar year (UTC) of the day the new label starts."""
    f = flips(labels)
    return f.groupby(f.index.year).sum().astype(int)


def next_day_stats(labels, returns):
    """Per label, stats of the **next-day** return (close t -> close t+1) after a day with that label.

    Also: the same stats without the label's single worst episode (lowest compounded next-day
    return over the episode), and the mechanical same-day mean (return of day t itself, which
    the label already "saw"). Returns are in basis points (1 bp = 0.01%).
    """
    nxt = returns.shift(-1)
    eps = episodes(labels)
    episode_id = pd.Series(np.nan, index=labels.index)
    for i, row in eps.iterrows():
        episode_id.loc[row["start"]:row["end"]] = i
    df = pd.DataFrame({"label": labels, "next": nxt, "same": returns, "ep": episode_id})
    df = df[df["label"].notna()]
    rows = []
    for label, g in df.groupby("label"):
        valid = g.dropna(subset=["next"])
        compounded = valid.groupby("ep")["next"].apply(lambda r: (1 + r).prod() - 1)
        worst = compounded.idxmin()
        ex = valid[valid["ep"] != worst]
        rows.append({
            "label": label,
            "mean next-day (bps)": 1e4 * valid["next"].mean(),
            "median next-day (bps)": 1e4 * valid["next"].median(),
            "daily σ next-day (%)": 100 * valid["next"].std(ddof=1),
            # naive standard error of the mean: assumes independent days, so it is too small
            # when volatility clusters. Only a rough sense of how noisy the mean is.
            "SE of mean, i.i.d. (bps)": 1e4 * valid["next"].std(ddof=1) / np.sqrt(len(valid)),
            "n days": len(valid),
            "n episodes": valid["ep"].nunique(),
            "worst episode start": eps.loc[int(worst), "start"].date(),
            "worst episode compounded (%)": 100 * compounded.loc[worst],
            "mean next-day ex-worst (bps)": 1e4 * ex["next"].mean(),
            "median next-day ex-worst (bps)": 1e4 * ex["next"].median(),
            "n days ex-worst": len(ex),
            "mechanical (same-day) mean (bps)": 1e4 * g["same"].mean(),
        })
    return pd.DataFrame(rows).set_index("label")


# --- 5. Look-ahead check -----------------------------------------------------------------------

def truncation_mismatches(label_fn, data, cutoff):
    """Recompute labels using only ``data`` up to ``cutoff`` and compare with the full-data labels.

    ``label_fn(data) -> Series of labels``. Returns the number of days <= ``cutoff`` whose label
    differs (NaN vs NaN counts as equal). A past-only rule must return 0 for every cutoff.
    """
    full = label_fn(data)
    part = label_fn(data.loc[:cutoff])
    full = full.loc[part.index]
    same = (full == part) | (full.isna() & part.isna())
    return int((~same).sum())
