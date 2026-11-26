"""Position-sizing helpers for notebook 11 (bootstrap paths, stop fills, a sizing simulator).

The notebook holds the entry rule fixed (random entries, no edge by design) and changes only
**how much** is bought. Everything here is about risk, not returns.

Conventions (fixed by the notebook brief):
- A day is a tuple ``(open, high, low, close) / previous close``. Paths are built by resampling
  whole tuples, never separate fields, so every simulated day keeps a real intraday shape.
- A path starts at price level 1. Day t's prices are ``level_{t-1} * tuple_t``.
- Past-only sigma inside a path comes from ``src/volatility.py`` (``past_only_sigma``,
  ``annualize``): the sigma on day t uses the 30 returns **before** day t.
- Long only, spot, no leverage: the cash spent on an entry (notional + fee) never exceeds the
  equity. Fees are charged per side on notional.
- Equity is marked at each daily close (cash + quantity * close). Daily bars cannot see what
  happened between the open and the close, so intraday equity is never observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volatility import DAYS_PER_YEAR, annualize, past_only_sigma

FEE_PER_SIDE = 0.001          # 10 bps spot taker (notebook 07)
EXIT_STOP, EXIT_TIME, EXIT_END = "stop", "time", "end of path"
_EXIT_CODES = {0: EXIT_STOP, 1: EXIT_TIME, 2: EXIT_END}


# --- 1. Sizing rule ------------------------------------------------------------------------------

def size(capital, risk, entry, stop):
    """Quantity to buy so that a fill exactly at ``stop`` loses ``risk * capital`` (before costs).

    ``size(10_000, 0.01, 100_000, 97_000)`` = 100 / 3,000 = 0.0333 BTC (3,333.33 USDT notional).
    Works on numbers or numpy arrays. The stop must be below the entry (long only).
    """
    entry, stop = np.asarray(entry, dtype=float), np.asarray(stop, dtype=float)
    if np.any(stop >= entry):
        raise ValueError("stop must be below entry for a long position")
    qty = np.asarray(capital, dtype=float) * risk / (entry - stop)
    return qty.item() if qty.ndim == 0 else qty


# --- 2. Daily tuples and resampling --------------------------------------------------------------

def day_tuples(ohlc):
    """``(open, high, low, close) / previous close`` for every day after the first.

    ``ohlc`` needs columns open/high/low/close on a regular daily grid with no missing day
    (a gap would turn two days into one tuple).
    """
    if getattr(ohlc.index, "tz", None) is None or str(ohlc.index.tz) != "UTC":
        raise ValueError("ohlc index must be timezone-aware UTC")
    ohlc = ohlc.sort_index()
    steps = ohlc.index.to_series().diff().dropna()
    if (steps != pd.Timedelta(days=1)).any() or ohlc[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("ohlc must be a complete daily grid (no missing days, no NaN)")
    prev = ohlc["close"].shift(1)
    return ohlc[["open", "high", "low", "close"]].div(prev, axis=0).iloc[1:]


def demean_tuples(tuples, mean_return=None, keep_open=False):
    """Remove the average daily close-to-close return from every tuple.

    The close ratio becomes ``c - m`` (so the new mean return is exactly 0). ``m`` defaults to the
    mean of ``close - 1`` over the tuples given. Returns ``(demeaned tuples, m)``.

    - ``keep_open=False`` (notebook 11's pre-specified rule): open, high and low are multiplied by
      the same factor ``(c - m) / c``, so the day's shape relative to its close is kept. Side effect:
      the whole drift removal lands in the gap from the previous close to the open, and
      open-to-close keeps the raw drift. A position bought at the open still earns that drift.
    - ``keep_open=True``: the open stays as it is, high and low are scaled by the same factor and
      then widened if needed so that ``low <= open, close <= high``. The drift is removed from the
      open-to-close part, where a trade entered at the open is exposed to it.
    """
    m = float((tuples["close"] - 1).mean()) if mean_return is None else float(mean_return)
    k = (tuples["close"] - m) / tuples["close"]
    out = tuples.mul(k, axis=0)
    if keep_open:
        out["open"] = tuples["open"]
        out["high"] = np.maximum(out["high"], out["open"])
        out["low"] = np.minimum(out["low"], out["open"])
    return out, m


def stationary_bootstrap_indices(n_obs, n_paths, n_days, mean_block, rng):
    """Politis-Romano stationary bootstrap: row indices of shape ``(n_paths, n_days)``.

    Each path starts at a random day. Every next day either continues the current block (the
    following day of the sample, wrapping from the last day to the first) or, with probability
    ``1 / mean_block``, jumps to a new random day. Block lengths are geometric with mean
    ``mean_block``, so runs of calm or wild days are kept together.
    """
    p = 1.0 / mean_block
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_paths)
    jump = rng.random((n_paths, n_days)) < p
    new_start = rng.integers(0, n_obs, (n_paths, n_days))
    for t in range(1, n_days):
        idx[:, t] = np.where(jump[:, t], new_start[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def iid_bootstrap_indices(n_obs, n_paths, n_days, rng):
    """Every day drawn independently (the control: any clustering is destroyed)."""
    return rng.integers(0, n_obs, (n_paths, n_days))


def path_sigma(close_ratios, window=30):
    """Annualized past-only sigma on every day of every path, shape ``(n_paths, n_days)``.

    ``close_ratios`` holds ``close_t / close_{t-1}`` per path. The tested ``past_only_sigma`` from
    notebook 09 is applied to each path (a placeholder daily calendar gives it a regular grid).
    NaN during the first ``window`` days.
    """
    close_ratios = np.asarray(close_ratios, dtype=float)
    calendar = pd.date_range("2000-01-01", periods=close_ratios.shape[1], freq="D", tz="UTC")
    returns = pd.DataFrame(close_ratios.T - 1, index=calendar)
    return annualize(past_only_sigma(returns, window), DAYS_PER_YEAR).to_numpy().T


# --- 3. Stop fills -------------------------------------------------------------------------------

def stop_fill(open_, low, stop, slippage=0.0):
    """Where a long stop is filled on one daily bar.

    - The day opens at or below the stop -> filled at the open (a gap: worse than the stop).
    - Otherwise, the low reaches the stop -> filled at ``stop * (1 - slippage)``.
    - Otherwise -> not hit (fill NaN).
    Returns ``(hit, fill)`` arrays.
    """
    open_, low, stop = (np.asarray(a, dtype=float) for a in (open_, low, stop))
    gap = open_ <= stop
    touch = ~gap & (low <= stop)
    fill = np.where(gap, open_, np.where(touch, stop * (1 - slippage), np.nan))
    return gap | touch, fill


# --- 4. Fractions of equity at entry -------------------------------------------------------------

def fraction_all_in(sigma, entry, equity):
    """Notional = 100% of equity (capped later so notional + fee fits in the cash)."""
    return np.ones_like(np.asarray(entry, dtype=float))


def fraction_fixed_risk(risk=0.01, stop_pct=0.05):
    """Fraction of equity whose loss at the stop is ``risk`` of equity: ``risk / stop_pct``."""
    def f(sigma, entry, equity):
        entry = np.asarray(entry, dtype=float)
        qty = size(equity, risk, entry, entry * (1 - stop_pct))
        return qty * entry / np.asarray(equity, dtype=float)
    return f


def fraction_vol_target(sigma_target):
    """``min(1, sigma_target / sigma_t)``: buy less when recent volatility is high."""
    def f(sigma, entry, equity):
        return np.minimum(1.0, sigma_target / np.asarray(sigma, dtype=float))
    return f


# --- 5. Simulator --------------------------------------------------------------------------------

def simulate(tuples, idx, coin, sigma, schemes, *, initial=10_000.0, fee=FEE_PER_SIDE,
             stop_pct=0.05, max_hold=20, p_entry=0.20, warmup=30, slippage=0.0,
             record_path=None, quantiles=None):
    """Run every sizing scheme on the same paths, entry dates and exits.

    ``tuples`` (n_obs, 4) are the day tuples; ``idx`` (n_paths, n_days) says which tuple each
    simulated day uses; ``coin`` (n_paths, n_days) holds uniform draws for the entry rule;
    ``sigma`` (n_paths, n_days) is the annualized past-only sigma; ``schemes`` maps a name to a
    function ``(sigma, entry price, equity) -> fraction of equity``.

    Rules on each day t, in this order:
    1. A pending entry (decided at close t-1) is filled at the open of t. The fraction uses
       sigma[t] (30 returns before t) and the equity at close t-1. Fee = ``fee * notional``.
    2. Stop at ``(1 - stop_pct) * entry``, checked with ``stop_fill`` (also on the entry day).
       If not stopped, a time exit at the close of day ``entry_day + max_hold``. Any position
       still open on the last day is sold at its close.
    3. Equity is marked at the close.
    4. If flat at the close of a day t >= warmup - 1 (and not the last day), enter with
       probability ``p_entry`` (``coin < p_entry``) at the next open. So no trade in the first
       ``warmup`` days.

    Entries and exits do not depend on the size, so every scheme trades on the same days.
    Returns ``(paths, trades, record)``: per-path statistics per scheme, one row per trade, and
    a dict (or None if nothing is recorded) with:
    - ``equity`` / ``exposure``: the daily values of ``record_path``, shape (n_schemes, n_days)
      for one path index, or (n_schemes, len(record_path), n_days) for a list of indices;
    - ``equity_quantiles``: if ``quantiles`` is given, the daily quantiles of equity across all
      paths, shape (n_schemes, len(quantiles), n_days).
    """
    tuples = np.asarray(tuples, dtype=float)
    n_paths, n_days = idx.shape
    names = list(schemes)
    n_s = len(names)

    cash = np.full((n_s, n_paths), float(initial))
    qty = np.zeros((n_s, n_paths))
    eq_entry = np.full((n_s, n_paths), np.nan)
    frac = np.zeros((n_s, n_paths))
    peak = cash.copy()
    max_dd = np.zeros((n_s, n_paths))
    min_eq = cash.copy()
    expo_sum = np.zeros((n_s, n_paths))
    n_trades = np.zeros(n_paths, dtype=np.int64)

    in_pos = np.zeros(n_paths, dtype=bool)
    pending = np.zeros(n_paths, dtype=bool)
    entry_px = np.zeros(n_paths)
    stop_px = np.zeros(n_paths)
    entry_day = np.full(n_paths, -1, dtype=np.int64)
    level = np.ones(n_paths)                    # previous close
    rec = {}
    rec_idx = None if record_path is None else np.atleast_1d(record_path)
    if rec_idx is not None:
        rec["equity"] = np.zeros((n_s, len(rec_idx), n_days))
        rec["exposure"] = np.zeros((n_s, len(rec_idx), n_days))
    if quantiles is not None:
        rec["equity_quantiles"] = np.zeros((n_s, len(quantiles), n_days))
    log = []

    for t in range(n_days):
        o, h, l, c = (level * tuples[idx[:, t], j] for j in range(4))

        # 1. entries at the open
        if pending.any():
            e = pending
            for s, name in enumerate(names):
                equity = cash[s, e]
                f = np.clip(schemes[name](sigma[e, t], o[e], equity), 0.0, None)
                notional = np.minimum(f * equity, equity / (1 + fee))
                qty[s, e] = notional / o[e]
                cash[s, e] = equity - notional * (1 + fee)
                frac[s, e] = notional / equity
                eq_entry[s, e] = equity
            in_pos[e] = True
            entry_px[e], stop_px[e], entry_day[e] = o[e], o[e] * (1 - stop_pct), t
            n_trades[e] += 1
            pending[:] = False

        # 2. exits
        hit, fill = stop_fill(o, l, stop_px, slippage)
        hit &= in_pos
        gap = hit & (o <= stop_px)               # filled at an open already below the stop
        timed = in_pos & ~hit & (t - entry_day >= max_hold)
        end = in_pos & ~hit & ~timed & (t == n_days - 1)
        ex = hit | timed | end
        if ex.any():
            px = np.where(hit, fill, c)[ex]
            cash[:, ex] += qty[:, ex] * px * (1 - fee)
            log.append({
                "path": np.flatnonzero(ex), "entry_day": entry_day[ex],
                "exit_day": np.full(ex.sum(), t), "exit_code": np.select(
                    [hit[ex], timed[ex]], [0, 1], 2), "gap_open": gap[ex],
                "entry_px": entry_px[ex], "exit_px": px,
                **{f"frac_{n}": frac[s, ex] for s, n in enumerate(names)},
                **{f"pnl_{n}": cash[s, ex] / eq_entry[s, ex] - 1 for s, n in enumerate(names)},
            })
            qty[:, ex] = 0.0
            in_pos[ex] = False

        # 3. mark at the close
        equity = cash + qty * c
        np.maximum(peak, equity, out=peak)
        np.maximum(max_dd, 1 - equity / peak, out=max_dd)
        np.minimum(min_eq, equity, out=min_eq)
        exposure = qty * c / equity
        if t >= warmup:
            expo_sum += exposure
        if rec_idx is not None:
            rec["equity"][:, :, t] = equity[:, rec_idx]
            rec["exposure"][:, :, t] = exposure[:, rec_idx]
        if quantiles is not None:
            rec["equity_quantiles"][:, :, t] = np.quantile(equity, quantiles, axis=1).T

        # 4. entry decision at the close
        if warmup - 1 <= t < n_days - 1:
            pending = ~in_pos & (coin[:, t] < p_entry)
        level = c

    if rec_idx is not None and np.ndim(record_path) == 0:
        rec["equity"], rec["exposure"] = rec["equity"][:, 0], rec["exposure"][:, 0]
    rec = rec or None

    paths = {name: pd.DataFrame({
        "final_equity": cash[s], "max_dd": max_dd[s], "min_equity": min_eq[s],
        "mean_exposure": expo_sum[s] / (n_days - warmup), "n_trades": n_trades,
    }) for s, name in enumerate(names)}
    if log:
        trades = pd.DataFrame({k: np.concatenate([d[k] for d in log]) for k in log[0]})
    else:
        trades = pd.DataFrame(columns=["path", "entry_day", "exit_day", "exit_code", "gap_open",
                                       "entry_px", "exit_px",
                                       *[f"frac_{n}" for n in names], *[f"pnl_{n}" for n in names]])
    trades["exit_type"] = trades["exit_code"].map(_EXIT_CODES)
    trades = trades.drop(columns="exit_code").sort_values(["path", "exit_day"], ignore_index=True)
    return paths, trades, rec


def simulate_buy_hold(tuples, idx, *, initial=10_000.0, fee=FEE_PER_SIDE, warmup=30):
    """Buy & hold on the same paths: buy at the open of day ``warmup`` (same first possible day as
    the active schemes), sell at the last close. Fee on both sides; notional + fee = equity.

    ``mean_exposure`` is 1 on every invested day except the last, where the equity is already net
    of the selling fee (exposure 1 / (1 - fee)); the effect on the average is about 1e-6.

    Returns per-path statistics with the same columns as ``simulate``.
    """
    qty, level, equity = _buy_hold_parts(tuples, idx, initial, fee, warmup)
    return pd.DataFrame({
        "final_equity": equity[:, -1], "max_dd": max_drawdown(equity),
        "min_equity": equity.min(axis=1),
        "mean_exposure": (qty[:, None] * level[:, warmup:] / equity[:, warmup:]).mean(axis=1),
        "n_trades": np.ones(len(idx), dtype=np.int64),
    })


def buy_hold_equity(tuples, idx, *, initial=10_000.0, fee=FEE_PER_SIDE, warmup=30):
    """Daily buy & hold equity on every path, shape (n_paths, n_days), marked at the closes
    (the last day already net of the selling fee). Same rules as ``simulate_buy_hold``."""
    return _buy_hold_parts(tuples, idx, initial, fee, warmup)[2]


def _buy_hold_parts(tuples, idx, initial, fee, warmup):
    tuples = np.asarray(tuples, dtype=float)
    o = tuples[idx, 0]
    c = tuples[idx, 3]
    level = np.cumprod(c, axis=1)                      # close levels, start level 1
    prev = np.concatenate([np.ones((len(idx), 1)), level[:, :-1]], axis=1)
    qty = initial / (1 + fee) / (prev[:, warmup] * o[:, warmup])
    equity = np.where(np.arange(idx.shape[1]) >= warmup, qty[:, None] * level, initial)
    equity[:, -1] *= (1 - fee)                        # sold at the last close
    return qty, level, equity


# --- 6. Path statistics --------------------------------------------------------------------------

def max_drawdown(equity):
    """Largest fall from a running peak, as a fraction (0.3 = -30%). Works along the last axis."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity, axis=-1)
    return (1 - equity / peak).max(axis=-1)


def max_drawdown_day(equity):
    """Index of the day on which the largest drawdown reaches its bottom (along the last axis)."""
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity, axis=-1)
    return (1 - equity / peak).argmax(axis=-1)


def required_recovery(loss):
    """Gain needed to get back to the peak after losing ``loss``: ``L / (1 - L)``."""
    loss = np.asarray(loss, dtype=float)
    return loss / (1 - loss)


def longest_losing_streak(trades, n_paths, pnl_col):
    """Longest run of consecutive losing trades (``pnl_col < 0``) on each path, in exit order.

    ``trades`` must be sorted by path and exit day (``simulate`` returns it that way). Paths with
    no trade or no loss get 0. Returns a Series indexed 0..n_paths-1.
    """
    if trades.empty:
        return pd.Series(0, index=range(n_paths))
    loss = (trades[pnl_col] < 0).to_numpy()
    path = trades["path"].to_numpy()
    new_run = ~loss | np.r_[True, path[1:] != path[:-1]]
    run_id = np.cumsum(new_run)
    runs = pd.DataFrame({"path": path, "run": run_id, "loss": loss})
    lengths = runs[runs["loss"]].groupby(["path", "run"]).size()
    return lengths.groupby(level="path").max().reindex(range(n_paths), fill_value=0)
