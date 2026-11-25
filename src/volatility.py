"""Volatility helpers for notebook 09 (returns, past-only sigma, tails, clustering, sqrt-t rule).

Definitions used everywhere (fixed by the notebook brief):
- Simple return: ``close_t / close_{t-1} - 1`` (not log).
- Volatility: standard deviation of simple returns with ``ddof=1``.
- Annualization: multiply by ``sqrt(periods per year)``; crypto trades every day, so 365 days
  (daily) and 365 * 24 hours (hourly).
- Past-only sigma at day t: std of the ``window`` returns **before** t (t itself excluded).
  ``z_t = r_t / sigma_{t-1}`` therefore uses only information known at the open of day t.

Inputs are pandas Series indexed by candle open time (UTC).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365
HOURS_PER_YEAR = 365 * 24
TRADING_DAYS_STOCKS = 252  # stock-market convention, shown only as a comparison


def step_returns(close, step):
    """Simple returns between consecutive candles that are exactly ``step`` apart.

    If a candle is missing, the return across the gap would cover two (or more) periods and look
    like an outlier, so it is set to NaN instead. The first candle has no return (NaN dropped).
    """
    close = close.sort_index()
    ret = close / close.shift(1) - 1
    gap = close.index.to_series().diff() != pd.Timedelta(step)
    ret[gap.to_numpy()] = np.nan
    return ret.iloc[1:]


def annualize(sigma, periods_per_year):
    """Scale a per-period volatility to a yearly one with the sqrt-of-time rule."""
    return sigma * math.sqrt(periods_per_year)


def rolling_vol(returns, window=30):
    """Std (ddof=1) of the last ``window`` returns **including** the current one.

    This is "realized volatility up to the end of day t": fine for a chart over time, but not for
    judging day t itself (use ``past_only_sigma`` for that). Needs ``window`` valid values.
    """
    return returns.rolling(window, min_periods=window).std(ddof=1)


def past_only_sigma(returns, window=30):
    """Std of the ``window`` returns strictly **before** each row (the current row excluded).

    The series must be on a regular grid (one row per period, NaN where missing), so that
    "shift by one row" means "the previous period". A window with any NaN gives NaN.
    """
    if returns.index.freq is None and len(returns) > 2:
        steps = returns.index.to_series().diff().dropna().unique()
        if len(steps) != 1:
            raise ValueError("returns must be on a regular grid; reindex with NaN for gaps first")
    return rolling_vol(returns, window).shift(1)


def past_only_z(returns, window=30):
    """``z_t = r_t / sigma_{t-1}``: today's return in units of the previous ``window`` days' std."""
    return returns / past_only_sigma(returns, window)


def normal_two_sided_tail(k):
    """P(|Z| > k) for a standard normal Z, via the complementary error function."""
    return math.erfc(k / math.sqrt(2))


def student_t_two_sided_tail(k, df, points=200_001):
    """P(|T| > k) for a Student-t variable with ``df`` degrees of freedom (no scipy needed).

    Needs numpy >= 2.0 (``np.trapezoid``).

    Computed as 1 - 2 * integral of the t density from 0 to k (trapezoid rule on a fine grid; the
    density is smooth and bounded, so this is accurate for any ``df``).

    Why it matters here: if returns were exactly normal but sigma is *estimated* from n = 30 past
    returns, z = r / sigma_hat follows a t distribution with n - 1 = 29 degrees of freedom, which
    has fatter tails than N(0, 1).
    """
    x = np.linspace(0.0, k, points)
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    density = np.exp(log_c - (df + 1) / 2 * np.log1p(x * x / df))
    return float(1 - 2 * np.trapezoid(density, x))


def tail_table(z, thresholds=(2, 3, 4, 5), t_df=None):
    """Days with |z| > k: observed count vs what a standard normal would give for the same n.

    With ``t_df`` set, also adds the count expected under a Student-t with that many degrees of
    freedom (normal returns, but sigma estimated from ``t_df + 1`` past returns).
    """
    z = z.dropna()
    n = len(z)
    rows = []
    for k in thresholds:
        observed = int((z.abs() > k).sum())
        expected = n * normal_two_sided_tail(k)
        row = {"threshold": f"|z| > {k}", "n_days": n, "observed": observed,
               "expected_normal": expected,
               "observed_over_expected": observed / expected if expected else np.nan}
        if t_df is not None:
            row[f"expected_t{t_df}"] = n * student_t_two_sided_tail(k, t_df)
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def acf(x, max_lag=20):
    """Sample autocorrelation at lags 1..max_lag (the standard estimator, common mean and
    denominator). NaNs are dropped first, so use it on a series without gaps.

    Returns ``(Series of acf by lag, n)``; ``+-1.96 / sqrt(n)`` is the approximate 95% band
    for a series with no autocorrelation.
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    d = x - x.mean()
    denom = (d ** 2).sum()
    values = {lag: (d[lag:] * d[:-lag]).sum() / denom for lag in range(1, max_lag + 1)}
    return pd.Series(values, name="acf").rename_axis("lag"), n


def acf_robust_band(x, max_lag=20, z_crit=1.96):
    """95% band for the autocorrelation at each lag that stays valid when volatility clusters.

    ``+-1.96 / sqrt(n)`` assumes i.i.d. data. When big moves come in streaks, the sample
    autocorrelation of the *returns* is noisier than that, so the i.i.d. band is too narrow and
    flags too many lags. This band uses the heteroskedasticity-robust variance
    ``sum(d_t^2 * d_{t-k}^2) / (sum(d_t^2))^2`` (d = x - mean), which equals 1/n for i.i.d. data
    with constant variance. Returns a Series of half-widths by lag.
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    d = x - x.mean()
    denom = (d ** 2).sum()
    values = {lag: z_crit * math.sqrt((d[lag:] ** 2 * d[:-lag] ** 2).sum()) / denom
              for lag in range(1, max_lag + 1)}
    return pd.Series(values, name="robust_band").rename_axis("lag")


def conditional_big_day(z, k=2.0):
    """P(big today | big yesterday) vs P(big today | not big yesterday), with ``big = |z| > k``.

    Only pairs of consecutive periods where both z values exist are used (the series must be on
    a regular grid). Returns a table with the count of each case.
    """
    big = (z.abs() > k).where(z.notna())
    pairs = pd.DataFrame({"yesterday": big.shift(1), "today": big}).dropna()
    rows = []
    for label, cond in [("big yesterday", True), ("not big yesterday", False)]:
        sub = pairs[pairs["yesterday"] == cond]
        n_big = int((sub["today"] == 1).sum())
        rows.append({"condition": label, "n_days": len(sub), "n_big_today": n_big,
                     "p_big_today": n_big / len(sub) if len(sub) else np.nan})
    return pd.DataFrame(rows).set_index("condition")


def sqrt_t_by_group(fine_returns, coarse_returns, fine_group, coarse_group, periods):
    """Compare ``std(fine) * sqrt(periods)`` with ``std(coarse)`` inside each group.

    Example: hourly returns (fine) vs daily returns (coarse), grouped by month, ``periods=24``.
    ``fine_group`` / ``coarse_group`` give each return its group label (same labels for both).
    Returns one row per group: both volatilities, their ratio (coarse / scaled fine) and the
    number of returns used on each side.
    """
    fine = fine_returns.dropna()
    coarse = coarse_returns.dropna()
    f = fine.groupby(fine_group.loc[fine.index]).agg(["std", "count"])
    c = coarse.groupby(coarse_group.loc[coarse.index]).agg(["std", "count"])
    out = pd.DataFrame({
        "scaled_fine_vol": f["std"] * math.sqrt(periods),
        "coarse_vol": c["std"],
        "n_fine": f["count"],
        "n_coarse": c["count"],
    }).dropna()
    out["ratio_coarse_over_scaled"] = out["coarse_vol"] / out["scaled_fine_vol"]
    return out


def sqrt_t_null_ratio(n):
    """What ``sqrt_t_by_group``'s ratio looks like **if the sqrt-t rule holds exactly**.

    Assumes normal returns and a precisely known scaled fine sigma; then the ratio is
    ``s / sigma`` for a std ``s`` computed from ``n`` coarse returns, and
    ``(n - 1) s^2 / sigma^2`` is chi-square with ``n - 1`` degrees of freedom. With few returns
    per group, ``s`` tends to come out **below** sigma, so the median ratio is below 1 even when
    the rule holds. Uses the Wilson-Hilferty approximation of the chi-square distribution.

    Returns ``(median ratio, probability that the ratio is > 1)``.
    """
    k = n - 1
    if k < 1:
        raise ValueError("need at least 2 returns per group")
    c = 2 / (9 * k)
    median_ratio = math.sqrt((1 - c) ** 3)
    p_above_one = 0.5 * math.erfc(math.sqrt(c) / math.sqrt(2))  # P(Z > sqrt(c))
    return median_ratio, p_above_one


def week_sunday(index):
    """The Sunday (as a tz-naive date) that ends the Monday-Sunday **UTC** week of each timestamp.

    Raises if the index is not in UTC: dropping another timezone would silently keep its local
    wall-clock time and put some candles in the wrong week.
    """
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("index must be timezone-aware UTC")
    return index.tz_localize(None).to_period("W-SUN").end_time.normalize()


def weekly_returns_from_daily(daily_close):
    """Monday-to-Sunday (UTC) weekly returns from daily closes: Sunday close / previous Sunday
    close - 1. Only weeks with all 7 daily candles count; the week is labelled by its Sunday.
    """
    daily_close = daily_close.sort_index()
    weeks = week_sunday(daily_close.index)
    full = daily_close.groupby(weeks).transform("size") == 7
    sundays = daily_close[(daily_close.index.day_of_week == 6) & full.to_numpy()]
    ret = sundays / sundays.shift(1) - 1
    consecutive = sundays.index.to_series().diff() == pd.Timedelta(days=7)
    return ret[consecutive.to_numpy()]
