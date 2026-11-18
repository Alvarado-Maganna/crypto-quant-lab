"""Funding-rate arithmetic for perpetual futures (notebook 05, module M7).

A perpetual has no expiry, so nothing forces its price to meet the spot price. Funding is the
mechanism that pulls them together: at each funding time, one side pays the other

    payment = notional * funding_rate          (notional = qty * mark price at that time)

A **positive** rate means **longs pay shorts** (the perp tends to trade above the index, so
holding a long is made more expensive); a negative rate means shorts pay longs.

Binance (USD-M) computes the rate of each interval as

    F = P + clamp(I - P, -0.05%, +0.05%)        (then divided by 8/N for N-hour intervals)

where P is the average premium of the perp over the index during the interval and I is the
interest rate, 0.01% per 8 h. If P is between -0.04% and +0.06%, the clamp absorbs the
difference and F is **exactly** I = 0.01%: the "dead zone".

Units: rates are **fractions per funding interval** (0.0001 = 0.01% = 1 bp per interval).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365          # 8,760; we use 365-day years (crypto trades every day)
BASELINE_RATE = Decimal("0.0001")  # 0.01% per 8 h: Binance's interest component
SIDES = ("long", "short")


# --------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------


def infer_intervals(funding_time_ms):
    """Time between consecutive funding payments, from the raw ms timestamps.

    Returns a DataFrame aligned with the input (first row NaN) with:
    - ``interval_ms``: the raw difference in milliseconds (kept: timestamps can be a few ms
      off the hour, e.g. 16:00:00.007);
    - ``interval_hours``: the same in hours (float);
    - ``interval_hours_rounded``: rounded to the nearest whole hour, **only for classifying**
      (e.g. 8 h, 4 h, 1 h). Never used for arithmetic.
    Raises if the timestamps are not strictly increasing.
    """
    t = pd.Series(funding_time_ms, dtype="int64").reset_index(drop=True)
    diff = t.diff()
    if (diff.dropna() <= 0).any():
        raise ValueError("funding times must be strictly increasing")
    hours = diff / 3_600_000
    return pd.DataFrame({"interval_ms": diff, "interval_hours": hours,
                         "interval_hours_rounded": hours.round().astype("Int64")})


# --------------------------------------------------------------------------------------
# Annualization
# --------------------------------------------------------------------------------------


def payments_per_year(interval_hours):
    """How many funding payments fit in a 365-day year: 8,760 / interval_hours (1,095 for 8 h)."""
    interval_hours = np.asarray(interval_hours, dtype=float)
    if (interval_hours <= 0).any():
        raise ValueError("interval_hours must be > 0")
    return HOURS_PER_YEAR / interval_hours


def annualize_simple(rate, interval_hours):
    """Simple annualization: rate * payments per year (no compounding).

    0.01% every 8 h -> 0.0001 * 1,095 = 10.95% per year. This is how exchanges and most
    dashboards quote "APR". Works element-wise on arrays/Series.
    """
    return rate * payments_per_year(interval_hours)


def annualize_compound(rate, interval_hours):
    """Compound annualization: (1 + rate) ** payments_per_year - 1.

    It assumes every payment is reinvested at the same rate, which rarely holds; it is always
    >= the simple figure for positive rates. 0.01% every 8 h -> about 11.57%.
    """
    return (1 + np.asarray(rate, dtype=float)) ** payments_per_year(interval_hours) - 1


# --------------------------------------------------------------------------------------
# Payments
# --------------------------------------------------------------------------------------


def funding_payment(side, notional, rate):
    """Funding **paid** by a position at one funding time, in quote currency (USDT).

    Positive = the position pays; negative = it receives.
    long pays ``notional * rate``; short pays ``-notional * rate``.
    ``notional`` is the absolute position value at the mark price (>= 0).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    notional = np.asarray(notional, dtype=float)
    if (notional < 0).any():
        raise ValueError("notional must be >= 0 (use side to say long or short)")
    sign = 1.0 if side == "long" else -1.0
    return sign * notional * rate


def total_funding_paid(side, qty, mark_prices, rates):
    """Total funding paid over several funding times by a position of fixed size ``qty``.

    At each time the notional is ``qty * mark_price`` (the mark of *that* funding time), so a
    position of fixed size pays less when the price has fallen. Returns a float (USDT).
    """
    mark_prices = np.asarray(mark_prices, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if mark_prices.shape != rates.shape:
        raise ValueError("one mark price per rate is needed")
    if np.isnan(mark_prices).any():
        raise ValueError("missing mark price")
    return float(np.sum(funding_payment(side, qty * mark_prices, rates)))


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def is_baseline(rate_str, tolerance=0.0, baseline=BASELINE_RATE):
    """Which rates equal the baseline (0.01%)?

    ``rate_str``: the rates as the **strings** sent by the API (e.g. ``"0.00010000"``).
    ``tolerance=0`` compares exactly with ``Decimal`` (no float rounding at all).
    ``tolerance>0`` accepts ``|rate - baseline| <= tolerance``, in rate units (1e-6 = 0.0001%,
    i.e. 1% of the 0.01% baseline). The tolerance comparison is also done in ``Decimal``.
    Returns a boolean numpy array.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    tol = Decimal(str(tolerance))
    return np.array([abs(Decimal(s) - baseline) <= tol for s in rate_str], dtype=bool)


def sign_shares(rate_str, tolerances=(0.0,)):
    """Share (%) of payments that are positive, negative, zero and equal to the baseline.

    One row per tolerance (only the baseline column depends on it). Signs are taken with
    ``Decimal`` on the API strings, so ``"0.00000000"`` counts as zero, not as positive.
    """
    values = [Decimal(s) for s in rate_str]
    n = len(values)
    if n == 0:
        raise ValueError("no rates")
    rows = {}
    for tol in tolerances:
        rows[tol] = {
            "payments": n,
            "positive (%)": 100 * sum(v > 0 for v in values) / n,
            "negative (%)": 100 * sum(v < 0 for v in values) / n,
            "zero (%)": 100 * sum(v == 0 for v in values) / n,
            "= 0.01% (%)": 100 * is_baseline(rate_str, tol).mean(),
            "= 0.01% (count)": int(is_baseline(rate_str, tol).sum()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "tolerance"
    return out


def binance_funding_from_premium(avg_premium, interest=0.0001, clamp=0.0005):
    """Funding of one 8-hour interval from its average premium: P + clamp(I - P, -c, +c).

    With the defaults, any P in [-0.04%, +0.06%] gives exactly I = 0.01% (the dead zone).
    The exchange cap/floor on the final rate is not applied here.
    """
    p = np.asarray(avg_premium, dtype=float)
    return p + np.clip(interest - p, -clamp, clamp)
