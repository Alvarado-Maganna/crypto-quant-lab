"""Tests for src/liquidation.py against cases computed by hand (tolerance 0.01 USDT)."""

import pytest

from src.liquidation import (approx_liquidation_price, equity_and_maintenance,
                             liquidation_price_isolated)

TOL = 0.01


def test_case1_long_10x_mmr_050pct():
    # (1000 - 0.1*100000) / (0.1*0.005 - 0.1) = -9000 / -0.0995
    lp = liquidation_price_isolated("long", 100_000, 0.1, 1_000, 0.005)
    assert lp == pytest.approx(90_452.26, abs=TOL)


def test_case2_long_10x_mmr_040pct():
    # -9000 / (0.1*0.004 - 0.1) = -9000 / -0.0996
    lp = liquidation_price_isolated("long", 100_000, 0.1, 1_000, 0.004)
    assert lp == pytest.approx(90_361.45, abs=TOL)


def test_case3_short_10x_mmr_040pct():
    # (1000 + 0.1*100000) / (0.1*0.004 + 0.1) = 11000 / 0.1004
    lp = liquidation_price_isolated("short", 100_000, 0.1, 1_000, 0.004)
    assert lp == pytest.approx(109_561.75, abs=TOL)


def test_case4_long_100x():
    # (100 - 10000) / -0.0996: a 0.6% drop is enough
    lp = liquidation_price_isolated("long", 100_000, 0.1, 100, 0.004)
    assert lp == pytest.approx(99_397.59, abs=TOL)


def test_case5_shortcut_is_within_0p1pct_of_exact():
    # 100000 * (1 - 1/10 + 0.005) = 90500
    approx = approx_liquidation_price("long", 100_000, 10, 0.005)
    exact = liquidation_price_isolated("long", 100_000, 0.1, 1_000, 0.005)
    assert approx == pytest.approx(90_500.00, abs=TOL)
    assert abs(approx - exact) / exact < 0.001


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("margin,mmr,cum", [(1_000, 0.004, 0.0), (100, 0.004, 0.0),
                                            (1_000, 0.005, 50.0)])
def test_case6_equity_equals_maintenance_at_liquidation(side, margin, mmr, cum):
    lp = liquidation_price_isolated(side, 100_000, 0.1, margin, mmr, cum)
    equity, maintenance = equity_and_maintenance(side, lp, 100_000, 0.1, margin, mmr, cum)
    assert equity == pytest.approx(maintenance, abs=TOL)


def test_short_shortcut_mirrors_long():
    assert approx_liquidation_price("short", 100_000, 10, 0.004) == pytest.approx(109_600, abs=TOL)


@pytest.mark.parametrize("kwargs", [dict(side="buy"), dict(entry_price=0), dict(qty=-0.1),
                                    dict(margin=0), dict(mmr=0.4 * 100)])
def test_bad_inputs_raise(kwargs):
    args = dict(side="long", entry_price=100_000, qty=0.1, margin=1_000, mmr=0.004) | kwargs
    with pytest.raises(ValueError):
        liquidation_price_isolated(**args)


# --- simulation on candles ------------------------------------------------------------

import pandas as pd  # noqa: E402

from src.liquidation import first_touch_at_or_below, simulate_isolated_longs  # noqa: E402

T = pd.date_range("2025-10-09", periods=6, freq="1min", tz="UTC")


def test_entry_candle_low_is_ignored_no_look_ahead():
    # The entry candle (minute 0) dips to 80, but that happened before the position existed.
    low = pd.Series([80, 99, 98, 97, 96, 95], index=T, dtype=float)
    touched = first_touch_at_or_below(low, T[:1], [90.0])
    assert pd.isna(touched.iloc[0])


def test_first_touch_is_the_earliest_and_equal_counts():
    low = pd.Series([100, 99, 90, 85, 90, 80], index=T, dtype=float)
    touched = first_touch_at_or_below(low, T[[0, 3]], [90.0, 90.0])
    assert touched.iloc[0] == T[2]   # low == level counts, and the later 85/80 are not chosen
    assert touched.iloc[1] == T[4]   # search starts the minute after entry, not at it


def test_entry_on_last_candle_is_never_touched():
    low = pd.Series([100, 99, 98, 97, 96, 1], index=T, dtype=float)
    assert pd.isna(first_touch_at_or_below(low, T[-1:], [50.0]).iloc[0])


def test_never_touched_keeps_utc_dtype_and_mixes_with_touched():
    low = pd.Series([100, 99, 50, 97, 96, 95], index=T, dtype=float)
    never = first_touch_at_or_below(low, T[:2], [10.0, 10.0])
    assert never.isna().all() and never.dtype == low.index.dtype
    sim = simulate_isolated_longs(pd.Series(100.0, index=T[:1]), {"mark": low}, [1.5, 10],
                                  1_000, 0.004)   # 1.5x survives, 10x is liquidated
    assert sim["liquidated_mark"].dtype == low.index.dtype


def test_entry_must_be_a_candle():
    low = pd.Series(100.0, index=T)
    with pytest.raises(ValueError):
        first_touch_at_or_below(low, [T[0] + pd.Timedelta(seconds=30)], [90.0])


def test_simulate_uses_exact_lp_and_each_low_series():
    close = pd.Series(100_000.0, index=T[:1])
    mark = pd.Series([100_000, 95_000, 92_000, 91_000, 90_000, 90_000], index=T, dtype=float)
    last = pd.Series([100_000, 95_000, 90_300, 91_000, 90_000, 90_000], index=T, dtype=float)
    sim = simulate_isolated_longs(close, {"mark": mark, "last": last}, [10], 1_000, 0.004)
    row = sim.loc[(10, T[0])]
    assert row["lp"] == pytest.approx(90_361.45, abs=TOL)        # same as hand case 2
    assert row["notional"] == pytest.approx(10_000)
    assert row["drop_to_lp_pct"] == pytest.approx(9.6386, abs=1e-3)
    assert row["liquidated_last"] == T[2]                         # last wick 90,300 < LP
    assert row["liquidated_mark"] == T[4]                         # mark only at 90,000
    dtype = sim["liquidated_mark"].dtype   # tz-aware datetimes, not plain objects
    assert isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"
