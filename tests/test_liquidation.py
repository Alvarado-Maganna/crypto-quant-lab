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
