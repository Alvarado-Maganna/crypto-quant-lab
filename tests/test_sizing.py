"""Tests for src/sizing.py (small hand-made paths, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.sizing import (EXIT_END, EXIT_STOP, EXIT_TIME, day_tuples, demean_tuples,
                        fraction_all_in, fraction_fixed_risk, fraction_vol_target,
                        iid_bootstrap_indices, longest_losing_streak, max_drawdown, path_sigma,
                        required_recovery, simulate, simulate_buy_hold, size,
                        stationary_bootstrap_indices, stop_fill)


# --- sizing rule ---

def test_size_brief_example():
    qty = size(capital=10_000, risk=0.01, entry=100_000, stop=97_000)
    assert qty == pytest.approx(1 / 30)                 # 0.0333... BTC
    assert qty * 100_000 == pytest.approx(3_333.33, abs=0.01)


def test_size_rejects_stop_above_entry():
    with pytest.raises(ValueError):
        size(10_000, 0.01, 100, 100)


def test_fixed_risk_fraction_is_risk_over_stop():
    f = fraction_fixed_risk(0.01, 0.05)(np.array([0.5]), np.array([123.0]), np.array([7_000.0]))
    assert f[0] == pytest.approx(0.20)


def test_vol_target_fraction_is_capped_at_one():
    f = fraction_vol_target(0.10)(np.array([0.05, 0.20, 0.50]), np.ones(3), np.ones(3))
    assert f == pytest.approx([1.0, 0.5, 0.2])


# --- stop fills ---

def test_stop_fill_gap_at_open_fills_at_open():
    hit, fill = stop_fill(open_=90, low=85, stop=95, slippage=0.01)
    assert hit and fill == 90                           # worse than the stop, no slippage added


def test_stop_fill_open_equal_to_stop_counts_as_gap():
    hit, fill = stop_fill(95, 94, 95, 0.01)
    assert hit and fill == 95


def test_stop_fill_intraday_touch_uses_slippage():
    hit, fill = stop_fill(100, 94, 95, 0.01)
    assert hit and fill == pytest.approx(95 * 0.99)
    hit0, fill0 = stop_fill(100, 95, 95, 0.0)           # low exactly at the stop
    assert hit0 and fill0 == 95


def test_stop_fill_not_hit():
    hit, fill = stop_fill(100, 95.01, 95, 0.01)
    assert not hit and np.isnan(fill)


# --- tuples ---

def ohlc_frame(rows, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def test_day_tuples_relative_to_previous_close():
    t = day_tuples(ohlc_frame([[1, 1, 1, 100], [100, 110, 90, 105], [105, 106, 100, 102]]))
    assert len(t) == 2
    assert t.iloc[0].tolist() == pytest.approx([1.0, 1.1, 0.9, 1.05])
    assert t.iloc[1].tolist() == pytest.approx([105 / 105, 106 / 105, 100 / 105, 102 / 105])


def test_day_tuples_rejects_gap():
    df = ohlc_frame([[1, 1, 1, 100], [100, 110, 90, 105], [105, 106, 100, 102]])
    with pytest.raises(ValueError):
        day_tuples(df.drop(df.index[1]))


def test_demean_gives_zero_mean_and_keeps_shape():
    t = day_tuples(ohlc_frame([[1, 1, 1, 100], [100, 110, 90, 105], [105, 106, 100, 102],
                               [102, 120, 101, 118]]))
    d, m = demean_tuples(t)
    assert m == pytest.approx((t["close"] - 1).mean())
    assert (d["close"] - 1).mean() == pytest.approx(0.0, abs=1e-15)
    assert (d["close"] - 1).to_numpy() == pytest.approx((t["close"] - 1 - m).to_numpy())
    for col in ["open", "high", "low"]:                  # same ratio to the close
        assert (d[col] / d["close"]).to_numpy() == pytest.approx((t[col] / t["close"]).to_numpy())


# --- bootstrap ---

def test_stationary_bootstrap_blocks_and_mean_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 2_000, 400, mean_block=20, rng=rng)
    assert idx.min() >= 0 and idx.max() < 500
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 500)
    assert 1 - cont.mean() == pytest.approx(1 / 20, abs=0.003)   # new block about every 20 days


def test_bootstraps_are_reproducible_with_seed():
    a = stationary_bootstrap_indices(50, 3, 10, 5, np.random.default_rng(7))
    b = stationary_bootstrap_indices(50, 3, 10, 5, np.random.default_rng(7))
    assert (a == b).all()
    c = iid_bootstrap_indices(50, 3, 10, np.random.default_rng(7))
    assert c.shape == (3, 10) and c.max() < 50


# --- past-only sigma inside paths ---

def test_path_sigma_uses_only_previous_returns():
    rng = np.random.default_rng(1)
    ratios = 1 + rng.normal(0, 0.03, (4, 80))
    sig = path_sigma(ratios, window=30)
    assert np.isnan(sig[:, :30]).all() and np.isfinite(sig[:, 30:]).all()
    assert sig[0, 30] == pytest.approx(np.std(ratios[0, :30] - 1, ddof=1) * np.sqrt(365))


@pytest.mark.parametrize("cut", [30, 45, 79])
def test_path_sigma_truncation_no_look_ahead(cut):
    """Changing every return from day ``cut`` on must not change sigma on days <= cut."""
    rng = np.random.default_rng(2)
    ratios = 1 + rng.normal(0, 0.03, (3, 80))
    changed = ratios.copy()
    changed[:, cut:] = 1 + rng.normal(0, 0.10, (3, 80 - cut))
    a, b = path_sigma(ratios), path_sigma(changed)
    np.testing.assert_array_equal(a[:, :cut + 1], b[:, :cut + 1])
    if cut < 79:                                         # the change does show up afterwards
        assert not np.allclose(a[:, cut + 1:], b[:, cut + 1:])


# --- simulator on hand-made paths ---

FLAT = [1.0, 1.0, 1.0, 1.0]


def run(days, coin_day, schemes=None, **kw):
    """One path made of the given tuples; enter only at the close of ``coin_day``."""
    tuples = np.array(days, dtype=float)
    n = len(days)
    idx = np.arange(n)[None, :]
    coin = np.ones((1, n))
    coin[0, coin_day] = 0.0
    sigma = np.full((1, n), 0.5)
    schemes = schemes or {"all_in": fraction_all_in}
    kw.setdefault("warmup", 2)
    return simulate(tuples, idx, coin, sigma, schemes, record_path=0, **kw)


def test_entry_next_open_stop_intraday_and_costs():
    days = [FLAT, FLAT, FLAT, [1.0, 1.0, 0.94, 0.96], FLAT]
    paths, trades, _ = run(days, coin_day=2, fee=0.001, slippage=0.0)
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["entry_day"] == 3 and tr["exit_day"] == 3 and tr["exit_type"] == EXIT_STOP
    assert tr["exit_px"] == pytest.approx(0.95) and not tr["gap_open"]
    notional = 10_000 / 1.001
    expected = notional / 1.0 * 0.95 * 0.999
    assert paths["all_in"]["final_equity"].iloc[0] == pytest.approx(expected)
    assert tr["pnl_all_in"] == pytest.approx(expected / 10_000 - 1)


def test_gap_below_stop_fills_at_open():
    days = [FLAT, FLAT, FLAT, [1.0, 1.0, 0.99, 1.0], [0.90, 0.92, 0.88, 0.91], FLAT]
    _, trades, _ = run(days, coin_day=2, fee=0.0, slippage=0.01)
    tr = trades.iloc[0]
    assert tr["exit_day"] == 4 and tr["exit_px"] == pytest.approx(0.90)   # gap: open, no slippage
    assert tr["gap_open"]


def test_time_exit_at_close_of_day_entry_plus_max_hold():
    days = [FLAT] * 3 + [[1.0, 1.01, 0.99, 1.0]] * 30
    _, trades, _ = run(days, coin_day=2, fee=0.0, max_hold=20)
    tr = trades.iloc[0]
    assert tr["entry_day"] == 3 and tr["exit_day"] == 23 and tr["exit_type"] == EXIT_TIME


def test_open_trade_closed_on_last_day_and_no_entry_in_warmup():
    days = [[1.0, 1.0, 0.9, 1.0]] * 3 + [FLAT] * 3
    paths, trades, _ = run(days, coin_day=0, warmup=3)       # coin at day 0 is ignored (warm-up)
    assert trades.empty and paths["all_in"]["n_trades"].iloc[0] == 0
    paths, trades, _ = run(days, coin_day=3, warmup=3)
    assert trades.iloc[0]["exit_type"] == EXIT_END and trades.iloc[0]["exit_day"] == 5


def test_fixed_risk_loses_about_one_percent_at_the_stop():
    days = [FLAT, FLAT, FLAT, [1.0, 1.0, 0.90, 0.92], FLAT]
    schemes = {"all_in": fraction_all_in, "risk": fraction_fixed_risk(0.01, 0.05)}
    _, trades, _ = run(days, coin_day=2, schemes=schemes, fee=0.0)
    assert trades.iloc[0]["frac_risk"] == pytest.approx(0.20)
    assert trades.iloc[0]["pnl_risk"] == pytest.approx(-0.01)
    assert trades.iloc[0]["pnl_all_in"] == pytest.approx(-0.05)


def test_spot_never_spends_more_than_equity():
    days = [FLAT, FLAT, FLAT, FLAT]
    paths, trades, rec = run(days, coin_day=2, fee=0.001)
    assert trades.iloc[0]["frac_all_in"] == pytest.approx(1 / 1.001)
    assert rec["exposure"][0].max() <= 1.0 + 1e-12


def test_incremental_max_dd_matches_recorded_equity():
    rng = np.random.default_rng(3)
    c = 1 + rng.normal(0, 0.03, 200)
    days = np.column_stack([np.ones(200), c + 0.01, np.minimum(1, c) - 0.01, c])
    tuples = days
    idx = np.arange(200)[None, :]
    coin = rng.random((1, 200))
    sigma = np.full((1, 200), 0.5)
    paths, _, rec = simulate(tuples, idx, coin, sigma, {"a": fraction_all_in}, record_path=0)
    assert paths["a"]["max_dd"].iloc[0] == pytest.approx(max_drawdown(rec["equity"][0]))
    assert paths["a"]["min_equity"].iloc[0] == pytest.approx(rec["equity"][0].min())


def test_buy_hold_matches_hand_calculation():
    tuples = np.array([FLAT, FLAT, [1.0, 1.0, 1.0, 1.10], [1.0, 1.0, 1.0, 0.5]])
    out = simulate_buy_hold(tuples, np.arange(4)[None, :], fee=0.001, warmup=2)
    qty = 10_000 / 1.001
    assert out["final_equity"].iloc[0] == pytest.approx(qty * 1.10 * 0.5 * 0.999)
    assert out["max_dd"].iloc[0] == pytest.approx(1 - 0.5 * 0.999)


# --- path statistics ---

def test_max_drawdown_and_recovery():
    assert max_drawdown([100, 120, 60, 130, 117]) == pytest.approx(0.5)
    assert required_recovery(0.5) == pytest.approx(1.0)
    assert required_recovery(0.2) == pytest.approx(0.25)


def test_longest_losing_streak():
    trades = pd.DataFrame({"path": [0, 0, 0, 0, 0, 0, 1, 1, 2],
                           "pnl": [-1, -1, 1, -1, -1, -1, 1, 1, -1]})
    out = longest_losing_streak(trades, n_paths=4, pnl_col="pnl")
    assert out.tolist() == [3, 0, 1, 0]               # path 3 has no trade
