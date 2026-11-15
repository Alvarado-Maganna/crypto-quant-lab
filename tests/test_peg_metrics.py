"""Tests for src/peg_metrics.py with small synthetic series whose answers are computed by hand."""

import pandas as pd
import pytest

from src.peg_metrics import (
    classify_wick,
    implied_cross,
    max_deviation,
    max_upside_deviation,
    minutes_off_peg,
    off_peg_span,
    rolling_median_close,
    share_in_band,
    time_to_recovery,
)

T0 = pd.Timestamp("2023-03-11 00:00", tz="UTC")
MIN = pd.Timedelta(minutes=1)


def series(values, start=T0):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="min"))


def test_max_deviation_value_and_time():
    dev, ts = max_deviation(series([1.0, 0.99, 0.95, 0.97]))
    assert dev == pytest.approx(-0.05)       # 0.95 - 1
    assert ts == T0 + 2 * MIN


def test_max_upside_deviation():
    dev, ts = max_upside_deviation(series([1.0, 1.002, 1.01, 1.0]))
    assert dev == pytest.approx(0.01) and ts == T0 + 2 * MIN


def test_minutes_off_peg_counts_only_strictly_outside_band():
    # u = 0.5%: 0.995 is ON the edge (not off), 0.99 and 1.006 are off
    close = series([1.0, 0.995, 0.99, 1.006, 1.0])
    assert minutes_off_peg(close, u=0.005) == 2


def test_minutes_off_peg_uses_clock_time_with_gaps():
    # two off-peg candles, with a missing minute between them: still 2 minutes, not 3
    close = pd.Series([0.98, 0.98], index=[T0, T0 + 2 * MIN])
    assert minutes_off_peg(close, u=0.005) == 2
    first, last_end, span = off_peg_span(close, u=0.005)
    assert first == T0 and last_end == T0 + 3 * MIN and span == 3 * MIN


def test_recovery_needs_the_band_to_hold():
    # trough at minute 0; back in band at minute 2 but leaves again at 4; stays from 5 on
    close = series([0.90, 0.98, 1.0, 1.0, 0.98] + [1.0] * 10)
    t, d = time_to_recovery(close, T0, u=0.005, hold=3 * MIN, min_traded=3)
    assert t == T0 + 5 * MIN and d == 5 * MIN


def test_recovery_ignores_filled_candles_and_needs_enough_trades():
    close = series([0.90, 1.0, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0])
    traded = series([True, True, False, True, False, True, True, True])
    # minute 2 is out of band but has no trades, so it does not break the window
    t, _ = time_to_recovery(close, T0, u=0.005, hold=3 * MIN, traded=traded, min_traded=2)
    assert t == T0 + 1 * MIN
    # requiring 3 trades in the 3-minute window: [1,2,3] has 2, [3,4,5] has 2, [5,6,7] has 3
    t, _ = time_to_recovery(close, T0, u=0.005, hold=3 * MIN, traded=traded, min_traded=3)
    assert t == T0 + 5 * MIN
    # sensitivity with filled candles included: minute 2 now counts and breaks [1, 4)
    t, _ = time_to_recovery(close, T0, u=0.005, hold=3 * MIN, min_traded=3)
    assert t == T0 + 3 * MIN


def test_recovery_not_declared_if_data_ends_before_the_hold_window():
    close = series([0.90, 1.0, 1.0])
    assert time_to_recovery(close, T0, u=0.005, hold=5 * MIN, min_traded=1) == (pd.NaT, pd.NaT)


def test_share_in_band():
    assert share_in_band(series([1.0, 1.001, 0.998, 1.0]), band=0.001) == 0.75


def test_classify_wick():
    df = pd.DataFrame({
        "open":   [1.0, 1.0, 1.0, 1.0, 0.95, 0.95],
        "high":   [1.0, 1.0, 1.0, 1.0, 0.95, 0.95],
        "low":    [1.0, 0.90, 1.0, 0.95, 0.95, 0.95],
        "close":  [1.0, 1.0, 1.0, 0.95, 0.95, 0.95],
        "volume": [1, 2, 1, 1, 1, 1],
    }, index=pd.date_range(T0, periods=6, freq="min"))
    wick = classify_wick(df, T0 + MIN, window=2 * MIN)       # low 0.90, closes back at 1.0
    assert wick["label"] == "isolated wick" and wick["wick"] == pytest.approx(0.10)
    shift = classify_wick(df, T0 + 3 * MIN, window=2 * MIN)  # close moves to 0.95 and stays
    assert shift["label"] == "level change"


def test_classify_wick_cluster_of_spikes_is_not_a_level_change():
    # level 1.004; three spike minutes closing near 1.047; then back to 1.004
    closes = [1.004] * 5 + [1.047, 1.047, 1.047] + [1.004] * 5
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                       "volume": 1.0}, index=pd.date_range(T0, periods=len(closes), freq="min"))
    # the 3rd spike: its previous close is also a spike, but the median before is 1.004
    out = classify_wick(df, T0 + 7 * MIN, window=5 * MIN)
    assert out["median_close_before"] == pytest.approx(1.004)
    assert out["label"] == "isolated wick"


def test_rolling_median_ignores_short_spikes_and_never_looks_ahead():
    close = series([1.0, 1.0, 1.05, 1.0, 1.0, 0.9, 0.9, 0.9])
    med = rolling_median_close(close, window=3 * MIN)
    assert med.iloc[2] == 1.0          # a 1-minute spike does not move the median
    assert med.iloc[6] == 0.9          # a move that stays is followed
    assert med.iloc[4] == 1.0          # at minute 4 the drop at minute 5 is not visible yet


def test_implied_cross_uses_only_common_timestamps():
    usdc = series([0.90, 0.95, 1.0])
    usdt = pd.Series([1.0, 1.05], index=[T0 + MIN, T0 + 2 * MIN])
    cross = implied_cross(usdc, usdt)
    assert list(cross.index) == [T0 + MIN, T0 + 2 * MIN]
    assert cross.tolist() == pytest.approx([0.95, 1.0 / 1.05])
