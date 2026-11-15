"""Tests for src/cross_exchange.py."""

import pandas as pd
import pytest

from src.cross_exchange import (
    bps_diff,
    common_minutes,
    coverage_report,
    longest_run,
    summarize_diff,
)

T0 = pd.Timestamp("2025-10-10 18:00", tz="UTC")


def minutes(n, start=0):
    return pd.date_range(T0 + pd.Timedelta(minutes=start), periods=n, freq="1min")


def frame(index, close=100.0, volume=1.0):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": volume}, index=index)


def test_bps_diff_sign_and_scale():
    assert bps_diff(pd.Series([100.2]), pd.Series([100.0]))[0] == pytest.approx(20.0)
    assert bps_diff(pd.Series([99.0]), pd.Series([100.0]))[0] == pytest.approx(-100.0)


def test_common_minutes_requires_presence_everywhere_and_volume():
    a = frame(minutes(5))
    b = frame(minutes(4, start=1))                  # missing minute 0
    c = frame(minutes(5)).assign(volume=[1, 1, 0, 1, 1])  # minute 2 had no trades
    assert list(common_minutes({"a": a, "b": b, "c": c})) == list(minutes(5)[[1, 3, 4]])
    assert len(common_minutes({"a": a, "b": b, "c": c}, require_volume=False)) == 4


def test_longest_run_counts_consecutive_minutes():
    above = pd.Series([True, True, False, True, True, True, False], index=minutes(7))
    assert longest_run(above) == 3
    assert longest_run(above & False) == 0


def test_longest_run_is_broken_by_a_missing_minute():
    idx = minutes(6).delete(3)                     # 18:03 missing
    above = pd.Series(True, index=idx)
    assert longest_run(above) == 3                 # 18:00-18:02, then 18:04-18:05


def test_summarize_diff():
    diff = pd.Series([0, 5, -25, 30, 1], index=minutes(5), dtype=float)
    s = summarize_diff(diff, threshold_bps=20)
    assert s["median_bps"] == 1
    assert s["max_abs_bps"] == 30 and s["max_at_utc"] == minutes(5)[3]
    assert s["pct_above"] == pytest.approx(40.0)
    assert s["longest_run_min"] == 2


def test_coverage_report_counts_gaps_and_fills():
    idx = minutes(10).delete([4, 5])
    df = frame(idx).assign(volume=[1, 1, 1, 1, 0, 1, 1, 1])  # 18:07 is a filled candle
    row = coverage_report(df, T0, T0 + pd.Timedelta(minutes=10), {"duplicates_dropped": 3})
    assert row["expected"] == 10 and row["present"] == 8 and row["gaps"] == 2
    assert row["filled_vol0"] == 1 and row["filled_flat_prev_close"] == 1
    assert row["duplicates_dropped"] == 3 and row["volume_base"] == 7


def test_runs_above_lists_each_run():
    from src.cross_exchange import runs_above

    diff = pd.Series([0, 25, -30, 0, 21, 0], index=minutes(6), dtype=float)
    runs = runs_above(diff, 20)
    assert runs["minutes"].tolist() == [2, 1]
    assert runs.loc[0, "max_abs_bps"] == 30 and runs.loc[0, "signed_at_max_bps"] == -30
    assert runs.loc[0, "first_utc"] == minutes(6)[1]
    assert runs_above(diff * 0, 20).empty
