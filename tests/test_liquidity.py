"""Offline tests of src/liquidity.py."""

import numpy as np
import pandas as pd
import pytest

from src import liquidity as lq


def test_cdmx_hour_is_utc_minus_6():
    assert lq.cdmx_hour(14) == 8
    assert lq.cdmx_hour(3) == 21
    assert list(lq.cdmx_hour([0, 5, 6])) == [18, 23, 0]
    assert lq.hour_label(3) == "03 UTC (21 CDMX)"


def test_add_time_labels_uses_utc_calendar():
    df = pd.DataFrame({"t": pd.to_datetime(["2026-11-23 03:10", "2026-11-22 23:59"], utc=True)})
    out = lq.add_time_labels(df, "t")
    assert out["weekday_name"].tolist() == ["Mon", "Sun"]       # Monday 03 UTC is Sunday in CDMX
    assert out["hour_cdmx"].tolist() == [21, 17]


def wide_from(rows):
    raw = pd.DataFrame(rows, columns=["timestamp_utc", "percentage", "depth", "notional"])
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    return lq.book_depth_wide(raw)


def test_check_cumulative_detects_non_monotonic():
    t1, t2 = "2026-11-22 00:00:01", "2026-11-22 00:00:31"
    rows = []
    for t, far_bid in ((t1, 30.0), (t2, 5.0)):         # t2: the -1% value is below -0.2%
        rows += [(t, -1.0, far_bid, far_bid * 60000), (t, -0.2, 10.0, 600000.0),
                 (t, 0.2, 9.0, 541000.0), (t, 1.0, 40.0, 2420000.0)]
    res = lq.check_cumulative(wide_from(rows))
    assert res.loc["bid (negative %)", "monotonic (% of timestamps)"] == 50.0
    assert res.loc["ask (positive %)", "monotonic (% of timestamps)"] == 100.0


def test_side_totals_and_implied_price():
    t = "2026-11-22 00:00:01"
    w = wide_from([(t, -0.2, 10.0, 600000.0), (t, 0.2, 9.0, 541000.0),
                   (t, -1.0, 30.0, 1790000.0), (t, 1.0, 40.0, 2420000.0)])
    s = lq.book_depth_side_totals(w, pcts=(0.2, 1.0))
    assert s["0.2pct_total"].iloc[0] == 1141000.0
    assert s["1pct_bid"].iloc[0] == 1790000.0
    assert lq.implied_price(w)[-0.2].iloc[0] == 60000.0


def test_match_nearest_respects_tolerance():
    left = pd.DataFrame({"timestamp_utc": pd.to_datetime(
        ["2026-11-22 00:00:00", "2026-11-22 00:10:00"], utc=True), "a": [1, 2]})
    right = pd.DataFrame({"local_receive_utc": pd.to_datetime(
        ["2026-11-22 00:00:20", "2026-11-22 00:11:00"], utc=True), "b": [10, 20]})
    out = lq.match_nearest(left, right, tolerance="30s")
    assert out["a"].tolist() == [1] and out["b"].tolist() == [10]
    assert out["gap_s"].tolist() == [20.0]


def test_lagged_correlation_finds_a_shift():
    idx = pd.date_range("2026-11-22", periods=48, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=48), index=idx)
    x = y.copy()
    x.index = x.index - pd.Timedelta(hours=3)         # x's clock is 3 h behind y's
    res = lq.lagged_correlation(x, y, range(-5, 6))
    assert res["corr"].idxmax() == 3
    assert res.loc[3, "corr"] == pytest.approx(1.0)


def test_week_heatmap_ratio_and_percentile():
    idx = pd.date_range("2026-11-02", periods=24 * 14, freq="1h", tz="UTC")   # Mon, 2 weeks
    h = lq.add_time_labels(pd.DataFrame({"t": idx}), "t")
    h["v"] = np.where(lq.in_window(h, **lq.BUSINESS), 2.0, 1.0)
    table = lq.week_heatmap(h, "v")
    assert table.shape == (7, 24) and table.loc["Tue", 14] == 2.0 and table.loc["Sun", 3] == 1.0
    assert lq.cell_percentile(table, "Tue", 14) == 100.0
    r = lq.ratio_by_month(h, "v")
    assert r["ratio"].iloc[0] == 2.0
    assert r["numerator hours"].iloc[0] == 10 * 8            # 10 weekdays x 8 hours


def test_stress_table():
    idx = pd.date_range("2026-08-01", periods=100, freq="D", tz="UTC")
    close = pd.Series(100.0, index=idx)
    daily = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close})
    daily.loc[idx[-1], ["high", "low"]] = [110.0, 90.0]     # last day: 20% range
    res = lq.stress_table(daily, [idx[-1], idx[-2]], lookback=90)
    assert bool(res["stress"].iloc[0]) is True
    assert res["range percentile"].iloc[0] == 100.0
    assert bool(res["stress"].iloc[1]) is False


def test_daily_from_minutes():
    idx = pd.date_range("2026-11-22", periods=3, freq="12h", tz="UTC")
    c = pd.DataFrame({"open": [1, 2, 3], "high": [2, 4, 5], "low": [0.5, 1, 2],
                      "close": [1.5, 3, 4], "volume": [1, 1, 1], "quote_volume": [1, 3, 4]},
                     index=idx, dtype=float)
    d = lq.daily_from_minutes(c)
    assert d["high"].iloc[0] == 4 and d["minutes"].tolist() == [2, 1]


def test_window_month_blocks():
    dates = pd.to_datetime(["2025-11-24", "2025-12-23", "2025-12-24", "2026-11-23"], utc=True)
    labels = lq.window_month(dates)
    assert labels[0] == labels[1] == "2025-11-24 to 2025-12-23"
    assert labels[2] == "2025-12-24 to 2026-01-23"
    assert labels[3] == "2026-10-24 to 2026-11-23"
    year = pd.date_range("2025-11-24", "2026-11-23", freq="D", tz="UTC")
    assert len(set(lq.window_month(year))) == 12


def test_side_totals_missing_percentage_is_nan():
    t = "2025-12-01 00:00:01"
    w = wide_from([(t, -1.0, 30.0, 1790000.0), (t, 1.0, 40.0, 2420000.0)])
    s = lq.book_depth_side_totals(w, pcts=(0.2, 1.0))
    assert s["0.2pct_total"].isna().all() and s["1pct_total"].iloc[0] == 4210000.0


def test_frozen_values_and_clean_book_depth():
    idx = pd.date_range("2026-01-01 23:58", periods=6, freq="30s", tz="UTC")   # crosses midnight
    sides = pd.DataFrame({"1pct_bid": [5.0, 6.0, 5.0, 5.0, 5.0, 9.0],          # 5.0 alternates
                          "1pct_ask": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    sides["1pct_total"] = sides["1pct_bid"] + sides["1pct_ask"]
    frozen = lq.frozen_values(sides, min_count=2)
    # day 1 (4 rows): 5.0 three times, not in a row -> frozen; day 2 (2 rows): 5.0 once -> not
    assert frozen["1pct_bid"].tolist() == [True, False, True, True, False, False]
    assert not frozen["1pct_ask"].any()
    clean, frozen, share, bad = lq.clean_book_depth(sides, min_count=2, max_day_share=0.6, pcts=(1.0,))
    assert clean["1pct_total"].isna().tolist() == [True, False, True, True, False, False]
    assert share.loc[pd.Timestamp("2026-01-01", tz="UTC"), "1pct_bid"] == 0.75
    assert list(bad) == [pd.Timestamp("2026-01-01", tz="UTC")]                  # 3 of 4 frozen
