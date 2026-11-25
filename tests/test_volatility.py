"""Tests for src/volatility.py (small hand-made series, no network)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.volatility import (acf, acf_robust_band, annualize, conditional_big_day, normal_two_sided_tail,
                            past_only_sigma, past_only_z, sqrt_t_by_group, step_returns,
                            sqrt_t_null_ratio, student_t_two_sided_tail, tail_table,
                            week_sunday, weekly_returns_from_daily)

DAY = pd.Timedelta(days=1)


def days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def test_simple_returns_not_log():
    close = pd.Series([100.0, 110.0, 99.0], index=days(3))
    r = step_returns(close, DAY)
    assert r.tolist() == pytest.approx([0.10, -0.10])
    assert r.index[0] == days(3)[1]  # the return belongs to the later candle


def test_return_across_a_missing_candle_is_nan():
    idx = days(4).delete(2)  # day 3 missing
    close = pd.Series([100.0, 101.0, 120.0], index=idx)
    r = step_returns(close, DAY)
    assert r.iloc[0] == pytest.approx(0.01)
    assert np.isnan(r.iloc[1])


def test_annualize_uses_sqrt_of_periods():
    assert annualize(0.01, 365) == pytest.approx(0.01 * math.sqrt(365))


def test_past_only_sigma_excludes_today():
    r = pd.Series([0.01, -0.01, 0.01, 0.50], index=days(4))
    sigma = past_only_sigma(r, window=3)
    # sigma on day 4 uses days 1-3 only: the 0.50 shock is not in it
    assert sigma.iloc[3] == pytest.approx(np.std([0.01, -0.01, 0.01], ddof=1))
    assert sigma.iloc[:3].isna().all()
    z = past_only_z(r, window=3)
    assert z.iloc[3] == pytest.approx(0.50 / sigma.iloc[3])


def test_past_only_sigma_rejects_irregular_index():
    r = pd.Series([0.01, 0.02, 0.03], index=days(4).delete(1))
    with pytest.raises(ValueError, match="regular grid"):
        past_only_sigma(r, window=2)


def test_normal_tail_values():
    assert normal_two_sided_tail(1.96) == pytest.approx(0.05, abs=1e-4)
    assert normal_two_sided_tail(3) == pytest.approx(0.0027, abs=1e-4)


def test_tail_table_counts_and_expected():
    z = pd.Series([0.0, 2.5, -3.5, 1.0, np.nan])
    t = tail_table(z, thresholds=(2, 3))
    assert t.loc["|z| > 2", "observed"] == 2 and t.loc["|z| > 3", "observed"] == 1
    assert t.loc["|z| > 2", "n_days"] == 4
    assert t.loc["|z| > 2", "expected_normal"] == pytest.approx(4 * normal_two_sided_tail(2))


def test_acf_of_alternating_series_is_minus_one_at_lag_1():
    values, n = acf([1, -1] * 50, max_lag=2)
    assert n == 100
    assert values[1] == pytest.approx(-0.99, abs=0.01)
    assert values[2] == pytest.approx(0.98, abs=0.01)


def test_conditional_big_day_counts_pairs():
    z = pd.Series([3.0, 3.0, 0.0, 0.0, 3.0, np.nan, 0.0], index=days(7))
    t = conditional_big_day(z, k=2)
    # pairs (y, t): (B,B) (B,n) (n,n) (n,B); pairs touching NaN are dropped
    assert t.loc["big yesterday", "n_days"] == 2
    assert t.loc["big yesterday", "p_big_today"] == pytest.approx(0.5)
    assert t.loc["not big yesterday", "n_days"] == 2
    assert t.loc["not big yesterday", "n_big_today"] == 1


def test_sqrt_t_ratio_is_one_for_iid_scaling():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    hourly = pd.Series(rng.normal(0, 0.01, len(hours)), index=hours)
    daily = (1 + hourly).groupby(hourly.index.floor("D")).prod() - 1
    group = lambda s: s.index.to_series().dt.year  # one group per year
    out = sqrt_t_by_group(hourly, daily, group(hourly), group(daily), periods=24)
    assert out["ratio_coarse_over_scaled"].iloc[0] == pytest.approx(1.0, abs=0.1)


def test_weekly_returns_are_sunday_to_sunday():
    idx = days(21, start="2024-01-01")  # 2024-01-01 is a Monday: 3 full weeks
    close = pd.Series(np.arange(1.0, 22.0), index=idx)
    w = weekly_returns_from_daily(close)
    assert list(w.index.day_of_week) == [6, 6]  # first Sunday has no previous Sunday
    assert w.iloc[0] == pytest.approx(14 / 7 - 1)


def test_weekly_returns_skip_incomplete_weeks():
    idx = days(21, start="2024-01-01").delete(9)  # a day missing in week 2
    close = pd.Series(np.arange(1.0, 21.0), index=idx)
    assert len(weekly_returns_from_daily(close)) == 0  # week 2 dropped, so no consecutive Sundays


def test_student_t_tail_known_values():
    assert student_t_two_sided_tail(1, df=1) == pytest.approx(0.5, abs=1e-4)      # Cauchy
    assert student_t_two_sided_tail(2, df=10) == pytest.approx(0.0734, abs=1e-4)
    assert student_t_two_sided_tail(2, df=100_000) == pytest.approx(normal_two_sided_tail(2), abs=1e-4)
    assert student_t_two_sided_tail(4, df=29) > normal_two_sided_tail(4)


def test_tail_table_adds_t_column():
    t = tail_table(pd.Series([0.0, 5.0]), thresholds=(2,), t_df=29)
    assert t.loc["|z| > 2", "expected_t29"] == pytest.approx(2 * student_t_two_sided_tail(2, 29))


def test_robust_band_matches_iid_band_for_iid_data():
    rng = np.random.default_rng(1)
    band = acf_robust_band(rng.normal(size=20_000), max_lag=3)
    assert band.to_numpy() == pytest.approx(1.96 / math.sqrt(20_000), rel=0.05)


def test_robust_band_is_wider_when_volatility_clusters():
    rng = np.random.default_rng(2)
    sigma = np.repeat(rng.choice([0.5, 3.0], size=400), 25)  # calm and wild blocks of 25 days
    x = rng.normal(size=sigma.size) * sigma
    assert (acf_robust_band(x, max_lag=5) > 1.96 / math.sqrt(x.size)).all()


def test_sqrt_t_null_ratio_values():
    median, p_above = sqrt_t_null_ratio(13)
    assert median == pytest.approx(0.972, abs=0.002) and p_above == pytest.approx(0.446, abs=0.005)
    assert sqrt_t_null_ratio(4)[0] == pytest.approx(0.89, abs=0.01)
    assert sqrt_t_null_ratio(100_000)[0] == pytest.approx(1.0, abs=1e-4)


def test_sqrt_t_null_ratio_matches_simulation():
    rng = np.random.default_rng(3)
    s = rng.normal(size=(200_000, 13)).std(axis=1, ddof=1)
    median, p_above = sqrt_t_null_ratio(13)
    assert np.median(s) == pytest.approx(median, abs=0.003)
    assert (s > 1).mean() == pytest.approx(p_above, abs=0.005)


def test_week_sunday_is_monday_to_sunday_utc():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-07 23:00", "2024-01-08 00:00"], tz="UTC")
    assert list(week_sunday(idx).strftime("%Y-%m-%d")) == ["2024-01-07", "2024-01-07", "2024-01-14"]


def test_week_sunday_rejects_non_utc():
    idx = pd.DatetimeIndex(["2024-01-01"], tz="America/Mexico_City")
    with pytest.raises(ValueError, match="UTC"):
        week_sunday(idx)


def test_daily_returns_of_a_week_compound_to_its_weekly_return():
    rng = np.random.default_rng(4)
    idx = days(70, start="2024-01-01")  # Monday: 10 full weeks
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 70)), index=idx)
    daily = step_returns(close, DAY)
    weekly = weekly_returns_from_daily(close)
    compounded = (1 + daily).groupby(week_sunday(daily.index)).prod() - 1
    assert compounded.loc[weekly.index.tz_localize(None)].to_numpy() == pytest.approx(weekly.to_numpy())


def test_hourly_returns_of_a_utc_day_compound_to_its_daily_return():
    rng = np.random.default_rng(5)
    hours = pd.date_range("2024-01-01", periods=24 * 5, freq="h", tz="UTC")
    hourly_close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, hours.size)), index=hours)
    daily_close = hourly_close[hourly_close.index.hour == 23]
    daily_close.index = daily_close.index.floor("D")       # daily close = 23:00 candle's close
    hourly = step_returns(hourly_close, pd.Timedelta(hours=1))
    daily = step_returns(daily_close, DAY)
    compounded = (1 + hourly).groupby(hourly.index.floor("D")).prod() - 1
    assert compounded.loc[daily.index].to_numpy() == pytest.approx(daily.to_numpy())
