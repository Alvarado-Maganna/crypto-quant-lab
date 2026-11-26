"""Tests for src/regimes.py (small hand-made series, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.regimes import (ABOVE, BELOW, CORRECTION, DEEP_BEAR, HIGH, LOW, MID, NEAR_ATH,
                         drawdown_from_running_max, drawdown_label, episode_summary, episodes,
                         expanding_tercile_thresholds, flips, flips_per_year, ma_trend_band_label,
                         ma_trend_label, next_day_stats, rolling_tercile_thresholds,
                         simple_moving_average, tercile_label, truncation_mismatches)


def days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def s(values, start="2024-01-01"):
    return pd.Series(values, index=days(len(values), start), dtype=float)


def random_walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=days(n))


# --- trend ---

def test_sma_includes_today_and_waits_for_full_window():
    ma = simple_moving_average(s([1, 2, 3, 4]), window=3)
    assert np.isnan(ma.iloc[1])
    assert ma.iloc[2] == pytest.approx(2.0)
    assert ma.iloc[3] == pytest.approx(3.0)


def test_ma_trend_label_and_warmup():
    lab = ma_trend_label(s([10, 10, 10, 13, 7]), window=3)
    assert lab.iloc[:2].isna().all()
    assert lab.iloc[2] == BELOW        # close == MA is not "above"
    assert lab.iloc[3] == ABOVE        # 13 > 11
    assert lab.iloc[4] == BELOW        # 7 < 10


def test_band_label_needs_clear_cross_and_keeps_state_inside_band():
    # window 1 would make MA == close; use window 2 so the MA lags.
    close = s([100, 100, 101, 110, 109, 107, 105, 90])
    lab = ma_trend_band_label(close, window=2, band=0.02)
    ma = simple_moving_average(close, 2)
    # day 2: 101 vs MA 100.5 -> inside the band, no label yet
    assert pd.isna(lab.iloc[2])
    assert lab.iloc[3] == ABOVE        # 110 > 105.5 * 1.02
    assert close.iloc[4] < ma.iloc[4] * 1.02 and lab.iloc[4] == ABOVE   # inside band: keep
    assert lab.iloc[6] == ABOVE        # 105 vs MA 106: below MA but inside the band
    assert lab.iloc[7] == BELOW        # 90 < 97.5 * 0.98
    assert ma_trend_label(close, 2).iloc[6] == BELOW   # the plain rule already flipped


# --- volatility terciles ---

def test_expanding_terciles_use_only_values_up_to_t():
    x = s([1, 2, 3, 100, 0.5])
    th = expanding_tercile_thresholds(x, min_periods=3)
    assert th.iloc[:2].isna().all().all()
    assert th["q1"].iloc[2] == pytest.approx(np.quantile([1, 2, 3], 1 / 3))
    assert th["q2"].iloc[2] == pytest.approx(np.quantile([1, 2, 3], 2 / 3))
    assert th["q2"].iloc[3] == pytest.approx(np.quantile([1, 2, 3, 100], 2 / 3))


def test_rolling_terciles_drop_old_values():
    x = s([100, 1, 2, 3])
    th = rolling_tercile_thresholds(x, window=3)
    assert th["q2"].iloc[3] == pytest.approx(np.quantile([1, 2, 3], 2 / 3))


def test_rolling_terciles_warmup_is_nan():
    th = rolling_tercile_thresholds(s([1, 2, 3, 4]), window=3)
    assert th.iloc[:2].isna().all().all()
    assert th.iloc[2:].notna().all().all()


def test_tercile_ties_go_to_mid():
    x = s([3, 6])
    th = pd.DataFrame({"q1": [3.0, 3.0], "q2": [6.0, 6.0]}, index=x.index)
    assert tercile_label(x, th).tolist() == [MID, MID]


def test_tercile_label_and_nan():
    x = s([1, 5, 9, np.nan])
    th = pd.DataFrame({"q1": [3.0] * 4, "q2": [6.0] * 4}, index=x.index)
    assert tercile_label(x, th).iloc[:3].tolist() == [LOW, MID, HIGH]
    assert pd.isna(tercile_label(x, th).iloc[3])


# --- drawdown ---

def test_drawdown_from_running_max_and_bands():
    close = s([100, 120, 110, 90, 80, 130])
    dd = drawdown_from_running_max(close)
    assert dd.tolist() == pytest.approx([0, 0, 110 / 120 - 1, 90 / 120 - 1, 80 / 120 - 1, 0])
    lab = drawdown_label(dd)
    assert lab.tolist() == [NEAR_ATH, NEAR_ATH, NEAR_ATH, CORRECTION, DEEP_BEAR, NEAR_ATH]


def test_drawdown_band_edges():
    lab = drawdown_label(s([-0.10, -0.1000001, -0.30, -0.3000001]))
    assert lab.tolist() == [NEAR_ATH, CORRECTION, CORRECTION, DEEP_BEAR]


# --- episodes and flips ---

def test_episodes_runs_censoring_and_nan_break():
    lab = pd.Series([None, "a", "a", "b", None, "b", "a", "a"], index=days(8), dtype=object)
    eps = episodes(lab)
    assert eps["label"].tolist() == ["a", "b", "b", "a"]
    assert eps["n_days"].tolist() == [2, 1, 1, 2]
    assert eps["censored_first"].tolist() == [True, False, False, False]
    assert eps["censored_last"].tolist() == [False, False, False, True]


def test_episode_summary_uses_complete_episodes_only():
    lab = pd.Series(list("aaabbbbbbaaaaaab"), index=days(16), dtype=object)
    summ = episode_summary(episodes(lab), short_days=5)
    # 'a': censored first run (3 d) + one complete run (6 d); 'b': complete 6 d + censored last 1 d
    assert summ.loc["a", "n episodes"] == 2
    assert summ.loc["a", "n complete"] == 1
    assert summ.loc["a", "median days (complete)"] == 6
    assert summ.loc["b", "n censored (last)"] == 1
    assert summ.loc["b", "n short (< 5 d)"] == 1


def test_trailing_nan_does_not_end_the_last_run():
    lab = pd.Series(["a", "b", "b", None], index=days(4), dtype=object)
    assert episodes(lab)["censored_last"].tolist() == [False, True]


def test_next_day_stats_skips_label_without_next_return():
    lab = pd.Series(["a", "a", "b"], index=days(3), dtype=object)
    st = next_day_stats(lab, s([0.0, 0.01, 0.02]))
    assert list(st.index) == ["a"]   # 'b' is only on the last day: no next-day return


def test_flips_per_year():
    lab = pd.Series(["a", "b", "b", "a", None, "b"], index=days(6, "2024-12-29"), dtype=object)
    f = flips(lab)
    assert f.tolist() == [False, True, False, True, False, False]   # NaN never counts as a flip
    assert flips_per_year(lab).to_dict() == {2024: 1, 2025: 1}


# --- next-day stats ---

def test_next_day_stats_use_next_return_not_same_day():
    lab = pd.Series(["up", "up", "down", "down"], index=days(4), dtype=object)
    r = s([0.01, 0.02, -0.03, 0.04])
    st = next_day_stats(lab, r)
    # 'up' days are day 0 and 1 -> next returns are r1 = 0.02 and r2 = -0.03
    assert st.loc["up", "mean next-day (bps)"] == pytest.approx(1e4 * (0.02 - 0.03) / 2)
    assert st.loc["up", "mechanical (same-day) mean (bps)"] == pytest.approx(1e4 * 0.015)
    # 'down' days 2, 3 -> next returns r3 = 0.04 and NaN (no day after the last one)
    assert st.loc["down", "n days"] == 1


def test_next_day_stats_ex_worst_episode():
    lab = pd.Series(["a", "a", "b", "a", "a", "b"], index=days(6), dtype=object)
    r = s([0, 0.01, -0.20, 0.01, 0.02, 0.0])
    st = next_day_stats(lab, r)
    # 'a' episodes: days 0-1 (next r1, r2 = 0.01, -0.20) and days 3-4 (next 0.02, 0.0)
    assert st.loc["a", "worst episode start"] == days(6)[0].date()
    assert st.loc["a", "mean next-day ex-worst (bps)"] == pytest.approx(1e4 * 0.01)
    assert st.loc["a", "n days ex-worst"] == 2


# --- truncation (look-ahead) ---

@pytest.mark.parametrize("cutoff", ["2024-06-01", "2024-12-31", "2025-05-01"])
def test_every_rule_is_stable_under_truncation(cutoff):
    close = random_walk()
    vol = close.pct_change().rolling(30).std()
    rules = {
        "trend": lambda c: ma_trend_label(c, 200),
        "trend band": lambda c: ma_trend_band_label(c, 200, 0.02),
        "drawdown": lambda c: drawdown_label(drawdown_from_running_max(c)),
    }
    for name, fn in rules.items():
        assert truncation_mismatches(fn, close, cutoff) == 0, name
    vol_rules = {
        "expanding": lambda v: tercile_label(v, expanding_tercile_thresholds(v, 100)),
        "rolling": lambda v: tercile_label(v, rolling_tercile_thresholds(v, 200)),
    }
    for name, fn in vol_rules.items():
        assert truncation_mismatches(fn, vol, cutoff) == 0, name


def test_truncation_catches_a_full_sample_rule():
    vol = random_walk().pct_change().rolling(30).std()

    def look_ahead(v):
        q = v.quantile([1 / 3, 2 / 3]).to_numpy()
        th = pd.DataFrame({"q1": q[0], "q2": q[1]}, index=v.index)
        return tercile_label(v, th)

    assert truncation_mismatches(look_ahead, vol, "2024-06-01") > 0
