"""Tests for src/data/funding.py and src/funding.py (fake API, no network).

Expected values are computed by hand in the comments.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.funding import (fetch_funding_window, fetch_raw_funding, funding_cache_path,
                              funding_to_frame, load_or_fetch_funding)
from src.funding import (annualize_compound, annualize_simple, binance_funding_from_premium,
                         funding_payment, infer_intervals, is_baseline, sign_shares,
                         total_funding_paid)
from src.liquidation import simulate_isolated_longs

H8 = 8 * 3_600_000
T0 = pd.Timestamp("2025-01-01", tz="UTC")
T0_MS = int(T0.timestamp() * 1000)


def row(i, rate="0.00010000", mark="100000.0", rate_type="Regular", offset_ms=0, symbol="BTCUSDT"):
    return {"symbol": symbol, "fundingTime": T0_MS + i * H8 + offset_ms, "fundingRate": rate,
            "markPrice": mark, "rateType": rate_type}


class FakeFundingAPI:
    """Like /fapi/v1/fundingRate: oldest first, startTime and endTime inclusive, capped pages."""

    def __init__(self, rows, max_page=1000):
        self.rows, self.max_page = rows, max_page
        self.calls = []

    def __call__(self, params):
        self.calls.append(dict(params))
        lo, hi = params["startTime"], params["endTime"]
        hits = [r for r in self.rows if lo <= r["fundingTime"] <= hi]
        return hits[: min(params["limit"], self.max_page)]


# ---------------------------------------------------------------- pagination


def test_pagination_advances_start_by_last_time_plus_one_ms():
    rows = [row(i, offset_ms=i % 8) for i in range(2_500)]   # some rows a few ms late
    api, pauses = FakeFundingAPI(rows), []
    end_ms = T0_MS + 3_000 * H8
    out = fetch_raw_funding(api, "BTCUSDT", T0_MS, end_ms, sleep=pauses.append)
    assert len(out) == 2_500
    # 2,500 rows / 1,000 per page = 3 pages; the 3rd is short (500 < 1000), so we stop there.
    assert len(api.calls) == 3
    assert api.calls[1]["startTime"] == rows[999]["fundingTime"] + 1
    assert api.calls[2]["startTime"] == rows[1999]["fundingTime"] + 1
    assert all(c["endTime"] == end_ms - 1 and c["limit"] == 1000 for c in api.calls)
    assert pauses == [0.7, 0.7]                               # a pause between pages, not before


def test_pagination_exact_multiple_of_limit_needs_one_empty_page():
    rows = [row(i) for i in range(2_000)]
    api = FakeFundingAPI(rows)
    out = fetch_raw_funding(api, "BTCUSDT", T0_MS, T0_MS + 5_000 * H8, sleep=lambda s: None)
    assert len(out) == 2_000 and len(api.calls) == 3          # 1000 + 1000 + empty


def test_pagination_works_when_api_caps_below_limit():
    api = FakeFundingAPI([row(i) for i in range(25)], max_page=10)
    out = fetch_raw_funding(api, "BTCUSDT", T0_MS, T0_MS + 100 * H8, limit=10, sleep=lambda s: None)
    assert [r["fundingTime"] for r in out] == [row(i)["fundingTime"] for i in range(25)]


def test_start_zero_is_refused():
    with pytest.raises(ValueError, match="startTime=0"):
        fetch_raw_funding(FakeFundingAPI([]), "BTCUSDT", 0, T0_MS)


def test_stuck_api_does_not_loop_forever():
    stuck = lambda params: [row(0) for _ in range(params["limit"])]  # always the same rows
    out = fetch_raw_funding(stuck, "BTCUSDT", T0_MS + 1, T0_MS + 10 * H8, sleep=lambda s: None)
    assert len(out) == 1000


# ---------------------------------------------------------------- parsing and window


def test_frame_parses_ms_utc_and_empty_mark():
    df = funding_to_frame([row(0, mark=""), row(1, offset_ms=7)])
    assert str(df.index.tz) == "UTC" and df.index[0] == T0
    assert df.index[1] == T0 + pd.Timedelta(hours=8, milliseconds=7)
    assert np.isnan(df["mark_price"].iloc[0]) and df["mark_price"].iloc[1] == 100_000.0
    assert df["funding_rate_str"].iloc[0] == "0.00010000"


def test_missing_rate_type_raises_instead_of_guessing():
    rows = [{k: v for k, v in row(i).items() if k != "rateType"} for i in range(2)]
    with pytest.raises(ValueError, match="rateType"):
        funding_to_frame(rows)


def test_window_end_excluded_special_dropped_duplicates_dropped():
    rows = [row(0), row(1), row(1), row(2, rate_type="Special"), row(3), row(4)]
    df, info = fetch_funding_window(FakeFundingAPI(rows), "BTCUSDT", T0,
                                    T0 + pd.Timedelta(hours=32), sleep=lambda s: None)
    # window [T0, T0+32h) holds rows 0..3; row 4 (at +32h) is excluded by endTime = end - 1 ms
    assert list(df.index) == [T0 + pd.Timedelta(hours=8 * i) for i in (0, 1, 3)]
    assert info["duplicates_dropped"] == 1 and info["special_dropped"] == 1
    assert info["start_ms"] == T0_MS and info["end_ms"] == T0_MS + 4 * H8


def test_conflicting_duplicates_raise():
    rows = [row(0), row(1), row(1, rate="0.00020000")]
    with pytest.raises(ValueError, match="Conflicting"):
        fetch_funding_window(FakeFundingAPI(rows), "BTCUSDT", T0, T0 + pd.Timedelta(days=1))


def test_cache_is_reused_and_checks_the_window(tmp_path):
    api = FakeFundingAPI([row(i) for i in range(6)])
    path = funding_cache_path(tmp_path, "binanceusdm", "BTCUSDT", "test")
    assert path.name == "binanceusdm_BTCUSDT_test_funding.parquet"
    start, end = T0, T0 + pd.Timedelta(days=2)
    fetch = lambda: fetch_funding_window(api, "BTCUSDT", start, end, sleep=lambda s: None)
    df1, _ = load_or_fetch_funding(path, fetch, start, end)
    calls = len(api.calls)
    df2, _ = load_or_fetch_funding(path, fetch, start, end)
    assert len(api.calls) == calls and len(df2) == len(df1) == 6
    with pytest.raises(ValueError, match="not the requested"):
        load_or_fetch_funding(path, fetch, start, end + pd.Timedelta(hours=8))


# ---------------------------------------------------------------- intervals


def test_intervals_keep_raw_ms_and_round_only_for_classification():
    times = [T0_MS, T0_MS + H8 + 7, T0_MS + 2 * H8 + 3, T0_MS + 2 * H8 + 4 * 3_600_000]
    iv = infer_intervals(times)
    assert np.isnan(iv["interval_ms"].iloc[0])
    assert list(iv["interval_ms"].iloc[1:]) == [28_800_007, 28_799_996, 14_399_997]
    assert list(iv["interval_hours_rounded"].iloc[1:]) == [8, 8, 4]
    assert iv["interval_hours"].iloc[1] == pytest.approx(8 + 7 / 3_600_000)


def test_intervals_must_increase():
    with pytest.raises(ValueError):
        infer_intervals([T0_MS, T0_MS])


# ---------------------------------------------------------------- annualization


@pytest.mark.parametrize("rate, hours, expected", [
    (0.0001, 8, 0.1095),    # 0.0001 * 8760/8 = 0.0001 * 1095
    (0.0003, 8, 0.3285),    # 0.0003 * 1095
    (0.0005, 8, 0.5475),    # 0.0005 * 1095
    (0.0001, 4, 0.2190),    # 0.0001 * 2190: same rate, twice as many payments
    (-0.0001, 8, -0.1095),
])
def test_annualize_simple(rate, hours, expected):
    assert annualize_simple(rate, hours) == pytest.approx(expected)


def test_annualize_compound_by_hand():
    # (1.0001)^1095 - 1: ln(1.0001) = 0.000099995, x1095 = 0.1094945,
    # e^0.1094945 = e^0.1 * e^0.0094945 = 1.1051709 * 1.0095397 = 1.115714 -> 0.115714
    assert annualize_compound(0.0001, 8) == pytest.approx(0.115714, abs=1e-6)
    # (1.0005)^1095 - 1: ln(1.0005) = 0.00049988, x1095 = 0.5473632,
    # e^0.5473632 = e^0.5 * e^0.0473632 = 1.6487213 * 1.0485027 = 1.728689 -> 0.728689
    assert annualize_compound(0.0005, 8) == pytest.approx(0.728689, abs=1e-6)
    # compound >= simple for positive rates
    assert annualize_compound(0.0003, 8) > annualize_simple(0.0003, 8)


def test_annualize_rejects_bad_interval():
    with pytest.raises(ValueError):
        annualize_simple(0.0001, 0)


# ---------------------------------------------------------------- payments


def test_payment_signs_and_the_m7_exercise():
    assert funding_payment("long", 10_000, 0.0005) == pytest.approx(5.0)    # long pays
    assert funding_payment("short", 10_000, 0.0005) == pytest.approx(-5.0)  # short receives
    assert funding_payment("long", 10_000, -0.0001) == pytest.approx(-1.0)  # long receives
    # one week = 7 days * 3 payments = 21 payments of 5 USDT = 105 USDT
    assert 21 * funding_payment("long", 10_000, 0.0005) == pytest.approx(105.0)


def test_total_funding_uses_the_mark_of_each_payment():
    # qty 0.1 BTC: 0.1*100,000*0.0001 = 1.0 paid; 0.1*90,000*(-0.0002) = -1.8 (received)
    assert total_funding_paid("long", 0.1, [100_000, 90_000], [0.0001, -0.0002]) \
        == pytest.approx(-0.8)
    with pytest.raises(ValueError):
        total_funding_paid("long", 0.1, [np.nan], [0.0001])


def test_payment_rejects_bad_inputs():
    with pytest.raises(ValueError):
        funding_payment("both", 1_000, 0.0001)
    with pytest.raises(ValueError):
        funding_payment("long", -1_000, 0.0001)


# ---------------------------------------------------------------- classification


def test_is_baseline_exact_and_with_tolerance():
    rates = ["0.00010000", "0.00010001", "0.00009999", "0.00010100"]
    assert list(is_baseline(rates, 0.0)) == [True, False, False, False]
    assert list(is_baseline(rates, 1e-8)) == [True, True, True, False]    # +-1 step of 1e-8
    assert list(is_baseline(rates, 1e-6)) == [True, True, True, True]     # 0.0000010 >= 0.000001


def test_sign_shares_counts_zero_separately():
    rates = ["0.00010000", "0.00000000", "-0.00010000", "0.00030000"]
    out = sign_shares(rates, tolerances=(0.0,)).loc[0.0]
    assert (out["positive (%)"], out["negative (%)"], out["zero (%)"]) == (50.0, 25.0, 25.0)
    assert out["= 0.01% (count)"] == 1


@pytest.mark.parametrize("premium, expected", [
    (0.0, 0.0001),      # 0 + clamp(0.0001, +-0.0005) = 0.0001
    (0.0006, 0.0001),   # upper edge of the dead zone: 0.0006 + (-0.0005)
    (0.0007, 0.0002),   # outside: 0.0007 - 0.0005
    (-0.0004, 0.0001),  # lower edge: -0.0004 + 0.0005
    (-0.0005, 0.0),     # -0.0005 + 0.0005
    (-0.0010, -0.0005),
])
def test_binance_formula_dead_zone(premium, expected):
    assert binance_funding_from_premium(premium) == pytest.approx(expected)


def test_premium_index_candles_get_their_own_cache_file(tmp_path):
    from src.data.ohlcv import cache_path
    path = cache_path(tmp_path, "binanceusdm", "BTC/USDT:USDT", "1m", "oct9-10", price="premiumIndex")
    assert path.name == "binanceusdm_BTCUSDT-USDT_1m_oct9-10_premiumIndex.parquet"


# ---------------------------------------------------------------- liquidation with funding


def test_funding_out_of_margin_moves_lp_up():
    entry = pd.Series([100_000.0], index=[T0])
    low = pd.Series([100_000.0, 95_000.0], index=[T0, T0 + pd.Timedelta(minutes=1)])
    base = simulate_isolated_longs(entry, {"mark": low}, [10], 1_000, 0.004)
    paid = simulate_isolated_longs(entry, {"mark": low}, [10], 1_000, 0.004, funding_per_qty=100)
    # qty = 0.1; base LP = (1000 - 10000) / (0.0004 - 0.1) = 90,361.45 (notebook 04, case 2)
    assert base["lp"].iloc[0] == pytest.approx(90_361.45, abs=0.01)
    # funding = 0.1 * 100 = 10 USDT -> margin 990: LP = (990 - 10000) / (-0.0996) = 90,461.85
    assert paid["funding_paid"].iloc[0] == pytest.approx(10.0)
    assert paid["lp"].iloc[0] == pytest.approx(90_461.85, abs=0.01)
