"""Tests for src/data/order_book.py and src/order_book.py (fake exchange, no network).

Expected values are computed by hand in the comments.
"""

import ccxt
import numpy as np
import pandas as pd
import pytest

from src.data.order_book import (HourlyParquetStore, Market, collect, default_markets,
                                 depth_weight, fetch_with_retry, load_snapshots, round_weight,
                                 snapshot_to_frame, weight_pause_s)
from src.order_book import (consecutive_changes, pick_altcoin, quantile_summary,
                            rank_usdt_spot_pairs, snapshot_metrics)

T0_MS = int(pd.Timestamp("2026-11-19 21:59:50", tz="UTC").timestamp() * 1000)
SPOT = Market("spot X/USDT", "binance", "X/USDT", "XUSDT", 5000)
PERP = Market("perp X/USDT:USDT", "binanceusdm", "X/USDT:USDT", "XUSDT", 1000)


def raw_book(bids, asks, update_id=1, event_ms=None):
    raw = {"lastUpdateId": update_id,
           "bids": [[str(p), str(q)] for p, q in bids],
           "asks": [[str(p), str(q)] for p, q in asks]}
    if event_ms is not None:
        raw["E"], raw["T"] = event_ms, event_ms - 3
    return raw


# A book around mid 100: bids 99.95 / 99.5 / 98.0, asks 100.05 / 100.5 / 101.5.
BIDS = [(99.95, 2.0), (99.5, 1.0), (98.0, 10.0)]
ASKS = [(100.05, 1.0), (100.5, 3.0), (101.5, 4.0)]


def frame(bids=BIDS, asks=ASKS, market=SPOT, request_ms=T0_MS, round_id=0, event_ms=None):
    snap = {"raw": raw_book(bids, asks, event_ms=event_ms), "local_request_ms": request_ms,
            "local_receive_ms": request_ms + 40}
    return snapshot_to_frame(snap, market, round_id)


# --- weights and limits ----------------------------------------------------------------------

def test_depth_weight_steps():
    assert [depth_weight("binance", n) for n in (100, 101, 500, 501, 1000, 1001, 5000)] == \
        [5, 25, 25, 50, 50, 250, 250]
    assert [depth_weight("binanceusdm", n) for n in (5, 50, 100, 500, 1000)] == [2, 2, 5, 10, 20]
    with pytest.raises(ValueError):
        depth_weight("binanceusdm", 1500)   # rejected by the API (-1130)
    with pytest.raises(ValueError):
        depth_weight("binance", 5001)       # undocumented: we refuse it


def test_round_weight_of_default_markets():
    # 3 spot markets x 250 + 1 perp x 20
    assert round_weight(default_markets()) == {"binance": 750, "binanceusdm": 20}


def test_weight_pause():
    assert weight_pause_s(4800, 6000, 15_000) == 0.0          # exactly 80%: no pause
    assert weight_pause_s(None, 6000, 15_000) == 0.0
    # 4801 > 80%; 15 s into the minute -> wait 45 s to the boundary + 1 s margin
    assert weight_pause_s(4801, 6000, 60_000 * 7 + 15_000) == pytest.approx(46.0)


# --- raw snapshot -> rows --------------------------------------------------------------------

def test_snapshot_to_frame_spot_has_no_exchange_time():
    df = frame()
    assert len(df) == 6 and list(df["side"]) == ["bid"] * 3 + ["ask"] * 3
    assert list(df["level"]) == [0, 1, 2, 0, 1, 2]
    assert df["price"].iloc[0] == 99.95 and df["amount"].iloc[5] == 4.0
    assert df["exchange_event_utc"].isna().all()
    assert str(df["local_request_utc"].dtype) == "datetime64[ms, UTC]"
    assert df["local_receive_utc"].iloc[0] - df["local_request_utc"].iloc[0] == \
        pd.Timedelta(milliseconds=40)


def test_snapshot_to_frame_perp_keeps_exchange_times():
    df = frame(market=PERP, event_ms=T0_MS + 25)
    assert df["exchange_event_utc"].iloc[0] == pd.to_datetime(T0_MS + 25, unit="ms", utc=True)
    assert df["exchange_transaction_utc"].iloc[0] == pd.to_datetime(T0_MS + 22, unit="ms", utc=True)


# --- storage ---------------------------------------------------------------------------------

def test_store_append_compact_and_load(tmp_path):
    store = HourlyParquetStore(tmp_path)
    a = frame(request_ms=T0_MS)                 # 21:59:50 -> hour 21
    b = frame(request_ms=T0_MS + 20_000)        # 22:00:10 -> hour 22
    store.append(pd.concat([a, b]))
    assert len(list((tmp_path / "parts").glob("*/*.parquet"))) == 2
    assert len(load_snapshots(tmp_path)) == 12  # readable before compaction
    store.compact_finished_hours(pd.Timestamp("2026-11-19 22:30", tz="UTC"))
    assert (tmp_path / "20261119_21.parquet").exists()          # hour 21 is over
    assert not (tmp_path / "20261119_22.parquet").exists()      # hour 22 is still running
    store.append(a)                              # the same rows again (e.g. after a restart)
    store.compact("20261119_21")
    loaded = load_snapshots(tmp_path)
    assert len(loaded) == 12                     # duplicates dropped
    assert not list(tmp_path.glob("**/*.tmp"))   # no temporary files left


# --- retries and the collection loop ---------------------------------------------------------

def test_fetch_with_retry_retries_network_errors_only():
    calls, pauses = [], []

    def flaky(params):
        calls.append(params)
        if len(calls) < 3:
            raise ccxt.RequestTimeout("timeout")
        return "ok"

    assert fetch_with_retry(flaky, {}, sleep=pauses.append, log=lambda m: None) == "ok"
    assert pauses == [2.0, 4.0]

    def bad(params):
        raise ccxt.BadRequest("invalid limit")

    with pytest.raises(ccxt.BadRequest):
        fetch_with_retry(bad, {}, sleep=pauses.append, log=lambda m: None)


def test_rate_limit_waits_at_least_60_s():
    pauses, calls = [], []

    def limited(params):
        calls.append(1)
        if len(calls) == 1:
            raise ccxt.RateLimitExceeded("429 Too Many Requests")
        return "ok"

    fetch_with_retry(limited, {}, sleep=pauses.append, log=lambda m: None)
    assert pauses == [60.0]


class FakeClock:
    def __init__(self, start_ms):
        self.t = start_ms

    def __call__(self):
        self.t += 5            # each read of the clock moves 5 ms
        return self.t

    def sleep(self, seconds):
        self.t += int(seconds * 1000)


class FakeExchange:
    """Answers depth calls; the spot market fails for good in round 1 (bad request)."""

    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call
        self.last_response_headers = {"x-mbx-used-weight-1m": "100"}

    def _answer(self, params):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ccxt.BadRequest("boom")
        return raw_book(BIDS, ASKS, update_id=self.calls, event_ms=T0_MS)

    publicGetDepth = fapiPublicGetDepth = _answer


def test_collect_skips_failed_market_and_flushes(tmp_path):
    clock, logs = FakeClock(T0_MS), []
    exchanges = {"binance": FakeExchange(fail_on_call=2), "binanceusdm": FakeExchange()}
    summary = collect(exchanges, [SPOT, PERP], HourlyParquetStore(tmp_path), interval_s=10,
                      n_rounds=3, flush_every=2, clock=clock, sleep=clock.sleep, log=logs.append)
    assert summary["rounds"] == 3 and summary["snapshots"] == 5 and summary["failures"] == 1
    assert any("round 1: spot X/USDT failed" in m for m in logs)
    levels = load_snapshots(tmp_path)
    assert levels.groupby("market")["round_id"].unique().map(list).to_dict() == \
        {"perp X/USDT:USDT": [0, 1, 2], "spot X/USDT": [0, 2]}
    # round k never starts before its slot: start + k * 10 s (start = first clock read)
    starts = levels.groupby("round_id")["local_request_utc"].min()
    slots = pd.to_datetime([T0_MS + 5 + k * 10_000 for k in range(3)], unit="ms", utc=True)
    assert all(start >= slot for start, slot in zip(starts, slots))


def test_collect_needs_a_stop_rule(tmp_path):
    with pytest.raises(ValueError):
        collect({}, [], HourlyParquetStore(tmp_path))


# --- metrics ---------------------------------------------------------------------------------

def test_spread_and_depth_by_hand():
    m = snapshot_metrics(frame()).iloc[0]
    # mid = (99.95 + 100.05) / 2 = 100; spread = 0.10 USDT = 0.10 / 100 * 1e4 = 10 bps
    assert m["mid"] == pytest.approx(100.0)
    assert m["spread_usdt"] == pytest.approx(0.10)
    assert m["spread_bps"] == pytest.approx(10.0)
    # ±0.1% band = [99.9, 100.1]: only the best level on each side
    assert m["depth_0.1pct_bid_base"] == pytest.approx(2.0)
    assert m["depth_0.1pct_bid_usdt"] == pytest.approx(199.9)
    assert m["depth_0.1pct_ask_base"] == pytest.approx(1.0)
    # ±1% band = [99, 101]: bids 2 + 1 (98 is outside), asks 1 + 3 (101.5 is outside)
    assert m["depth_1pct_bid_base"] == pytest.approx(3.0)
    assert m["depth_1pct_bid_usdt"] == pytest.approx(99.95 * 2 + 99.5)
    assert m["depth_1pct_ask_usdt"] == pytest.approx(100.05 + 100.5 * 3)
    # deepest levels at 2% (bid) and 1.5% (ask) from the mid: both bands are complete
    assert m["reach_bid_pct"] == pytest.approx(2.0) and m["reach_ask_pct"] == pytest.approx(1.5)
    assert m["complete_1pct_bid"] and m["complete_1pct_ask"]
    # total notional: far below 1M USDT -> flagged
    assert not m["covers_1m_bid"]


def test_truncated_band_is_flagged_not_filled():
    # the ask side stops at 100.5 (0.5% from the mid): the ±1% ask depth is only a lower bound
    m = snapshot_metrics(frame(asks=ASKS[:2])).iloc[0]
    assert not m["complete_1pct_ask"] and m["complete_1pct_bid"]
    assert m["depth_1pct_ask_base"] == pytest.approx(4.0)  # what we saw, nothing added
    assert m["complete_0.1pct_ask"]


def test_band_edge_is_included():
    # a level exactly 0.1% away from mid 100 (100.1) counts inside the ±0.1% band
    m = snapshot_metrics(frame(bids=[(99.9, 1.0)], asks=[(100.1, 1.0)])).iloc[0]
    assert m["depth_0.1pct_ask_base"] == pytest.approx(1.0)
    assert m["depth_0.1pct_bid_base"] == pytest.approx(1.0)


def test_crossed_book_raises():
    with pytest.raises(ValueError, match="crossed"):
        snapshot_metrics(frame(bids=[(100.1, 1.0)], asks=[(100.0, 1.0)]))


def test_consecutive_changes_and_summary():
    levels = pd.concat([frame(request_ms=T0_MS),
                        frame(asks=[(100.15, 1.0)] + ASKS[1:], request_ms=T0_MS + 10_000)])
    m = snapshot_metrics(levels)
    ch = consecutive_changes(m, "spread_bps")
    # spread 0.10 -> 0.20 USDT; mid 100.0 -> 100.05: 10 bps -> 0.2 / 100.05 * 1e4 = 19.99 bps
    assert len(ch) == 1 and ch["gap_s"].iloc[0] == 10.0
    assert ch["change"].iloc[0] == pytest.approx(0.2 / 100.05 * 1e4 - 10)
    s = quantile_summary(pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert s["p50"] == 2.0 and s["n"] == 3


# --- altcoin choice --------------------------------------------------------------------------

def test_pick_altcoin_skips_stablecoins_and_non_crypto():
    names = ["BTC", "USDX", "NVDAB", "ALT1", "ALT2"]
    tickers = {f"{n}/USDT": {"quoteVolume": v, "last": last, "high": last * 1.001 if n == "USDX"
                             else last * 1.1, "low": last}
               for n, v, last in zip(names, [500, 400, 300, 200, 100], [60000, 1.0, 140, 2, 3])}
    markets = {f"{n}/USDT": {"spot": True, "active": True, "quote": "USDT", "base": n}
               for n in names}
    markets["ETH/BTC"] = {"spot": True, "active": True, "quote": "BTC", "base": "ETH"}
    tickers["ETH/BTC"] = {"quoteVolume": 999, "last": 0.03, "high": 0.03, "low": 0.03}
    ranking = rank_usdt_spot_pairs(tickers, markets)
    assert list(ranking["symbol"]) == [f"{n}/USDT" for n in names]  # ETH/BTC is not USDT
    symbol, classified = pick_altcoin(ranking, start_rank=2)
    assert symbol == "ALT1/USDT"   # rank 2 is a stablecoin, rank 3 a tokenized stock
    assert list(classified["excluded_as"])[:3] == ["", "stablecoin", "tokenized equity / ETF"]
