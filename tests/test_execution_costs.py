"""Tests for src/execution_costs.py (hand-made books, no network).

Expected values are computed by hand in the comments.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.order_book import Market, snapshot_to_frame
from src.execution_costs import (STATUS_EXHAUSTED, STATUS_FILLED, STATUS_TRUNCATED, cost_shares,
                                 market_order_legs, round_trips, summarize_round_trips)

T0_MS = int(pd.Timestamp("2026-11-19 22:00:00", tz="UTC").timestamp() * 1000)
SPOT = Market("spot X/USDT", "binance", "X/USDT", "XUSDT", 3)     # asked for 3 levels per side
FEES = {"spot X/USDT": {"maker": 0.001, "taker": 0.001}}

# mid = 100. Asks: 101 x 1 (101 USDT), 102 x 2 (204 USDT), 104 x 10 (1040 USDT).
#            Bids:  99 x 1 ( 99 USDT),  98 x 2 (196 USDT)  -> only 2 levels: the whole book.
ASKS = [(101.0, 1.0), (102.0, 2.0), (104.0, 10.0)]
BIDS = [(99.0, 1.0), (98.0, 2.0)]


def book(bids=BIDS, asks=ASKS, request_ms=T0_MS, latency_ms=150, market=SPOT):
    raw = {"lastUpdateId": 1, "bids": [[str(p), str(q)] for p, q in bids],
           "asks": [[str(p), str(q)] for p, q in asks]}
    snap = {"raw": raw, "local_request_ms": request_ms,
            "local_receive_ms": request_ms + latency_ms}
    return snapshot_to_frame(snap, market, 0)


def leg(legs, action, size):
    row = legs[(legs["action"] == action) & (legs["size_usdt"] == size)]
    assert len(row) == 1
    return row.iloc[0]


def test_buy_inside_first_level():
    # 50 USDT at 101: base = 50/101; half spread = base * 1 USDT -> 1/101 of N = 99.0099 bps.
    legs = market_order_legs(book(), sizes=(50.0,))
    b = leg(legs, "buy", 50.0)
    assert b["status"] == STATUS_FILLED
    assert b["avg_price"] == pytest.approx(101.0)
    assert b["levels_used"] == 1
    assert b["impact_bps"] == pytest.approx(0.0)
    assert b["half_spread_bps"] == pytest.approx(1 / 101 * 1e4)
    assert b["mid"] == pytest.approx(100.0)


def test_buy_walks_two_levels():
    # 305 USDT = all of level 0 (101 USDT, 1 coin) + all of level 1 (204 USDT, 2 coins).
    # base = 3, avg = 305/3 = 101.667; half spread = 3 * 1 = 3 USDT; impact = 3 * 0.667 = 2 USDT.
    legs = market_order_legs(book(), sizes=(305.0,))
    b = leg(legs, "buy", 305.0)
    assert b["status"] == STATUS_FILLED
    assert b["base_filled"] == pytest.approx(3.0)
    assert b["avg_price"] == pytest.approx(305 / 3)
    assert b["levels_used"] == 2
    assert b["half_spread_usdt"] == pytest.approx(3.0)
    assert b["impact_usdt"] == pytest.approx(2.0)
    assert b["slippage_bps"] == pytest.approx(5 / 305 * 1e4)


def test_partial_last_level():
    # 200 USDT: 101 USDT at 101 (1 coin) + 99 USDT at 102 (99/102 coin).
    legs = market_order_legs(book(), sizes=(200.0,))
    b = leg(legs, "buy", 200.0)
    base = 1 + 99 / 102
    assert b["base_filled"] == pytest.approx(base)
    assert b["impact_usdt"] == pytest.approx(base * (200 / base - 101))


def test_sell_walks_bids():
    # 148 USDT: 99 at 99 (1 coin) + 49 at 98 (0.5 coin); base 1.5; half spread 1.5;
    # impact = 1.5 * (99 - 148/1.5) = 1.5 * 0.3333 = 0.5 USDT.
    legs = market_order_legs(book(), sizes=(148.0,))
    s = leg(legs, "sell", 148.0)
    assert s["status"] == STATUS_FILLED
    assert s["avg_price"] == pytest.approx(148 / 1.5)
    assert s["half_spread_usdt"] == pytest.approx(1.5)
    assert s["impact_usdt"] == pytest.approx(0.5)


def test_exhausted_vs_truncated():
    # 1000 USDT: the bids hold 295 USDT in 2 levels (< 3 asked): the whole book -> exhausted.
    # The asks hold 1345 USDT: fillable. 2000 USDT: the asks returned 3 levels = the limit asked,
    # so the book goes on beyond what we saw -> truncated. No cost in either case.
    legs = market_order_legs(book(), sizes=(1000.0, 2000.0))
    assert leg(legs, "sell", 1000.0)["status"] == STATUS_EXHAUSTED
    assert leg(legs, "buy", 1000.0)["status"] == STATUS_FILLED
    t = leg(legs, "buy", 2000.0)
    assert t["status"] == STATUS_TRUNCATED
    assert np.isnan(t["slippage_bps"]) and np.isnan(t["avg_price"])


def test_exact_fill_of_the_whole_side():
    # 295 USDT = exactly every bid: fillable (not exhausted).
    legs = market_order_legs(book(), sizes=(295.0,))
    assert leg(legs, "sell", 295.0)["status"] == STATUS_FILLED


def test_crossed_book_raises():
    with pytest.raises(ValueError):
        market_order_legs(book(bids=[(102.0, 1.0)]), sizes=(10.0,))


def test_receive_time_is_carried():
    legs = market_order_legs(book(), sizes=(10.0,))
    expected = pd.Timestamp(T0_MS + 150, unit="ms", tz="UTC")
    assert (legs["local_receive_utc"] == expected).all()


def test_round_trips_arithmetic():
    # Size 50: buy half spread 1/101 of N, sell half spread 1/99 of N, no impact; fee 10 bps.
    legs = market_order_legs(book(), sizes=(50.0,))
    rt = round_trips(legs, FEES).set_index(["combo", "direction"])
    hs_buy, hs_sell = 1 / 101 * 1e4, 1 / 99 * 1e4
    tt = rt.loc[("taker -> taker", "long")]
    assert tt["total_bps"] == pytest.approx(hs_buy + hs_sell + 20)
    assert tt["total_usdt"] == pytest.approx((hs_buy + hs_sell + 20) * 50 / 1e4)
    assert rt.loc[("maker -> maker", "long"), "total_bps"] == pytest.approx(20)
    # long: maker buy (fee only) + taker sell (half spread of the bids + fee)
    assert rt.loc[("maker -> taker", "long"), "total_bps"] == pytest.approx(hs_sell + 20)
    assert rt.loc[("maker -> taker", "short"), "total_bps"] == pytest.approx(hs_buy + 20)


def test_round_trip_unfillable_leg_gives_no_cost():
    # 1000 USDT: the sell side is exhausted. Taker -> taker fails; a long maker -> taker exits
    # with a market sell (fails); a short maker -> taker exits with a market buy (fillable).
    legs = market_order_legs(book(), sizes=(1000.0,))
    rt = round_trips(legs, FEES).set_index(["combo", "direction"])
    assert rt.loc[("taker -> taker", "long"), "status"] == STATUS_EXHAUSTED
    assert np.isnan(rt.loc[("taker -> taker", "long"), "total_bps"])
    assert rt.loc[("maker -> taker", "long"), "status"] == STATUS_EXHAUSTED
    assert rt.loc[("maker -> taker", "short"), "status"] == STATUS_FILLED
    assert rt.loc[("maker -> maker", "long"), "status"] == STATUS_FILLED


def test_summary_counts_and_shares():
    # Two snapshots: the second has a thinner bid side, so a 200 USDT sell is exhausted there.
    levels = pd.concat([book(), book(bids=[(99.0, 1.0)], request_ms=T0_MS + 10_000)])
    legs = market_order_legs(levels, sizes=(50.0, 200.0))
    rt = round_trips(legs, FEES)
    s = summarize_round_trips(rt)
    row = s.loc[("spot X/USDT", "taker -> taker", "long", 200.0)]
    assert row["n_snapshots"] == 2 and row["n_fillable"] == 1 and row["n_exhausted"] == 1
    small = s.loc[("spot X/USDT", "taker -> taker", "long", 50.0)]
    assert small["n_fillable"] == 2
    shares = cost_shares(s)
    r = shares.loc[("spot X/USDT", "taker -> taker", "long", 50.0)]
    assert r[["spread (%)", "impact (%)", "fee (%)"]].sum() == pytest.approx(100)
    assert r["impact (%)"] == pytest.approx(0)
