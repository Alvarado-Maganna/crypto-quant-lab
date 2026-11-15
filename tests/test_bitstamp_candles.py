"""Tests for src/data/bitstamp.py and src/data/candles.py. No network access."""

import pandas as pd
import pytest

from src.data.bitstamp import bitstamp_to_frame, fetch_bitstamp_ohlc, plan_chunks
from src.data.candles import (
    drop_exact_duplicates,
    mark_exchange_filled,
    missing_runs,
    validate_candles,
)
from src.data.ohlcv import fetch_raw_ohlcv

T0 = int(pd.Timestamp("2023-03-09", tz="UTC").timestamp())  # Unix seconds


def row(minute, close=1.0, volume=5.0, high=None, low=None):
    """A raw Bitstamp-style candle (numbers as strings, like the real API)."""
    high = close if high is None else high
    low = close if low is None else low
    return {"timestamp": str(T0 + 60 * minute), "open": str(close), "high": str(high),
            "low": str(low), "close": str(close), "volume": str(volume)}


# --- Bitstamp pagination -------------------------------------------------------------

def test_plan_chunks_limit_matches_chunk_size():
    chunks = plan_chunks(T0, T0 + 60 * 2499, step_s=60, max_limit=1000)  # 2500 candles
    assert [c[2] for c in chunks] == [1000, 1000, 500]
    assert chunks[1][0] == chunks[0][1] + 60        # no overlap, no gap
    assert chunks[-1][1] == T0 + 60 * 2499


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class QuirkySession:
    """Mimics the real quirk: returns `limit` candles ENDING at `end`, ignoring `start`."""

    def __init__(self, n_available):
        self.all = [row(m) for m in range(n_available)]
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        upto = [r for r in self.all if int(r["timestamp"]) <= params["end"]]
        return FakeResponse({"data": {"ohlc": upto[-params["limit"]:]}})


def test_fetch_has_no_overlap_despite_api_quirk():
    session = QuirkySession(n_available=3000)
    start = pd.Timestamp(T0, unit="s", tz="UTC")
    end = start + pd.Timedelta(minutes=2499)
    rows = fetch_bitstamp_ohlc("usdcusd", start, end, session=session, pause_s=0)
    stamps = [int(r["timestamp"]) for r in rows]
    assert len(stamps) == 2500 and len(set(stamps)) == 2500
    assert stamps == sorted(stamps)
    assert session.calls[0]["start"] == T0          # seconds, not milliseconds


def test_timestamps_are_read_as_seconds_utc():
    df = bitstamp_to_frame([row(0)])
    assert df.index[0] == pd.Timestamp("2023-03-09 00:00", tz="UTC")


# --- Duplicates ----------------------------------------------------------------------

def test_identical_duplicates_are_dropped_and_sorted():
    df = bitstamp_to_frame([row(2), row(0), row(1), row(2)])
    clean, n_dropped = drop_exact_duplicates(df)
    assert n_dropped == 1
    assert list(clean.index.minute) == [0, 1, 2]


def test_conflicting_duplicates_raise():
    df = bitstamp_to_frame([row(0), row(1, close=1.0), row(1, close=0.9)])
    with pytest.raises(ValueError, match="Conflicting"):
        drop_exact_duplicates(df)


# --- Validation ----------------------------------------------------------------------

def test_validate_reports_missing_without_filling():
    df = bitstamp_to_frame([row(0), row(1), row(4)])
    report = validate_candles(df, "2023-03-09 00:00", "2023-03-09 00:04")
    assert report["expected"] == 5 and report["candles"] == 3 and report["missing"] == 2
    assert len(df) == 3                              # input untouched, nothing invented
    runs = missing_runs(report["missing_index"])
    assert runs["minutes"].tolist() == [2]


def test_validate_rejects_duplicates_and_unsorted():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_candles(bitstamp_to_frame([row(0), row(0)]), "2023-03-09", "2023-03-09 00:01")
    with pytest.raises(ValueError, match="sorted"):
        validate_candles(bitstamp_to_frame([row(1), row(0)]), "2023-03-09", "2023-03-09 00:01")


# --- Exchange-filled candles ---------------------------------------------------------

def test_mark_exchange_filled():
    df = bitstamp_to_frame([row(0, close=0.98), row(1, close=0.98, volume=0),
                            row(2, close=0.97, volume=0)])
    out = mark_exchange_filled(df)
    assert out["filled_by_exchange"].tolist() == [False, True, True]
    # minute 1 repeats the previous close; minute 2 does not (so it is flagged, not assumed)
    assert out["flat_at_prev_close"].tolist() == [False, True, False]


# --- Generalized ccxt pagination (used for Binance) -----------------------------------

class FakeExchange:
    def __init__(self, stamps):
        self.stamps = stamps

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [[t, 1, 1, 1, 1, 1] for t in self.stamps if t >= since][:limit]


def test_fetch_raw_ohlcv_stops_at_until():
    exchange = FakeExchange(list(range(0, 100_000, 1000)))   # 100 candles, 1 s apart in ms
    rows = fetch_raw_ohlcv(exchange, "USDC/USDT", since=0, limit=7, until=20_000)
    assert [r[0] for r in rows] == list(range(0, 21_000, 1000))
