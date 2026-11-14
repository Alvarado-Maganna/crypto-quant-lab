"""Tests for src/data/ohlcv.py. They use a fake exchange, so they never touch the network."""

import pandas as pd
import pytest

from src.data.ohlcv import (
    align_closes,
    check_daily_candles,
    drop_incomplete_candles,
    fetch_daily_ohlcv,
    fetch_raw_ohlcv,
    find_missing_days,
    raw_to_frame,
)

DAY_MS = 24 * 60 * 60 * 1000
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)  # 2024-01-01 00:00 UTC in ms


def candle(day, price=100.0):
    """A raw candle `day` days after T0: [timestamp_ms, open, high, low, close, volume]."""
    return [T0 + day * DAY_MS, price, price + 1, price - 1, price, 10.0]


class FakeExchange:
    """Mimics ccxt's fetch_ohlcv: returns up to `limit` candles with timestamp >= since."""

    def __init__(self, candles):
        self.candles = candles
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append(since)
        return [c for c in self.candles if c[0] >= since][:limit]


def test_pagination_gets_every_candle_once():
    exchange = FakeExchange([candle(d) for d in range(25)])
    rows = fetch_raw_ohlcv(exchange, "BTC/USDT", since=T0, limit=10)
    assert [r[0] for r in rows] == [candle(d)[0] for d in range(25)]
    # pages start at T0, then last timestamp + 1 ms
    assert exchange.calls[1] == candle(9)[0] + 1


def test_timestamps_are_read_as_milliseconds_utc():
    df = raw_to_frame([candle(0)])
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_drops_todays_incomplete_candle():
    df = raw_to_frame([candle(0), candle(1), candle(2)])  # Jan 1, 2, 3
    now = pd.Timestamp("2024-01-03 15:30", tz="UTC")     # Jan 3 candle still forming
    kept = drop_incomplete_candles(df, now)
    assert list(kept.index.day) == [1, 2]


def test_keeps_candle_that_closed_exactly_now():
    df = raw_to_frame([candle(0), candle(1)])
    now = pd.Timestamp("2024-01-03 00:00", tz="UTC")     # Jan 2 candle closed at this instant
    assert len(drop_incomplete_candles(df, now)) == 2


def test_naive_now_is_rejected():
    df = raw_to_frame([candle(0)])
    with pytest.raises(ValueError):
        drop_incomplete_candles(df, pd.Timestamp("2024-01-03"))


def test_detects_duplicates():
    df = raw_to_frame([candle(0), candle(1), candle(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        check_daily_candles(df)


def test_detects_missing_days():
    df = raw_to_frame([candle(0), candle(1), candle(4)])  # Jan 3 and Jan 4 missing
    missing = find_missing_days(df)
    assert list(missing.day) == [3, 4]
    assert len(check_daily_candles(df)["missing_days"]) == 2


def test_rejects_non_positive_prices():
    df = raw_to_frame([candle(0), candle(1, price=0.0)])
    with pytest.raises(ValueError, match="<= 0"):
        check_daily_candles(df)


def test_fetch_daily_ohlcv_end_to_end():
    exchange = FakeExchange([candle(d) for d in range(5)])  # Jan 1..5
    now = pd.Timestamp("2024-01-05 08:00", tz="UTC")
    df, report = fetch_daily_ohlcv(exchange, "BTC/USDT", "2024-01-01", now_utc=now)
    assert report["candles"] == 4 and report["missing_days"] == []
    assert df.index.max() == pd.Timestamp("2024-01-04", tz="UTC")


def test_align_closes_keeps_only_common_dates_and_reports_dropped():
    a = raw_to_frame([candle(0, 10.0), candle(1, 11.0), candle(2, 12.0)])  # Jan 1, 2, 3
    b = raw_to_frame([candle(1, 20.0), candle(2, 21.0), candle(3, 22.0)])  # Jan 2, 3, 4
    closes, dropped = align_closes({"A": a, "B": b})
    assert list(closes.index.day) == [2, 3]
    assert list(closes.columns) == ["A", "B"]
    # each close stays on its own date (no shifting between the two series)
    assert closes.loc[pd.Timestamp("2024-01-02", tz="UTC")].tolist() == [11.0, 20.0]
    assert list(dropped["A"].day) == [1] and list(dropped["B"].day) == [4]


def test_align_closes_never_fills_a_missing_day():
    a = raw_to_frame([candle(0), candle(1), candle(2)])  # Jan 1, 2, 3
    b = raw_to_frame([candle(0), candle(2)])             # Jan 2 missing
    closes, dropped = align_closes({"A": a, "B": b})
    assert list(closes.index.day) == [1, 3]               # the day is dropped, not invented
    assert closes.notna().all().all()
    assert list(dropped["A"].day) == [2]
