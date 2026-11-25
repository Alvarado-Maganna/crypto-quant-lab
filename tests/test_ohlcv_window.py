"""Tests for the intraday part of src/data/ohlcv.py (fake exchanges, no network)."""

import ccxt
import pandas as pd
import pytest

from src.data.ohlcv import fetch_ohlcv_window, fetch_page_with_retry

MIN_MS = 60_000
T0 = pd.Timestamp("2025-10-10 18:00", tz="UTC")
T0_MS = int(T0.timestamp() * 1000)


def minute(i, price=100.0, volume=1.0):
    return [T0_MS + i * MIN_MS, price, price + 1, price - 1, price, volume]


class FakeMinuteExchange:
    """Like ccxt: oldest first, ``since`` inclusive, pages silently capped at ``max_page``."""

    def __init__(self, candles, max_page=300, overlap=0):
        self.candles, self.max_page, self.overlap = candles, max_page, overlap
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((since, limit))
        start = since - self.overlap * MIN_MS  # a buggy API that repeats some old candles
        return [c for c in self.candles if c[0] >= start][: min(limit, self.max_page)]


def test_window_is_start_inclusive_end_exclusive_and_paginates():
    ex = FakeMinuteExchange([minute(i) for i in range(1000)], max_page=300)
    df, info = fetch_ohlcv_window("okx", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=720),
                                  now_utc="2026-01-01", exchange=ex)
    assert len(df) == 720
    assert df.index[0] == T0 and df.index[-1] == T0 + pd.Timedelta(minutes=719)
    assert ex.calls[0] == (T0_MS, 300)          # measured OKX max is used as the page size
    assert info["duplicates_dropped"] == 0


def test_timestamps_are_ms_and_utc():
    ex = FakeMinuteExchange([minute(i) for i in range(5)])
    df, _ = fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=5),
                               now_utc="2026-01-01", exchange=ex)
    assert str(df.index.tz) == "UTC" and df.index[0].year == 2025


def test_identical_duplicates_from_overlapping_pages_are_dropped():
    ex = FakeMinuteExchange([minute(i) for i in range(50)], max_page=10, overlap=2)
    df, info = fetch_ohlcv_window("bybit", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=50),
                                  now_utc="2026-01-01", exchange=ex, limit=10)
    assert len(df) == 50 and df.index.is_unique
    assert info["duplicates_dropped"] > 0


def test_conflicting_duplicates_raise():
    candles = [minute(i) for i in range(5)] + [minute(2, price=999.0)]
    ex = FakeMinuteExchange(sorted(candles, key=lambda c: c[0]))
    with pytest.raises(ValueError, match="Conflicting"):
        fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=5),
                           now_utc="2026-01-01", exchange=ex)


def test_forming_minute_is_dropped():
    ex = FakeMinuteExchange([minute(i) for i in range(10)])
    now = T0 + pd.Timedelta(minutes=9, seconds=30)  # minute 9 is still forming
    df, info = fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=10),
                                  now_utc=now, exchange=ex)
    assert len(df) == 9 and info["incomplete_dropped"] == 1


def test_gaps_are_kept_as_gaps_not_filled():
    ex = FakeMinuteExchange([minute(i) for i in range(10) if i not in (3, 4)])
    df, _ = fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=10),
                               now_utc="2026-01-01", exchange=ex)
    assert len(df) == 8


class Flaky:
    def __init__(self, errors):
        self.errors = list(errors)

    def fetch_ohlcv(self, *args, **kwargs):
        if self.errors:
            raise self.errors.pop(0)
        return [minute(0)]


def test_rate_limit_is_retried_with_backoff():
    pauses = []
    ex = Flaky([ccxt.RateLimitExceeded("429 Too Many Requests")] * 3)
    rows = fetch_page_with_retry(ex, "BTC/USDT", "1m", T0_MS, 10, sleep=pauses.append)
    assert rows == [minute(0)]
    assert pauses == [1.0, 2.0, 4.0]


def test_region_block_is_not_retried():
    pauses = []
    ex = Flaky([ccxt.ExchangeNotAvailable("binance GET ... 451 Unavailable For Legal Reasons")])
    with pytest.raises(PermissionError):
        fetch_page_with_retry(ex, "BTC/USDT", "1m", T0_MS, 10, sleep=pauses.append)
    assert pauses == []


def test_gives_up_after_max_retries():
    ex = Flaky([ccxt.RequestTimeout("timeout")] * 10)
    with pytest.raises(RuntimeError, match="retries"):
        fetch_page_with_retry(ex, "BTC/USDT", "1m", T0_MS, 10, max_retries=2, sleep=lambda s: None)


def test_cache_round_trip_does_not_refetch(tmp_path):
    from src.data.ohlcv import load_or_fetch_window

    ex = FakeMinuteExchange([minute(i) for i in range(5)])
    calls = []

    def fetch():
        calls.append(1)
        return fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(minutes=5),
                                  now_utc="2026-01-01", exchange=ex)

    df1, info1 = load_or_fetch_window(tmp_path / "w.parquet", fetch)
    df2, info2 = load_or_fetch_window(tmp_path / "w.parquet", fetch)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)
    assert info1 == info2


def test_measure_max_page_finds_the_silent_cap():
    from src.data.ohlcv import measure_max_page

    ex = FakeMinuteExchange([minute(i) for i in range(2000)], max_page=300)
    assert measure_max_page(ex, "BTC/USDT", T0_MS) == {100: 100, 300: 300, 1000: 300, 1500: 300}


class HistoryLimitedExchange(FakeMinuteExchange):
    """Returns nothing when asked for data older than ``served_from`` (like OKX)."""

    def __init__(self, candles, served_from):
        super().__init__(candles)
        self.served_from = served_from

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        return [] if since < self.served_from else super().fetch_ohlcv(symbol, timeframe,
                                                                       since, limit)


def test_first_available_candle_direct_and_by_bisection():
    from src.data.ohlcv import first_available_candle

    candles = [minute(i) for i in range(100, 1000)]
    assert first_available_candle(FakeMinuteExchange(candles), "BTC/USDT", 0,
                                  T0_MS + 500 * MIN_MS) == T0_MS + 100 * MIN_MS
    ex = HistoryLimitedExchange(candles, served_from=T0_MS + 137 * MIN_MS)
    assert first_available_candle(ex, "BTC/USDT", T0_MS, T0_MS + 500 * MIN_MS) == \
        T0_MS + 137 * MIN_MS


def test_timestamp_containing_451_is_not_a_region_block():
    pauses = []
    ex = Flaky([ccxt.RequestTimeout("timeout GET ...?startTime=1760119200451")])
    assert fetch_page_with_retry(ex, "BTC/USDT", "1m", T0_MS, 10, sleep=pauses.append)
    assert pauses == [1.0]


# --- price="mark" (notebook 04) -------------------------------------------------------


class PriceAwareExchange(FakeMinuteExchange):
    """Serves different candles for last and mark, and records the params it received."""

    def __init__(self, last, mark):
        super().__init__(last, max_page=1000)
        self.mark, self.params_seen = mark, []

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        self.params_seen.append(params)
        source = self.mark if (params or {}).get("price") == "mark" else self.candles
        return [c for c in source if c[0] >= since][:limit]


def test_price_none_calls_fetch_ohlcv_exactly_as_before():
    ex = PriceAwareExchange([minute(i) for i in range(3)], [minute(i, price=50.0) for i in range(3)])
    df, info = fetch_ohlcv_window("binanceusdm", "BTC/USDT:USDT", T0, T0 + pd.Timedelta(minutes=3),
                                  now_utc="2026-01-01", exchange=ex)
    assert ex.params_seen == [None]          # no params keyword was sent at all
    assert (df["close"] == 100.0).all() and "price" not in info


def test_price_mark_is_passed_to_ccxt_and_returns_mark_candles():
    ex = PriceAwareExchange([minute(i) for i in range(3)],
                            [minute(i, price=50.0, volume=0.0) for i in range(3)])
    df, info = fetch_ohlcv_window("binanceusdm", "BTC/USDT:USDT", T0, T0 + pd.Timedelta(minutes=3),
                                  now_utc="2026-01-01", exchange=ex, price="mark")
    assert ex.params_seen == [{"price": "mark"}]
    assert (df["close"] == 50.0).all() and info["price"] == "mark"


def test_unknown_price_type_is_rejected():
    with pytest.raises(ValueError, match="price"):
        fetch_page_with_retry(FakeMinuteExchange([]), "BTC/USDT:USDT", "1m", T0_MS, 10,
                              price="index")


def test_mark_cache_neither_reads_nor_overwrites_last_cache(tmp_path):
    from src.data.ohlcv import cache_path, load_or_fetch_window

    ex = PriceAwareExchange([minute(i) for i in range(3)], [minute(i, price=50.0) for i in range(3)])
    end = T0 + pd.Timedelta(minutes=3)

    def fetcher(price):
        return lambda: fetch_ohlcv_window("binanceusdm", "BTC/USDT:USDT", T0, end,
                                          now_utc="2026-01-01", exchange=ex, price=price)

    last_path = cache_path(tmp_path, "binanceusdm", "BTC/USDT:USDT", "1m", "w")
    mark_path = cache_path(tmp_path, "binanceusdm", "BTC/USDT:USDT", "1m", "w", price="mark")
    assert last_path != mark_path
    assert last_path.name == "binanceusdm_BTCUSDT-USDT_1m_w_last.parquet"
    assert mark_path.name == "binanceusdm_BTCUSDT-USDT_1m_w_mark.parquet"

    last_df, _ = load_or_fetch_window(last_path, fetcher(None))
    last_bytes = last_path.read_bytes()
    mark_df, mark_info = load_or_fetch_window(mark_path, fetcher("mark"))

    assert len(ex.params_seen) == 2              # mark was downloaded, not read from last's file
    assert (mark_df["close"] == 50.0).all() and (last_df["close"] == 100.0).all()
    assert mark_info["price"] == "mark"
    assert last_path.read_bytes() == last_bytes  # last's cache is untouched


def test_hourly_window_uses_one_hour_step():
    hour_ms = 3_600_000
    candles = [[T0_MS + i * hour_ms, 100.0, 101.0, 99.0, 100.0, 1.0] for i in range(30)]
    ex = FakeMinuteExchange(candles)
    df, info = fetch_ohlcv_window("binance", "BTC/USDT", T0, T0 + pd.Timedelta(hours=24),
                                  timeframe="1h", now_utc="2026-01-01", exchange=ex)
    assert len(df) == 24 and df.index[-1] == T0 + pd.Timedelta(hours=23)
