"""Download and clean daily OHLCV candles from a crypto exchange (via ccxt).

OHLCV = Open, High, Low, Close, Volume. Each candle summarizes one period (here: one UTC day).

Conventions (see CLAUDE.md):
- Exchange timestamps are Unix **milliseconds** and mark the candle's **open time**, in **UTC**.
- A price belongs to one exchange and one quote currency (e.g. BTC/USDT on Binance spot).
- The candle that is still forming (today's) is dropped: its high/low/close/volume are not final.

The exchange is passed in as an object with a ccxt-style ``fetch_ohlcv`` method, so the same
code works for any ccxt exchange and tests can use a fake exchange without any network access.
"""

from __future__ import annotations

import pandas as pd

ONE_DAY = pd.Timedelta(days=1)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_raw_ohlcv(exchange, symbol, since, timeframe="1d", limit=1000, max_pages=1000):
    """Download all candles from ``since`` (Unix ms) up to now, page by page.

    Each call returns at most ``limit`` candles. The next page starts at the last timestamp
    we received + 1 ms, so the last candle is never requested twice. We stop when a page
    comes back empty or does not move forward in time.

    Returns a list of ``[timestamp_ms, open, high, low, close, volume]`` rows, as sent by
    the exchange (no cleaning yet).
    """
    rows = []
    for _ in range(max_pages):
        page = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not page:
            break
        rows.extend(page)
        next_since = page[-1][0] + 1
        if next_since <= since:  # defensive: never loop forever on a stuck API
            break
        since = next_since
    else:
        raise RuntimeError(f"Hit max_pages={max_pages} for {symbol}; increase it or check the API")
    return rows


def raw_to_frame(rows):
    """Turn raw exchange rows into a DataFrame indexed by the candle open time (UTC).

    The timestamp unit is **milliseconds** (``unit="ms"``). Reading it as seconds would put
    the dates ~50,000 years in the future (pandas would actually fail first).
    """
    df = pd.DataFrame(rows, columns=["timestamp_ms", *OHLCV_COLUMNS])
    df["open_time_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df.set_index("open_time_utc")[["timestamp_ms", *OHLCV_COLUMNS]]


def drop_incomplete_candles(df, now_utc, period=ONE_DAY):
    """Keep only candles that have fully closed by ``now_utc``.

    A daily candle that opens at T covers [T, T + 1 day). It is closed only when
    T + 1 day <= now. Today's candle fails this test: its close is just the latest trade,
    and its high/low/volume can still change. Using it would mean using information that,
    in a backtest, would not have been final yet.
    """
    now_utc = pd.Timestamp(now_utc)
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware (UTC)")
    closed = df.index + period <= now_utc
    return df[closed]


def find_duplicate_dates(df):
    """Return the index values that appear more than once (empty if none)."""
    return df.index[df.index.duplicated()].unique()


def find_missing_days(df):
    """Return the calendar days (UTC) between the first and last candle that have no candle."""
    expected = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
    return expected.difference(df.index)


def check_daily_candles(df):
    """Run the data-quality checks and return a small report (dict).

    Raises ``ValueError`` for problems we must never analyze through (duplicates, wrong order,
    non-positive prices). Missing days are only *reported*, because an exchange outage is a
    real event: the caller decides what to do about them.
    """
    duplicates = find_duplicate_dates(df)
    if len(duplicates):
        raise ValueError(f"Duplicate candle dates: {list(duplicates)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Candles are not sorted by time")
    if not (df.index == df.index.normalize()).all():
        raise ValueError("Some daily candles do not open at 00:00 UTC")
    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("Found prices <= 0")
    return {
        "candles": len(df),
        "first": df.index.min(),
        "last": df.index.max(),
        "duplicates": 0,
        "missing_days": list(find_missing_days(df)),
    }


def fetch_daily_ohlcv(exchange, symbol, start_utc, now_utc=None):
    """Download, clean and check daily candles for one symbol.

    ``start_utc`` is a date string or Timestamp, interpreted as UTC.
    ``now_utc`` defaults to the current time; pass it explicitly to make runs reproducible.
    Returns ``(df, report)``.
    """
    start = pd.Timestamp(start_utc, tz="UTC")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    since_ms = int(start.timestamp() * 1000)

    rows = fetch_raw_ohlcv(exchange, symbol, since_ms, timeframe="1d")
    df = raw_to_frame(rows).sort_index()
    df = drop_incomplete_candles(df, now)
    report = check_daily_candles(df)
    return df, report


def align_closes(frames):
    """Combine the ``close`` column of several symbols into one table, one row per date.

    ``frames`` is a dict ``{name: ohlcv_df}``. Only dates present for **every** symbol are kept
    (an inner join), and the dates dropped from each symbol are returned so nothing disappears
    silently. Returns ``(closes, dropped)``.
    """
    closes = pd.concat({name: df["close"] for name, df in frames.items()}, axis=1, join="inner")
    dropped = {name: df.index.difference(closes.index) for name, df in frames.items()}
    return closes, dropped
