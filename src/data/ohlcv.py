"""Download and clean OHLCV candles from a crypto exchange (via ccxt).

OHLCV = Open, High, Low, Close, Volume. Each candle summarizes one period (one UTC day in
notebook 01, one UTC minute in notebook 03).

Conventions (see CLAUDE.md):
- Exchange timestamps are Unix **milliseconds** and mark the candle's **open time**, in **UTC**.
- A price belongs to one exchange and one quote currency (e.g. BTC/USDT on Binance spot).
- The candle that is still forming is dropped: its high/low/close/volume are not final.

The exchange is passed in as an object with a ccxt-style ``fetch_ohlcv`` method, so the same
code works for any ccxt exchange and tests can use a fake exchange without any network access.
``make_exchange(exchange_id)`` builds the real, public (no API key) ccxt client.

Measured on 2026-11-15 with ccxt 4.5.92 (BTC/USDT spot, 1m), see notebook 03, step 0:
- Max candles per request: Binance 1000, Bybit 1000, OKX 300 (asking for more silently
  returns the max; asking for nothing returns 500 / 200 / 100).
- ccxt returns candles **oldest first** for all three (Bybit and OKX send newest first; ccxt
  re-sorts them). ``since`` is **inclusive**.
- ccxt does not paginate by itself; with ``params={"paginate": True}`` it ignored ``since``
  on Binance, so we paginate by hand.
"""

from __future__ import annotations

import re
import time

import pandas as pd

ONE_DAY = pd.Timedelta(days=1)
ONE_MINUTE = pd.Timedelta(minutes=1)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Largest page each exchange really returns for 1m candles (measured, not from the docs).
EXCHANGE_MAX_LIMIT = {"binance": 1000, "bybit": 1000, "okx": 300}

# HTTP codes that mean "you are not allowed here" (e.g. region block): retrying will not help.
ACCESS_DENIED_CODES = ("451", "403")


def fetch_raw_ohlcv(exchange, symbol, since, timeframe="1d", limit=1000, max_pages=1000,
                    until=None):
    """Download all candles from ``since`` (Unix ms) up to now, page by page.

    Each call returns at most ``limit`` candles. The next page starts at the last timestamp
    we received + 1 ms, so the last candle is never requested twice. We stop when a page
    comes back empty or does not move forward in time.

    ``until`` (Unix ms, optional): stop once we reach candles opening after it, and drop
    those candles. Without it, the download runs up to the present.

    Returns a list of ``[timestamp_ms, open, high, low, close, volume]`` rows, as sent by
    the exchange (no cleaning yet).
    """
    rows = []
    for _ in range(max_pages):
        page = fetch_page_with_retry(exchange, symbol, timeframe, since, limit)
        if not page:
            break
        if until is not None:
            reached_end = page[-1][0] >= until
            page = [row for row in page if row[0] <= until]
            rows.extend(page)
            if reached_end or not page:
                break
            if page[-1][0] + 1 <= since:  # defensive: never loop forever on a stuck API
                break
            since = page[-1][0] + 1
            continue
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


# --------------------------------------------------------------------------------------
# Intraday windows from any ccxt exchange (notebook 03)
# --------------------------------------------------------------------------------------

TIMEFRAME_STEP = {"1m": ONE_MINUTE, "1d": ONE_DAY}


def make_exchange(exchange_id):
    """Build a public ccxt client (no API key) with ccxt's built-in rate limiter switched on.

    ``exchange_id`` is the ccxt name, e.g. ``"binance"``, ``"okx"``, ``"bybit"``.
    """
    import ccxt  # imported here so the offline tests do not need ccxt's network stack

    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_page_with_retry(exchange, symbol, timeframe, since, limit, max_retries=5,
                          base_pause_s=1.0, sleep=time.sleep):
    """Call ``exchange.fetch_ohlcv`` once, retrying temporary failures with exponential backoff.

    Temporary failures are ccxt ``NetworkError`` subclasses: HTTP 429 (rate limit), 418, time-outs,
    5xx. We wait 1 s, 2 s, 4 s, ... between tries. HTTP 451/403 (blocked for your region / not
    allowed) is **not** retried, because waiting does not fix it: we raise ``PermissionError``.
    Errors that are not network errors (e.g. a bad symbol) are raised immediately.
    """
    try:
        from ccxt import NetworkError
    except ImportError:  # pragma: no cover - ccxt is a declared dependency
        NetworkError = OSError
    for attempt in range(max_retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except NetworkError as error:
            message = str(error)
            # Whole-word match, so a timestamp like 1760119200451 in the message does not count.
            if any(re.search(rf"\b{code}\b", message) for code in ACCESS_DENIED_CODES):
                raise PermissionError(f"Access denied (region block?) for {symbol}: {message}") \
                    from error
            if attempt == max_retries:
                raise RuntimeError(f"Still failing after {max_retries} retries: {message}") \
                    from error
            sleep(base_pause_s * 2**attempt)


def fetch_ohlcv_window(exchange_id, symbol, start_utc, end_utc, timeframe="1m", now_utc=None,
                       exchange=None, limit=None):
    """Download candles whose open time is in ``[start_utc, end_utc)`` (end excluded).

    Steps: paginate (``fetch_raw_ohlcv``), parse ms -> UTC, keep only the window, drop exact
    duplicates (raise if one timestamp comes with two different candles), and drop candles not
    closed by ``now_utc`` (the minute still forming).

    ``exchange`` lets tests inject a fake; by default ``make_exchange(exchange_id)`` is used.
    ``limit`` defaults to the measured maximum page size for that exchange.
    Returns ``(df, info)``: df has columns open/high/low/close/volume (volume in the **base**
    currency, i.e. BTC for BTC/USDT); info counts what each cleaning step removed.
    """
    from src.data.candles import drop_exact_duplicates, to_utc

    step = TIMEFRAME_STEP[timeframe]
    start, end = to_utc(start_utc), to_utc(end_utc)
    if end <= start:
        raise ValueError("end_utc must be after start_utc")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else to_utc(now_utc)
    exchange = exchange if exchange is not None else make_exchange(exchange_id)
    limit = limit or EXCHANGE_MAX_LIMIT.get(exchange_id, 500)

    since_ms = int(start.timestamp() * 1000)
    until_ms = int((end - step).timestamp() * 1000)  # open time of the last candle we want
    rows = fetch_raw_ohlcv(exchange, symbol, since_ms, timeframe=timeframe, limit=limit,
                           until=until_ms)
    df = raw_to_frame(rows)[OHLCV_COLUMNS].astype(float)
    received = len(df)
    df = df[(df.index >= start) & (df.index < end)]
    outside = received - len(df)
    df, n_duplicates = drop_exact_duplicates(df)
    before = len(df)
    df = drop_incomplete_candles(df, now, period=step)
    info = {
        "exchange": exchange_id,
        "rows_received": received,
        "outside_window": outside,
        "duplicates_dropped": n_duplicates,
        "incomplete_dropped": before - len(df),
        "candles": len(df),
    }
    return df, info


def load_or_fetch_window(path, fetch):
    """Cache ``(df, info)`` from ``fetch()``: df as Parquet at ``path``, info as JSON next to it.

    If both files exist they are read back and ``fetch`` is not called (no network).
    The cache lives under ``data/`` (gitignored).
    """
    import json
    from pathlib import Path

    path = Path(path)
    info_path = path.with_suffix(".info.json")
    if path.exists() and info_path.exists():
        return pd.read_parquet(path), json.loads(info_path.read_text())
    df, info = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    info_path.write_text(json.dumps(info))
    return df, info


# --------------------------------------------------------------------------------------
# Source checks (notebook 03, step 0): measure, don't assume
# --------------------------------------------------------------------------------------


def measure_max_page(exchange, symbol, since_ms, asks=(100, 300, 1000, 1500), timeframe="1m"):
    """Ask for pages of growing size and return ``{asked_limit: candles_returned}``.

    The largest returned value is the real cap per request (an exchange that silently caps
    returns the same number for every bigger ask). ``None`` asks for the exchange default.
    """
    return {limit: len(fetch_page_with_retry(exchange, symbol, timeframe, since_ms, limit))
            for limit in asks}


def first_available_candle(exchange, symbol, lo_ms, hi_ms, timeframe="1m", step_ms=60_000):
    """Open time (ms) of the oldest candle the exchange serves, searched in ``[lo_ms, hi_ms]``.

    Asks for candles ``since=lo_ms``. If the exchange returns some, the first one is the oldest
    (exchanges send the next available candle). If it returns nothing (it does not serve that
    far back), we bisect: find the smallest ``since`` that returns data, to ``step_ms``
    precision. ``hi_ms`` must be a time known to have data.
    """
    page = fetch_page_with_retry(exchange, symbol, timeframe, lo_ms, 1)
    if page:
        return page[0][0]
    while hi_ms - lo_ms > step_ms:
        mid = (lo_ms + hi_ms) // 2 // step_ms * step_ms
        if fetch_page_with_retry(exchange, symbol, timeframe, mid, 1):
            hi_ms = mid
        else:
            lo_ms = mid
    return fetch_page_with_retry(exchange, symbol, timeframe, hi_ms, 1)[0][0]
