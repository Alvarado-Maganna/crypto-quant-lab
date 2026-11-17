"""Download and cache the funding-rate history of a Binance USD-M perpetual (notebook 05).

Endpoint: ``GET https://fapi.binance.com/fapi/v1/fundingRate`` (public, no API key). We call it
through ccxt's implicit method ``fapiPublicGetFundingRate``, which sends our parameters unchanged
and returns the raw JSON rows, so no field is lost or renamed. ccxt's client also brings its
rate limiter (``make_exchange("binanceusdm")``).

Measured on 2026-11-17 with ccxt 4.5.92 (BTCUSDT, ETHUSDT), see notebook 05, step 0:
- ``limit`` up to 1000 is honoured; 1500 is **rejected** ("illegal params"). Default: 100.
- Rows come **oldest first**; ``startTime`` is inclusive.
- **Trap:** ``startTime=0`` is treated as "no startTime" and returns the *latest* rows, not the
  first ones. Always start from a real date (we refuse ``start_ms <= 0``).
- ``fundingTime`` is Unix **milliseconds**, UTC, and is often a few ms after the hour
  (e.g. ``16:00:00.007``). We keep the raw value; rounding is only for classification.
- Fields per row: ``symbol``, ``fundingRate`` (string), ``fundingTime`` (ms), ``markPrice``
  (string, empty ``""`` in old rows), ``rateType`` (``"Regular"`` or ``"Special"``).
- ccxt's unified ``fetch_funding_rate_history`` also returned 1000 rows per call and kept
  ``rateType``/``markPrice`` inside ``info``; we use the raw method because it is simpler to
  check (one row in = one row out).

Rate limit (docs): 500 requests / 5 min / IP, shared with ``/fapi/v1/fundingInfo``. We pause
``pause_s`` (default 0.7 s) between pages, which keeps us under ~430 requests / 5 min even
without ccxt's limiter.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

FUNDING_MAX_LIMIT = 1000          # measured: 1000 accepted, 1500 rejected
RATE_TYPES = ("Regular", "Special")
ACCESS_DENIED_CODES = ("451", "403")


def call_with_retry(fn, params, max_retries=5, base_pause_s=1.0, sleep=time.sleep):
    """Call ``fn(params)``, retrying ccxt network errors with exponential backoff (1, 2, 4 s...).

    HTTP 451/403 (blocked / not allowed) is not retried: we raise ``PermissionError``.
    Any other error (e.g. bad parameters) is raised at once.
    """
    try:
        from ccxt import NetworkError
    except ImportError:  # pragma: no cover - ccxt is a declared dependency
        NetworkError = OSError
    for attempt in range(max_retries + 1):
        try:
            return fn(params)
        except NetworkError as error:
            message = str(error)
            if any(re.search(rf"\b{code}\b", message) for code in ACCESS_DENIED_CODES):
                raise PermissionError(f"Access denied (region block?): {message}") from error
            if attempt == max_retries:
                raise RuntimeError(f"Still failing after {max_retries} retries: {message}") \
                    from error
            sleep(base_pause_s * 2**attempt)


def fetch_raw_funding(fetch_page, symbol, start_ms, end_ms, limit=FUNDING_MAX_LIMIT,
                      pause_s=0.7, sleep=time.sleep, max_pages=1000):
    """Download every funding row of ``symbol`` with ``start_ms <= fundingTime < end_ms``.

    ``fetch_page(params) -> list[dict]`` is one call to the endpoint (in the notebook,
    ``exchange.fapiPublicGetFundingRate``; in tests, a fake). Each page asks for
    ``startTime = last fundingTime received + 1`` (so no row is asked for twice) and
    ``endTime = end_ms - 1`` (the API's endTime is inclusive; ours is exclusive).
    We stop when a page is empty or shorter than ``limit`` (the API had nothing more).

    Returns the raw rows (dicts), unchanged.
    """
    if start_ms <= 0:
        raise ValueError("start_ms must be > 0: the API treats startTime=0 as missing and "
                         "returns the latest rows instead of the first ones")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be after start_ms")
    rows, since = [], int(start_ms)
    for page_number in range(max_pages):
        if page_number:
            sleep(pause_s)
        params = {"symbol": symbol, "startTime": since, "endTime": int(end_ms) - 1,
                  "limit": limit}
        page = call_with_retry(fetch_page, params, sleep=sleep)
        if not page:
            break
        rows.extend(page)
        last = int(page[-1]["fundingTime"])
        if len(page) < limit or last + 1 >= end_ms:
            break
        if last + 1 <= since:  # defensive: never loop forever on a stuck API
            break
        since = last + 1
    else:
        raise RuntimeError(f"Hit max_pages={max_pages} for {symbol}; increase it or check the API")
    return rows


def funding_to_frame(rows):
    """Turn raw rows into a DataFrame indexed by ``funding_time_utc`` (parsed with ``unit="ms"``).

    Columns:
    - ``funding_time_ms`` (int): the raw timestamp, kept so ms offsets are not lost;
    - ``funding_rate`` (float) and ``funding_rate_str`` (the exact string sent by the API,
      for exact comparisons with ``Decimal``);
    - ``mark_price`` (float, NaN where the API sent ``""``);
    - ``rate_type`` and ``symbol``.
    """
    columns = ["funding_time_ms", "funding_rate", "funding_rate_str", "mark_price",
               "rate_type", "symbol"]
    if not rows:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="funding_time_utc")
        return empty
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "funding_time_ms": df["fundingTime"].astype("int64"),
        "funding_rate": df["fundingRate"].astype(float),
        "funding_rate_str": df["fundingRate"].astype(str),
        "mark_price": pd.to_numeric(df["markPrice"].replace("", None), errors="coerce"),
        "rate_type": df["rateType"].astype(str) if "rateType" in df else "Regular",
        "symbol": df["symbol"].astype(str),
    })
    out.index = pd.to_datetime(out["funding_time_ms"], unit="ms", utc=True)
    out.index.name = "funding_time_utc"
    return out


def fetch_funding_window(fetch_page, symbol, start_utc, end_utc, **kwargs):
    """Download, clean and check funding rows with time in ``[start_utc, end_utc)``.

    Cleaning steps (each is counted in ``info``; nothing is filled):
    1. sort by time; drop rows outside the window (should be 0);
    2. drop exact duplicate rows; **raise** if one timestamp comes with two different rows;
    3. unknown ``rateType`` values raise; ``"Special"`` rows are counted and dropped (they are
       extra one-off payments, not the regular funding we study).

    Returns ``(df, info)``. ``info`` stores the requested window in ms so a cache can be checked.
    """
    from src.data.candles import to_utc

    start, end = to_utc(start_utc), to_utc(end_utc)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    rows = fetch_raw_funding(fetch_page, symbol, start_ms, end_ms, **kwargs)
    df = funding_to_frame(rows).sort_index(kind="stable")
    received = len(df)
    df = df[(df["funding_time_ms"] >= start_ms) & (df["funding_time_ms"] < end_ms)]
    outside = received - len(df)

    repeated = df[df["funding_time_ms"].duplicated(keep=False)]
    if len(repeated):
        versions = repeated.drop_duplicates().groupby("funding_time_ms").size()
        if (versions > 1).any():
            raise ValueError(f"Conflicting funding rows at: {list(versions[versions > 1].index)}")
    before = len(df)
    df = df[~df["funding_time_ms"].duplicated(keep="first")]
    duplicates = before - len(df)

    unknown = set(df["rate_type"]) - set(RATE_TYPES)
    if unknown:
        raise ValueError(f"Unknown rateType values: {unknown}")
    special = int((df["rate_type"] == "Special").sum())
    df = df[df["rate_type"] == "Regular"]
    if (df["symbol"] != symbol).any():
        raise ValueError(f"Rows for another symbol than {symbol}")
    info = {
        "symbol": symbol,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "rows_received": received,
        "outside_window": outside,
        "duplicates_dropped": duplicates,
        "special_dropped": special,
        "rows": len(df),
    }
    return df, info


def funding_cache_path(cache_dir, exchange_id, symbol, label):
    """Cache file for a funding history, separate from candle caches (suffix ``_funding``).

    Example: ``binanceusdm_BTCUSDT_full_funding.parquet``.
    """
    return Path(cache_dir) / f"{exchange_id}_{symbol}_{label}_funding.parquet"


def load_or_fetch_funding(path, fetch, start_utc, end_utc):
    """Cache ``(df, info)`` from ``fetch()``, and check the cache covers exactly the window asked.

    The window is stored in the ``.info.json`` file next to the Parquet file. If the cached
    window differs from ``[start_utc, end_utc)``, or any cached row falls outside it, we raise
    instead of silently reusing data from another window. Delete the files to download again.
    """
    from src.data.candles import to_utc

    path = Path(path)
    info_path = path.with_suffix(".info.json")
    start_ms = int(to_utc(start_utc).timestamp() * 1000)
    end_ms = int(to_utc(end_utc).timestamp() * 1000)
    if path.exists() and info_path.exists():
        df, info = pd.read_parquet(path), json.loads(info_path.read_text())
    else:
        df, info = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        info_path.write_text(json.dumps(info))
    if (info.get("start_ms"), info.get("end_ms")) != (start_ms, end_ms):
        raise ValueError(f"Cache {path.name} covers [{info.get('start_ms')}, {info.get('end_ms')}) "
                         f"ms, not the requested [{start_ms}, {end_ms}); delete it to refetch")
    if len(df) and ((df["funding_time_ms"] < start_ms) | (df["funding_time_ms"] >= end_ms)).any():
        raise ValueError(f"Cache {path.name} has rows outside the requested window")
    return df, info
