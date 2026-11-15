"""Download intraday OHLC candles from Bitstamp's public API (no API key needed).

Endpoint: ``GET https://www.bitstamp.net/api/v2/ohlc/{pair}/?step=..&limit=..&start=..&end=..``

Conventions (see CLAUDE.md):
- ``start`` / ``end`` and the returned ``timestamp`` are Unix **seconds** (not milliseconds),
  and mark the candle's **open time**, in **UTC**.
- Bitstamp returns a candle for every step, even when nothing traded: those candles have
  ``volume == 0`` and repeat the previous close. They are kept and flagged downstream
  (see ``src/data/candles.py``), never silently treated as real trades.

Observed API quirk (2026-11-14): when the requested range holds fewer than ``limit`` candles,
Bitstamp ignores ``start`` and returns ``limit`` candles *ending* at ``end``. To avoid
overlapping pages we always send ``limit`` equal to the exact number of candles in the chunk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.candles import to_utc

BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
MAX_LIMIT = 1000


def plan_chunks(start_s, end_s, step_s=60, max_limit=MAX_LIMIT):
    """Split [start_s, end_s] (inclusive candle open times, Unix seconds) into requests.

    Returns a list of ``(chunk_start, chunk_end, limit)`` where ``limit`` is the exact number
    of candles in that chunk (never more than ``max_limit``).
    """
    if end_s < start_s:
        raise ValueError("end_s must be >= start_s")
    if (end_s - start_s) % step_s:
        raise ValueError("end_s - start_s must be a multiple of step_s")
    chunks = []
    chunk_start = start_s
    while chunk_start <= end_s:
        chunk_end = min(chunk_start + (max_limit - 1) * step_s, end_s)
        limit = (chunk_end - chunk_start) // step_s + 1
        chunks.append((chunk_start, chunk_end, limit))
        chunk_start = chunk_end + step_s
    return chunks


def _get_json(session, url, params, max_retries=6, pause_s=0.5):
    """GET with exponential backoff on HTTP 429 (too many requests)."""
    for attempt in range(max_retries):
        response = session.get(url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        time.sleep(pause_s)  # be polite between requests
        return response.json()
    raise RuntimeError(f"Still rate-limited after {max_retries} retries: {url}")


def fetch_bitstamp_ohlc(pair, start_utc, end_utc, step_s=60, session=None, pause_s=0.5):
    """Download raw candles for ``pair`` whose open time is in [start_utc, end_utc].

    ``start_utc`` / ``end_utc`` are UTC timestamps (strings or Timestamps). Returns the raw
    list of dicts sent by Bitstamp (strings for numbers), concatenated over all pages.
    """
    session = session or requests.Session()
    start_s = int(to_utc(start_utc).timestamp())  # seconds, not ms
    end_s = int(to_utc(end_utc).timestamp())
    rows = []
    for chunk_start, chunk_end, limit in plan_chunks(start_s, end_s, step_s):
        params = {"step": step_s, "limit": limit, "start": chunk_start, "end": chunk_end}
        payload = _get_json(session, BITSTAMP_OHLC_URL.format(pair=pair), params, pause_s=pause_s)
        rows.extend(payload["data"]["ohlc"])
    return rows


def bitstamp_to_frame(rows):
    """Turn raw Bitstamp rows into a float DataFrame indexed by open time (UTC).

    Timestamps are read with ``unit="s"``. No sorting or de-duplication happens here.
    """
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.astype({"timestamp": "int64", "open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    df.index = pd.to_datetime(df.pop("timestamp"), unit="s", utc=True)
    df.index.name = "open_time_utc"
    return df


def load_or_fetch_json(cache_path, fetch):
    """Return the JSON cached at ``cache_path``; if missing, call ``fetch()`` and cache it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    data = fetch()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data
