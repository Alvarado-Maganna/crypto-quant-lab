"""Collect and store raw order-book snapshots from Binance spot and USD-M (notebook 06).

One *snapshot* is one REST call to the depth endpoint of one market: the best ``limit`` price
levels on each side (bids = buy orders, asks = sell orders), each level being a price and the
total amount resting there. A *round* is one snapshot of every market, taken one after another.

Endpoints (public, no API key), called through ccxt's raw methods so the response is unchanged:
- Spot: ``GET /api/v3/depth`` (``exchange.publicGetDepth``), exchange id ``binance``.
- USD-M futures: ``GET /fapi/v1/depth`` (``exchange.fapiPublicGetDepth``), id ``binanceusdm``.

Measured on 2026-11-19 with ccxt 4.5.92 (see notebook 06, step 0):
- Spot ``limit``: weight 5 for 1-100 levels, 25 for 101-500, 50 for 501-1000, 250 for
  1001-5000 (read from the ``x-mbx-used-weight-1m`` header after each call); same as the docs.
  The docs say "If limit > 5000, only 5000 entries will be returned", but ``limit=6000``
  returned 6000 levels per side (weight 250). We stay at the documented 5000.
- USD-M ``limit``: only 5, 10, 20, 50, 100, 500, 1000 are valid (1500, 2000, 5000 are rejected
  with error -1130). Weight 2 for 5-50, 5 for 100, 10 for 500, 20 for 1000 (same as the docs).
  The docs add that "Retail Price Improvement (RPI) orders are not visible" in this response.
- Weight budget per IP and minute (``exchangeInfo``): spot 6000, USD-M 2400.
- Spot responses carry no exchange timestamp (fields: ``lastUpdateId``, ``bids``, ``asks``).
  USD-M responses add ``E`` (message time) and ``T`` (transaction time), Unix **ms**, UTC.
- Prices and amounts come as strings (e.g. ``"61672.00000000"``); amounts are in the
  **base** currency (BTC for BTC/USDT).

Storage (``HourlyParquetStore``): one Parquet file per UTC hour, ``<out_dir>/YYYYMMDD_HH.parquet``.
While an hour is in progress, each flush writes a small part file in
``<out_dir>/parts/YYYYMMDD_HH/`` (written to a temporary name, then renamed, so a crash never
leaves a half-written file). When the hour is over, its parts are merged into the hour file.
An interruption loses at most the rounds not yet flushed. ``load_snapshots`` reads hour files
and any leftover parts. Schema: see ``SNAPSHOT_COLUMNS``.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPOT_DEPTH_MAX_LIMIT = 5000                          # documented max (5001+ accepted, undocumented)
USDM_DEPTH_VALID_LIMITS = (5, 10, 20, 50, 100, 500, 1000)
SPOT_WEIGHT_STEPS = ((100, 5), (500, 25), (1000, 50), (5000, 250))   # (up to N levels, weight)
USDM_WEIGHT_STEPS = ((50, 2), (100, 5), (500, 10), (1000, 20))
WEIGHT_LIMIT_PER_MINUTE = {"binance": 6000, "binanceusdm": 2400}

# One row per price level. Timestamps are tz-aware UTC (millisecond precision).
SNAPSHOT_COLUMNS = {
    "market": "label, e.g. 'spot BTC/USDT' or 'perp BTC/USDT:USDT'",
    "exchange": "ccxt exchange id: 'binance' (spot) or 'binanceusdm' (USD-M futures)",
    "symbol": "ccxt symbol, e.g. 'BTC/USDT' or 'BTC/USDT:USDT'",
    "side": "'bid' (buy orders) or 'ask' (sell orders)",
    "level": "0 = best price on that side, 1 = next one, ...",
    "price": "price in the quote currency (USDT)",
    "amount": "amount resting at that price, in the base currency (BTC, ETH, ...)",
    "local_request_utc": "our clock just before the first attempt of the request (UTC); if the "
                         "request was retried, the latency includes the waits",
    "local_receive_utc": "our clock just after receiving the answer (UTC): the first moment the "
                         "snapshot is known, so use it (or 'E' on USD-M) to line snapshots up "
                         "with other data without look-ahead",
    "exchange_event_utc": "exchange message time 'E' (USD-M only; NaT on spot)",
    "exchange_transaction_utc": "exchange transaction time 'T' (USD-M only; NaT on spot)",
    "last_update_id": "exchange book version; equal ids = identical books",
    "requested_limit": "levels per side we asked for",
    "round_id": "slot number within one collection run (0, 1, 2, ...; jumps where slots were "
                "skipped). It restarts at 0 in every run, so identify a snapshot by (market, "
                "local_request_utc), never by round_id alone",
}
DEDUP_KEY = ["market", "local_request_utc", "side", "level"]


@dataclass(frozen=True)
class Market:
    """One market to snapshot. ``label`` names it in the data; ``limit`` = levels per side."""

    label: str
    exchange_id: str
    symbol: str
    api_symbol: str
    limit: int


def default_markets(alt_symbol="PENGU/USDT"):
    """The 4 markets of notebook 06: spot BTC, ETH, one altcoin, and the BTC USD-M perpetual."""
    spot = [Market(f"spot {s}", "binance", s, s.replace("/", ""), SPOT_DEPTH_MAX_LIMIT)
            for s in ("BTC/USDT", "ETH/USDT", alt_symbol)]
    perp = Market("perp BTC/USDT:USDT", "binanceusdm", "BTC/USDT:USDT", "BTCUSDT", 1000)
    return spot + [perp]


def depth_weight(exchange_id, limit):
    """Request weight of one depth call (measured, see module docstring)."""
    if exchange_id == "binanceusdm" and limit not in USDM_DEPTH_VALID_LIMITS:
        raise ValueError(f"USD-M depth limit must be one of {USDM_DEPTH_VALID_LIMITS}")
    steps = SPOT_WEIGHT_STEPS if exchange_id == "binance" else USDM_WEIGHT_STEPS
    for max_levels, weight in steps:
        if 1 <= limit <= max_levels:
            return weight
    raise ValueError(f"limit {limit} out of range for {exchange_id}")


def round_weight(markets):
    """Total weight of one round, per exchange id."""
    totals = {}
    for m in markets:
        totals[m.exchange_id] = totals.get(m.exchange_id, 0) + depth_weight(m.exchange_id, m.limit)
    return totals


def now_ms():
    return time.time_ns() // 1_000_000


def raw_depth_method(exchange, exchange_id):
    """The ccxt raw method that calls the depth endpoint of ``exchange_id``."""
    return exchange.publicGetDepth if exchange_id == "binance" else exchange.fapiPublicGetDepth


def fetch_with_retry(fn, params, max_retries=6, base_pause_s=2.0, max_pause_s=120.0,
                     sleep=time.sleep, log=print):
    """Call ``fn(params)``; on ccxt network / rate-limit errors wait and retry (2, 4, 8 ... s).

    Rate-limit answers (HTTP 429 "too many requests") wait at least 60 s. HTTP 418 means the IP
    is **banned** for a while (minutes to days, longer for repeat offenders): retrying would make
    it worse, so we raise ``IPBanned`` and ``collect`` stops. Other errors (e.g. invalid
    parameters) are raised at once: retrying will not fix them.
    """
    import ccxt

    for attempt in range(max_retries + 1):
        try:
            return fn(params)
        except ccxt.NetworkError as error:  # includes RateLimitExceeded, DDoSProtection, timeouts
            message = str(error)[:200]
            if " 418 " in f" {message} ":
                raise IPBanned(f"HTTP 418, IP banned by Binance: {message}") from error
            if attempt == max_retries:
                raise RuntimeError(f"Still failing after {max_retries} retries: {message}") \
                    from error
            pause = min(base_pause_s * 2**attempt, max_pause_s)
            if isinstance(error, ccxt.DDoSProtection) or " 429 " in f" {message} ":
                pause = max(pause, 60.0)
            log(f"{type(error).__name__} on {params} (attempt {attempt + 1}): {message}; "
                f"retrying in {pause:.0f} s")
            sleep(pause)


def used_weight(exchange):
    """Weight used in the current minute, from the last response header (None if absent)."""
    headers = getattr(exchange, "last_response_headers", None) or {}
    for key, value in headers.items():
        if key.lower() == "x-mbx-used-weight-1m":
            return int(value)
    return None


def weight_pause_s(used, limit, now_ms_value, max_fraction=0.8, margin_s=3.0):
    """Seconds to wait so the weight counter resets, if ``used`` exceeds ``max_fraction`` of
    ``limit``; else 0. Binance resets the counter at each minute boundary of *its* clock; we
    wait for the next boundary of ours plus ``margin_s`` (our PC was measured ~0.85 s ahead of
    Binance in notebook 06, so a 1 s margin would leave almost none).
    """
    if used is None or used <= max_fraction * limit:
        return 0.0
    ms_into_minute = now_ms_value % 60_000
    return (60_000 - ms_into_minute) / 1000 + margin_s


class IPBanned(RuntimeError):
    """Binance answered HTTP 418: the IP is banned for a while. Collection must stop."""


def fetch_snapshot(exchange, market, clock=now_ms, **retry_kwargs):
    """One raw depth snapshot of ``market``, with our request/receive times (Unix ms, UTC)."""
    method = raw_depth_method(exchange, market.exchange_id)
    params = {"symbol": market.api_symbol, "limit": market.limit}
    request_ms = clock()
    raw = fetch_with_retry(method, params, **retry_kwargs)
    receive_ms = clock()
    return {"raw": raw, "local_request_ms": request_ms, "local_receive_ms": receive_ms}


def snapshot_to_frame(snapshot, market, round_id):
    """Turn one raw snapshot into the long table of ``SNAPSHOT_COLUMNS`` (one row per level)."""
    raw = snapshot["raw"]
    parts = []
    for side, key in (("bid", "bids"), ("ask", "asks")):
        levels = raw[key]
        parts.append(pd.DataFrame({
            "side": side,
            "level": pd.Series(range(len(levels)), dtype="int32"),
            "price": pd.Series([float(lv[0]) for lv in levels], dtype="float64"),
            "amount": pd.Series([float(lv[1]) for lv in levels], dtype="float64"),
        }))
    df = pd.concat(parts, ignore_index=True)

    def utc(ms):
        return pd.to_datetime(ms, unit="ms", utc=True) if ms is not None else pd.NaT

    df.insert(0, "market", market.label)
    df.insert(1, "exchange", market.exchange_id)
    df.insert(2, "symbol", market.symbol)
    df["local_request_utc"] = utc(snapshot["local_request_ms"])
    df["local_receive_utc"] = utc(snapshot["local_receive_ms"])
    df["exchange_event_utc"] = utc(raw.get("E") and int(raw["E"]))
    df["exchange_transaction_utc"] = utc(raw.get("T") and int(raw["T"]))
    df["last_update_id"] = int(raw["lastUpdateId"])
    df["requested_limit"] = int(market.limit)
    df["round_id"] = int(round_id)
    for col in ("local_request_utc", "local_receive_utc", "exchange_event_utc",
                "exchange_transaction_utc"):
        df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ms, UTC]")
    return df[list(SNAPSHOT_COLUMNS)]


def hour_key(ts):
    """'YYYYMMDD_HH' of a UTC timestamp: the name of the hourly file it belongs to."""
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y%m%d_%H")


LOCKED_FILE_PAUSES_S = (0.5, 1.0, 2.0, 4.0, 8.0)   # waits between attempts on a locked file


def _retry_if_locked(action, pauses_s=LOCKED_FILE_PAUSES_S, sleep=time.sleep):
    """Run ``action()``; on ``PermissionError`` wait and try again (``pauses_s``), then raise.

    On Windows, renaming over or deleting a file fails with ``PermissionError`` while another
    program has it open (an antivirus scan, the search indexer, a notebook or
    ``collector_status.py`` reading it). Such locks usually last well under a second.
    """
    for pause in (*pauses_s, None):
        try:
            return action()
        except PermissionError:
            if pause is None:
                raise
            sleep(pause)


def _atomic_write_parquet(df, path, sleep=time.sleep):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    _retry_if_locked(lambda: df.to_parquet(tmp, index=False), sleep=sleep)
    try:  # atomic on the same disk: readers see the old file or the new one
        _retry_if_locked(lambda: os.replace(tmp, path), sleep=sleep)
    except BaseException:
        tmp.unlink(missing_ok=True)   # do not leave a stray .tmp behind
        raise


class HourlyParquetStore:
    """Append-safe storage: one Parquet file per UTC hour (see module docstring)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.parts_dir = self.out_dir / "parts"

    def hour_path(self, key):
        return self.out_dir / f"{key}.parquet"

    def append(self, df):
        """Write ``df`` as new part file(s), split by the UTC hour of ``local_request_utc``."""
        if df.empty:
            return []
        written = []
        for key, chunk in df.groupby(df["local_request_utc"].dt.strftime("%Y%m%d_%H")):
            first_ms = int(chunk["local_request_utc"].min().timestamp() * 1000)
            path = self.parts_dir / key / f"part-{first_ms}-{uuid.uuid4().hex[:8]}.parquet"
            _atomic_write_parquet(chunk, path)
            written.append(path)
        return written

    def compact(self, key):
        """Merge the parts of hour ``key`` (and its existing hour file) into the hour file."""
        part_dir = self.parts_dir / key
        parts = sorted(part_dir.glob("*.parquet")) if part_dir.exists() else []
        if not parts:
            return None
        frames = [pd.read_parquet(p) for p in parts]
        if self.hour_path(key).exists():
            frames.insert(0, pd.read_parquet(self.hour_path(key)))
        merged = _dedup(pd.concat(frames, ignore_index=True))
        _atomic_write_parquet(merged, self.hour_path(key))
        for p in parts:  # only after the hour file is safely written
            try:
                _retry_if_locked(p.unlink)
            except PermissionError:
                pass  # still locked: harmless, the next merge drops its duplicate rows
        try:
            part_dir.rmdir()
        except OSError:
            pass
        return self.hour_path(key)

    def compact_finished_hours(self, now_utc):
        """Compact every hour that is over (strictly before the hour of ``now_utc``)."""
        if not self.parts_dir.exists():
            return []
        current = hour_key(now_utc)
        return [self.compact(d.name) for d in sorted(self.parts_dir.iterdir())
                if d.is_dir() and d.name < current]


def _dedup(df):
    df = df.drop_duplicates(subset=DEDUP_KEY, keep="last")
    return df.sort_values(["local_request_utc", "market", "side", "level"], kind="stable") \
        .reset_index(drop=True)


CATEGORICAL_COLUMNS = ["market", "exchange", "symbol", "side"]


def load_snapshots(out_dir, markets=None):
    """Read every stored level (hour files + leftover parts) as one DataFrame.

    ``markets``: optional list of ``market`` labels to keep (read with a Parquet filter, so the
    other markets never enter memory: 45 minutes of 4 markets is ~7.4M rows, ~1 GB in pandas;
    a 24 h run is ~40M rows). Text columns are returned as categoricals to save memory.
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*.parquet")) + sorted(out_dir.glob("parts/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No snapshot files in {out_dir}")
    filters = [("market", "in", list(markets))] if markets is not None else None
    frames = [pd.read_parquet(f, filters=filters) for f in files]
    df = _dedup(pd.concat(frames, ignore_index=True))
    return df.astype({c: "category" for c in CATEGORICAL_COLUMNS})


def collect(exchanges, markets, store, interval_s=10.0, n_rounds=None, duration_s=None,
            flush_every=1, max_weight_fraction=0.8, clock=now_ms, sleep=time.sleep, log=print):
    """Take a snapshot of every market each ``interval_s`` seconds and store them.

    ``exchanges``: dict exchange id -> ccxt client. Stops after ``n_rounds`` rounds or once
    ``duration_s`` has passed (one of them is required). Round ``k`` is scheduled at
    ``start + k * interval_s``. If a round ends after the next slot has passed (a retry, a
    weight pause, the computer sleeping), the missed slots are **skipped**, not caught up: the
    next round is the first slot still in the future, so ``round_id`` jumps and the gap stays
    visible (catching up would fire rounds back to back and over-weight that moment). A market
    that keeps failing after the retries is logged and skipped for that round (its snapshot is
    missing, not filled); an HTTP 418 ban stops the collection. Rounds are flushed to ``store``
    every ``flush_every`` rounds and at the end (also on Ctrl+C or an error). Locked files
    (Windows) are retried for ~15 s (``_retry_if_locked``); a write that still fails keeps the
    snapshots in memory for the next round, and a failed merge of an hour file is retried later:
    neither stops the collection.

    Returns a summary dict: rounds, skipped slots, snapshots, failures, max weight seen per
    exchange. It is also logged at the end, even after Ctrl+C or an error.
    """
    if n_rounds is None and duration_s is None:
        raise ValueError("give n_rounds or duration_s")
    start_ms = clock()
    buffer, summary = [], {"rounds": 0, "skipped_slots": 0, "snapshots": 0, "failures": 0,
                           "max_used_weight": {}, "start_ms": start_ms}

    def flush():
        if buffer:
            store.append(pd.concat(buffer, ignore_index=True))
            buffer.clear()

    try:
        k = 0
        while True:
            if n_rounds is not None and k >= n_rounds:
                break
            scheduled = start_ms + k * interval_s * 1000
            if duration_s is not None and scheduled - start_ms >= duration_s * 1000:
                break
            wait = (scheduled - clock()) / 1000
            if wait > 0:
                sleep(wait)
            for market in markets:
                exchange = exchanges[market.exchange_id]
                try:
                    snap = fetch_snapshot(exchange, market, clock=clock, sleep=sleep, log=log)
                except IPBanned:
                    raise
                except Exception as error:  # log, skip this market this round, keep going
                    summary["failures"] += 1
                    log(f"round {k}: {market.label} failed: {type(error).__name__}: "
                        f"{str(error)[:200]}")
                    continue
                buffer.append(snapshot_to_frame(snap, market, k))
                summary["snapshots"] += 1
                used = used_weight(exchange)
                if used is not None:
                    prev = summary["max_used_weight"].get(market.exchange_id, 0)
                    summary["max_used_weight"][market.exchange_id] = max(prev, used)
                    pause = weight_pause_s(used, WEIGHT_LIMIT_PER_MINUTE[market.exchange_id],
                                           clock(), max_weight_fraction)
                    if pause:
                        log(f"weight {used} on {market.exchange_id}: pausing {pause:.0f} s")
                        sleep(pause)
            summary["rounds"] += 1
            if summary["rounds"] % flush_every == 0:
                try:
                    flush()
                except Exception as error:  # e.g. a file still locked after the retries
                    summary["failed_flushes"] = summary.get("failed_flushes", 0) + 1
                    log(f"writing to disk failed ({type(error).__name__}: {str(error)[:200]}); "
                        f"{len(buffer)} snapshot(s) kept in memory, will retry next round")
                try:
                    store.compact_finished_hours(pd.to_datetime(clock(), unit="ms", utc=True))
                except Exception as error:  # e.g. Windows: file open in another program
                    log(f"merging finished hours failed ({type(error).__name__}: "
                        f"{str(error)[:200]}); will retry later")
            next_slot = math.ceil((clock() - start_ms) / (interval_s * 1000))
            if next_slot > k + 1:
                summary["skipped_slots"] += next_slot - (k + 1)
                log(f"round {k} ended late: skipping {next_slot - (k + 1)} missed slot(s)")
            k = max(k + 1, next_slot)
    finally:
        try:
            flush()
        except Exception as error:
            log(f"final write to disk failed ({type(error).__name__}: {str(error)[:200]}); "
                f"{len(buffer)} snapshot(s) lost")
        summary["end_ms"] = clock()
        log(f"collection summary: {summary}")
    return summary


def make_exchanges(exchange_ids=("binance", "binanceusdm")):
    """Public (no API key) ccxt clients **without** ccxt's own rate limiter.

    Why off: ccxt's limiter waits *inside* the call (about 2.5 s before each 5000-level spot
    request), after we read our clock, so ``local_request_utc`` would be up to seconds earlier
    than the real snapshot, and the 4 snapshots of a round would spread over ~5 s (measured on
    2026-11-19, first attempt of notebook 06's session). The budget is kept by ``collect``
    instead: a round costs 750 of the 6000 spot weight per minute (4500 per minute at one round
    every 10 s), the ``x-mbx-used-weight-1m`` header is read after every call with a pause at
    80% of the budget, HTTP 429 answers wait at least 60 s and HTTP 418 (ban) stops the
    collection (``fetch_with_retry``).
    """
    import ccxt

    return {eid: getattr(ccxt, eid)({"enableRateLimit": False, "timeout": 30_000})
            for eid in exchange_ids}
