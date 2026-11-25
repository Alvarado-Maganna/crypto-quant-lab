"""Download and read Binance's free public data files (data.binance.vision), notebook 08.

Two datasets are used:

- **USD-M ``bookDepth``** (``data/futures/um/daily/bookDepth/<SYMBOL>/``): one zip per UTC day
  with one CSV ``timestamp, percentage, depth, notional``. Checked on 2026-11-24: a row every
  ~30 s for each of 12 percentages (-5, -4, -3, -2, -1, -0.2, +0.2, +1 ... +5), 34,560 rows per
  day. ``timestamp`` is text (``"2026-11-22 00:00:01"``) **without a time zone**. Binance does
  not document this file; notebook 08 tests what the columns mean (cumulative or per band, sign,
  units, time zone) before using them.
- **Spot ``klines``** (``data/spot/{monthly,daily}/klines/<SYMBOL>/<interval>/``): candles, no
  header, 12 columns (``KLINE_COLUMNS``). Checked on 2026-11-24: since 2025 the spot files give
  ``open_time`` / ``close_time`` in Unix **microseconds** (16 digits, e.g. 1761955200000000 =
  2025-11-01 00:00 UTC); older files used milliseconds. ``read_klines_zip`` reads the unit from the
  number of digits and refuses anything else.

Every zip has a ``.CHECKSUM`` file next to it (``"<sha256>  <file name>"``); ``download`` checks
it and deletes the zip if it does not match. Files are written to a temporary name and renamed,
so an interrupted download never leaves a half file that looks complete.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE_URL = "https://data.binance.vision/data"
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
BOOK_DEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]


def book_depth_path(symbol, day):
    """Path (under ``BASE_URL``) of the USD-M bookDepth zip of one UTC day."""
    day = pd.Timestamp(day).strftime("%Y-%m-%d")
    return f"futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{day}.zip"


def spot_klines_path(symbol, interval, period):
    """Path of a spot klines zip. ``period``: 'YYYY-MM' (monthly file) or 'YYYY-MM-DD' (daily)."""
    kind = "monthly" if len(period) == 7 else "daily"
    return f"spot/{kind}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"


def futures_klines_path(symbol, interval, period):
    """Path of a USD-M futures klines zip (trade prices, 'last'). These CSVs have a header and
    millisecond timestamps (checked 2026-11-24); ``read_klines_zip`` handles both."""
    kind = "monthly" if len(period) == 7 else "daily"
    return f"futures/um/{kind}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"


def klines_periods(start, end):
    """Monthly files for every full month in [start, end], daily files for the rest.

    ``start``/``end`` are UTC days (inclusive). A month counts as full only if it ends on or
    before ``end``; the trailing partial month (whose monthly file is not published yet) is
    covered with daily files. Returns a list of 'YYYY-MM' and 'YYYY-MM-DD' strings.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out, day = [], start
    while day <= end:
        month_end = day + pd.offsets.MonthEnd(0)
        if day.day == 1 and month_end <= end:
            out.append(day.strftime("%Y-%m"))
            day = month_end + pd.Timedelta(days=1)
        else:
            out.append(day.strftime("%Y-%m-%d"))
            day += pd.Timedelta(days=1)
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-quant-lab/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download(path, dest_dir, retries=4, pause_s=2.0, sleep=time.sleep):
    """Download ``BASE_URL/path`` into ``dest_dir`` (same file name) and verify its checksum.

    Returns the local path. Skips the download if the file is already there (it was verified
    when it was written). Raises ``FileNotFoundError`` on HTTP 404 (file not published) and
    ``ValueError`` if the checksum does not match.
    """
    dest_dir = Path(dest_dir)
    dest = dest_dir / Path(path).name
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{path}"
    for attempt in range(retries + 1):
        try:
            data = _get(url)
            expected = _get(url + ".CHECKSUM").decode().split()[0].lower()
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(f"not published: {url}") from error
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        sleep(pause_s * 2 ** attempt)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    if _sha256(tmp) != expected:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}")
    os.replace(tmp, dest)
    return dest


def _read_single_csv(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zip_path}: expected one CSV, found {names}")
        return z.read(names[0])


def read_book_depth_zip(zip_path):
    """Raw bookDepth rows of one day: ``timestamp_utc`` (parsed **as UTC**, an assumption tested
    in notebook 08), ``percentage``, ``depth``, ``notional``. Nothing else is changed."""
    df = pd.read_csv(io.BytesIO(_read_single_csv(zip_path)))
    if list(df.columns) != BOOK_DEPTH_COLUMNS:
        raise ValueError(f"{zip_path}: unexpected columns {list(df.columns)}")
    df.insert(0, "timestamp_utc", pd.to_datetime(df.pop("timestamp"), format="%Y-%m-%d %H:%M:%S",
                                                 utc=True).astype("datetime64[ms, UTC]"))
    return df


def timestamp_unit(values):
    """'us' for 16-digit Unix timestamps (microseconds), 'ms' for 13 digits. Raises otherwise."""
    digits = {len(str(int(v))) for v in pd.Series(values).dropna()}
    if digits == {16}:
        return "us"
    if digits == {13}:
        return "ms"
    raise ValueError(f"cannot tell the unit of timestamps with {sorted(digits)} digits")


def read_klines_zip(zip_path):
    """Candles of one spot klines zip, indexed by ``open_time_utc`` (UTC).

    The unit of ``open_time`` is read from its number of digits (``timestamp_unit``): a file
    mixing units, or any other length, raises. A header row, if present, is dropped.
    """
    raw = pd.read_csv(io.BytesIO(_read_single_csv(zip_path)), header=None, names=KLINE_COLUMNS)
    if not str(raw.iloc[0, 0]).strip().isdigit():          # header line (some newer files)
        raw = raw.iloc[1:]
    raw = raw.astype({c: "int64" for c in ("open_time", "close_time", "trades")}) \
             .astype({c: "float64" for c in ("open", "high", "low", "close", "volume",
                                             "quote_volume", "taker_buy_base",
                                             "taker_buy_quote")})
    unit = timestamp_unit(raw["open_time"])
    raw.index = pd.to_datetime(raw["open_time"], unit=unit, utc=True)
    raw.index.name = "open_time_utc"
    return raw.drop(columns=["ignore"])
