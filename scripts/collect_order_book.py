"""Collect raw Binance order-book snapshots for many hours (same method as notebook 06).

Markets (public data, no API key): Binance spot BTC/USDT, ETH/USDT and one altcoin
(default PENGU/USDT, chosen in notebook 06), plus the USD-M perpetual BTC/USDT:USDT.
Every ``--interval-s`` seconds it takes one snapshot of each market (spot: 5000 levels per
side, USD-M: 1000, the maximum of each endpoint) and stores every level.

Output (see ``src/data/order_book.py`` for the schema):
- ``<out-dir>/YYYYMMDD_HH.parquet``: one file per UTC hour, written when the hour is over.
- ``<out-dir>/parts/YYYYMMDD_HH/part-*.parquet``: the hour in progress, one small file per flush.
  Files are written under a temporary name and then renamed, so an interruption (Ctrl+C, a crash,
  the computer going to sleep) never corrupts what is already saved. Restarting in the same folder
  is safe: old parts are merged into their hour file and duplicates are dropped.
- ``<out-dir>/collector.log``: start/end, every retried or failed request, rate-limit pauses.

Errors: network errors and rate limits are retried with growing waits (2, 4, 8 ... s, at
least 60 s after an HTTP 429/418). A market that still fails is skipped for that round and
logged; its snapshot is missing, never filled in. Weight per round: 750 (spot) + 20 (USD-M) of
the 6000 / 2400 allowed per minute; if a response header shows more than 80% used, it pauses
until the next minute.

Usage (from the repo root, with the .venv active):

    python scripts/collect_order_book.py                       # 24 h, one round per minute
    python scripts/collect_order_book.py --duration-h 2 --interval-s 60 --out-dir data/ob_test
    python scripts/collect_order_book.py --alt SOL/USDT        # a different altcoin

Keep the computer awake (no sleep mode) while it runs. Stop it any time with Ctrl+C.
Size: about 32,000 rows per round, i.e. roughly 46 million rows per day at one round per minute
(a few hundred MB of Parquet). ``data/`` is gitignored.

Load the result in Python with::

    from src.data.order_book import load_snapshots
    levels = load_snapshots("data/order_book_24h")
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.order_book import (HourlyParquetStore, collect, default_markets,  # noqa: E402
                                 make_exchanges, round_weight)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duration-h", type=float, default=24.0,
                        help="how long to run, in hours (default 24)")
    parser.add_argument("--interval-s", type=float, default=60.0,
                        help="seconds between rounds (default 60; minimum 10)")
    parser.add_argument("--out-dir", default="data/order_book_24h",
                        help="output folder (default data/order_book_24h)")
    parser.add_argument("--alt", default="PENGU/USDT",
                        help="spot altcoin pair (default PENGU/USDT, see notebook 06)")
    parser.add_argument("--flush-every", type=int, default=1,
                        help="write to disk every N rounds (default 1: lose at most 1 round)")
    args = parser.parse_args(argv)
    if args.interval_s < 10:
        parser.error("--interval-s must be at least 10 (one round uses 750 of the 6000 "
                     "spot weight per minute)")
    if args.duration_h <= 0:
        parser.error("--duration-h must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "collector.log", "a", encoding="utf-8")

    def log(message):
        line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    markets = default_markets(args.alt)
    store = HourlyParquetStore(out_dir)
    merged = store.compact_finished_hours(datetime.now(timezone.utc))
    log(f"start: {args.duration_h} h, every {args.interval_s} s, {len(markets)} markets "
        f"({', '.join(m.label for m in markets)}), weight per round {round_weight(markets)}, "
        f"out {out_dir}; merged {len(merged)} old hour(s)")
    try:
        summary = collect(make_exchanges(), markets, store, interval_s=args.interval_s,
                          duration_s=args.duration_h * 3600, flush_every=args.flush_every,
                          log=log)
        log(f"end: {json.dumps(summary)}")
    except KeyboardInterrupt:
        log("stopped by the user (Ctrl+C); everything flushed so far is saved")
    finally:
        store.compact_finished_hours(datetime(2100, 1, 1, tzinfo=timezone.utc))  # merge all
        log_file.close()


if __name__ == "__main__":
    main()
