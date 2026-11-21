"""Collect raw Binance order-book snapshots for many hours (same method as notebook 06).

Markets (public data, no API key): Binance spot BTC/USDT, ETH/USDT and one altcoin
(default PENGU/USDT, chosen in notebook 06), plus the USD-M perpetual BTC/USDT:USDT.
Every ``--interval-s`` seconds it takes one snapshot of each market (spot: 5000 levels per
side, USD-M: 1000, the maximum of each endpoint) and stores every level.

Defaults (the planned 48 h run): one round every 60 s for 48 h 20 min (2,900 min), output in
``data/collector/<UTC date of the start>/``. Change them with ``--interval-s``,
``--duration-min`` (or ``--duration-h``) and ``--out-dir``.

Output (see ``src/data/order_book.py`` for the schema, the same as notebook 06):
- ``<out-dir>/YYYYMMDD_HH.parquet``: one file per UTC hour, written when the hour is over.
- ``<out-dir>/parts/YYYYMMDD_HH/part-*.parquet``: the hour in progress, one small file per flush.
  Files are written under a temporary name and then renamed, so an interruption (Ctrl+C, a crash,
  the computer going to sleep) never corrupts what is already saved. Restarting in the same folder
  is safe: old parts are merged into their hour file and duplicates are dropped.
- ``<out-dir>/collector.log``: start/end, every retried or failed request, rate-limit pauses,
  skipped slots, failed writes. Check a run with ``python scripts/collector_status.py <out-dir>``.

Errors: network errors and rate limits are retried with growing waits (2, 4, 8 ... s, at
least 60 s after an HTTP 429/418). A market that still fails is skipped for that round and
logged; its snapshot is missing, never filled in. An HTTP 418 (IP ban) stops the run. If a
round ends late (e.g. after a retry or the computer sleeping), the missed slots are skipped, not
caught up, so there are never bursts of rounds. Weight per round: 750 (spot) + 20 (USD-M) of
the 6000 / 2400 allowed per minute; if a response header shows more than 80% used, it pauses
until the next minute plus 3 s. A file locked by another program (Windows) is retried for ~15 s;
if it is still locked, the round stays in memory and is written with the next one.

Usage (from the repo root, with the .venv active):

    python scripts/collect_order_book.py                        # 48 h 20 min, one round per minute
    python scripts/collect_order_book.py --duration-min 15 --out-dir data/collector/test-15min
    python scripts/collect_order_book.py --alt SOL/USDT         # a different altcoin

Sleep: on Windows the script itself asks Windows not to sleep on idle while it runs
(``src/keep_awake.py``) and restores the normal setting when it ends or is stopped. It cannot
stop sleep from closing the lid, the power button or an empty battery: keep the laptop plugged
in and open. Stop it any time with Ctrl+C (press it once, then wait for the "final merge done" line).
Size: about 27,000 rows per round (thin books return fewer than 5000 levels), i.e. roughly
40 million rows per day at one round per minute (a few hundred MB of Parquet). Loading a whole
day needs several GB of memory: use ``load_snapshots(folder, markets=[...])`` to read one market. ``data/`` is gitignored.

Load the result in Python with::

    from src.data.order_book import load_snapshots
    levels = load_snapshots("data/collector/2026-11-21", markets=["spot BTC/USDT"])
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
from src.keep_awake import keep_awake  # noqa: E402

DEFAULT_DURATION_MIN = 48 * 60 + 20      # 48 h 20 min


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--duration-min", type=float, default=None,
                          help=f"how long to run, in minutes (default {DEFAULT_DURATION_MIN} "
                               f"= 48 h 20 min)")
    duration.add_argument("--duration-h", type=float, default=None,
                          help="how long to run, in hours (alternative to --duration-min)")
    parser.add_argument("--interval-s", type=float, default=60.0,
                        help="seconds between rounds (default 60; minimum 10)")
    parser.add_argument("--out-dir", default=None,
                        help="output folder (default data/collector/<UTC date of the start>)")
    parser.add_argument("--alt", default="PENGU/USDT",
                        help="spot altcoin pair (default PENGU/USDT, see notebook 06)")
    parser.add_argument("--flush-every", type=int, default=1,
                        help="write to disk every N rounds (default 1: lose at most 1 round)")
    args = parser.parse_args(argv)
    if args.interval_s < 10:
        parser.error("--interval-s must be at least 10 (one round uses 750 of the 6000 "
                     "spot weight per minute)")
    if args.duration_h is not None:
        args.duration_min = args.duration_h * 60
    elif args.duration_min is None:
        args.duration_min = DEFAULT_DURATION_MIN
    if args.duration_min <= 0:
        parser.error("the duration must be positive")
    if args.out_dir is None:
        args.out_dir = f"data/collector/{datetime.now(timezone.utc):%Y-%m-%d}"
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
    try:  # a previous run in this folder may have left parts of finished hours
        merged = store.compact_finished_hours(datetime.now(timezone.utc))
    except Exception as error:
        log(f"merging old hours failed ({type(error).__name__}: {str(error)[:200]}); "
            f"will retry during the run")
        merged = []
    merged_names = [p.name for p in merged if p is not None]
    log(f"start: {args.duration_min:g} min ({args.duration_min / 60:.2f} h), every "
        f"{args.interval_s:g} s, {len(markets)} markets "
        f"({', '.join(m.label for m in markets)}), weight per round {round_weight(markets)}, "
        f"limits {[m.limit for m in markets]}, out {out_dir}; merged old hour(s): "
        f"{merged_names}")
    try:
        with keep_awake(log=log):
            summary = collect(make_exchanges(), markets, store, interval_s=args.interval_s,
                              duration_s=args.duration_min * 60, flush_every=args.flush_every,
                              log=log)
        log(f"end: {json.dumps(summary)}")
    except KeyboardInterrupt:
        log("stopped by the user (Ctrl+C); everything flushed so far is saved")
    except Exception as error:  # e.g. IPBanned: log it, merge what we have, then re-raise
        log(f"stopped by an error: {type(error).__name__}: {error}")
        raise
    finally:
        try:  # merge every hour, including the last (unfinished) one
            store.compact_finished_hours(datetime(2100, 1, 1, tzinfo=timezone.utc))
            log("final merge done: one Parquet file per UTC hour")
        except Exception as error:  # parts stay on disk and load_snapshots still reads them
            log(f"final merge failed ({type(error).__name__}: {str(error)[:200]}); the part "
                f"files are safe and are merged on the next start in this folder")
        log_file.close()


if __name__ == "__main__":
    main()
