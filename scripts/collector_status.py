"""Quick health check of a collector run: is it writing, are rounds missing, any errors?

Read-only (safe while ``collect_order_book.py`` is running). Shows:
- state of the run (from the log) and its start line;
- snapshots per UTC hour and market, complete rounds vs expected;
- UTC minutes without any snapshot (gaps), grouped into ranges;
- errors and warnings from ``collector.log``, counted by kind, and the latest ones;
- time of the newest file written, size of the run, free disk space and a rough estimate of
  the space the rest of the run needs.

Usage (from the repo root, with the .venv active):

    python scripts/collector_status.py data/collector/2026-11-21
    python scripts/collector_status.py data/collector/2026-11-21 --last-errors 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.order_book import default_markets  # noqa: E402
from src.data.run_status import (disk_report, log_events, missing_minutes,  # noqa: E402
                                 read_snapshots, run_state, snapshots_per_hour)

CDMX = "America/Mexico_City"


def both_clocks(ts):
    return f"{ts:%Y-%m-%d %H:%M:%S} UTC ({ts.tz_convert(CDMX):%a %H:%M} Mexico City)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="folder of the run, e.g. data/collector/2026-11-21")
    parser.add_argument("--interval-s", type=float, default=60.0,
                        help="interval of the run, in seconds (default 60)")
    parser.add_argument("--alt", default="PENGU/USDT",
                        help="altcoin of the run (default PENGU/USDT), to list every market")
    parser.add_argument("--last-errors", type=int, default=10,
                        help="how many of the latest log warnings to print (default 10)")
    parser.add_argument("--planned-min", type=float, default=48 * 60 + 20,
                        help="planned duration in minutes, for the disk estimate (default 2900)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.is_dir():
        parser.error(f"no such folder: {run_dir}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    now = pd.Timestamp.now(tz="UTC")

    state, start_line = run_state(run_dir / "collector.log")
    print(f"== {run_dir}\nnow: {both_clocks(now)}\nstate: {state}")
    if start_line:
        print(f"last start: {start_line}")

    snaps = read_snapshots(run_dir)
    print(f"\n== snapshots: {len(snaps):,}")
    if not snaps.empty:
        first, last = snaps["local_request_utc"].min(), snaps["local_request_utc"].max()
        print(f"first: {both_clocks(first)}\nlast:  {both_clocks(last)}  "
              f"({(now - last).total_seconds() / 60:.1f} min ago)")
        print(f"\n== snapshots per UTC hour (expected = rounds that fit in the covered part "
              f"of the hour at one every {args.interval_s:g} s)")
        markets = [m.label for m in default_markets(args.alt)]
        print(snapshots_per_hour(snaps, args.interval_s, markets).to_string())
        gaps = missing_minutes(snaps)
        print(f"\n== minutes without any snapshot: {int(gaps['minutes'].sum()) if len(gaps) else 0}"
              f" in {len(gaps)} gap(s)")
        if len(gaps):
            print(gaps.tail(20).to_string(index=False))

    events = log_events(run_dir / "collector.log")
    print(f"\n== log warnings and errors: {len(events)}")
    if len(events):
        print(events["kind"].value_counts().to_string())
        print(f"-- latest {args.last_errors}:")
        for line in events["line"].tail(args.last_errors):
            print(line)

    disk = disk_report(run_dir)
    print("\n== disk")
    for key, value in disk.items():
        if isinstance(value, pd.Timestamp):
            value = f"{both_clocks(value)}, {(now - value).total_seconds() / 60:.1f} min ago"
        elif isinstance(value, float):
            value = f"{value:,.2f}"
        print(f"{key}: {value}")
    if not snaps.empty:
        hours_done = (snaps["local_request_utc"].max() - snaps["local_request_utc"].min()) \
            .total_seconds() / 3600
        hours_left = max(args.planned_min / 60 - hours_done, 0)
        if hours_done > 0.25:
            need_gb = disk["run size (MB)"] / hours_done * hours_left / 1e3
            print(f"estimate for the remaining {hours_left:.1f} h of a {args.planned_min:g}-min "
                  f"run: ~{need_gb:.2f} GB more (free: {disk['free disk (GB)']:.1f} GB)")


if __name__ == "__main__":
    main()
