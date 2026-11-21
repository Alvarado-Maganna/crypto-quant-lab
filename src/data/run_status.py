"""Health check of a collector run folder (``scripts/collect_order_book.py``), read-only.

Reads only what is needed: from every Parquet file (hour files and parts) the rows of level 0
(the best bid and best ask), i.e. two rows per snapshot, three columns. A 48 h run has ~80M
rows; this reads ~23k. It never writes, so it is safe while the collector is running. A part
file can disappear between listing and reading (the collector merged it into its hour file);
then the folder is listed again.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import pandas as pd

SNAPSHOT_KEY = ["market", "local_request_utc"]
# Log lines worth a look, by kind (first match wins). Matching is case-insensitive.
LOG_PATTERNS = {
    "IP ban (HTTP 418)": r"HTTP 418|IPBanned",
    "stopped": r"stopped by",
    "write/merge failed": r"(writing to disk|final write to disk|merging .*|final merge) failed",
    "request failed (snapshot missing)": r"round \d+: .* failed",
    "retry": r"retrying in",
    "weight pause": r"pausing",
    "skipped slots": r"skipping \d+ missed slot",
    "keep-awake problem": r"keep-awake: could not",
}


def parquet_files(run_dir):
    run_dir = Path(run_dir)
    return sorted(run_dir.glob("*.parquet")) + sorted(run_dir.glob("parts/*/*.parquet"))


def read_snapshots(run_dir, attempts=3):
    """One row per snapshot: market, local_request_utc, local_receive_utc."""
    for attempt in range(attempts):
        try:
            frames = [pd.read_parquet(f, columns=SNAPSHOT_KEY + ["local_receive_utc"],
                                      filters=[("level", "==", 0)])
                      for f in parquet_files(run_dir)]
            break
        except (FileNotFoundError, PermissionError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)  # the collector was merging: list the folder again
    if not frames:
        return pd.DataFrame(columns=SNAPSHOT_KEY + ["local_receive_utc"])
    df = pd.concat(frames, ignore_index=True)
    df["market"] = df["market"].astype(str)
    return df.drop_duplicates(SNAPSHOT_KEY).sort_values("local_request_utc") \
        .reset_index(drop=True)


def snapshots_per_hour(snaps, interval_s=60.0, markets=None):
    """Snapshots per market and UTC hour, plus ``complete rounds`` (the minimum across markets)
    and ``expected`` (rounds that fit in the part of the hour the run covered).

    ``markets``: the labels the run should have; a market that never succeeded still gets a
    column of zeros, so it pulls ``complete rounds`` down instead of vanishing."""
    if snaps.empty:
        return pd.DataFrame()
    hour = snaps["local_request_utc"].dt.floor("h")
    table = snaps.groupby([hour, "market"]).size().unstack("market", fill_value=0)
    if markets is not None:
        extra = [c for c in table.columns if c not in markets]
        table = table.reindex(columns=list(markets) + extra, fill_value=0)
    table.index.name = "hour (UTC)"
    table["complete rounds"] = table.min(axis=1)
    first, last = snaps["local_request_utc"].min(), snaps["local_request_utc"].max()
    covered = [(min(h + pd.Timedelta(hours=1), last + pd.Timedelta(seconds=interval_s))
                - max(h, first)).total_seconds() for h in table.index]
    table["expected"] = [int(round(c / interval_s)) for c in covered]
    return table


def missing_minutes(snaps):
    """UTC minutes between the first and the last snapshot with no snapshot of any market,
    grouped into ranges. Meaningful for intervals of up to 60 s (one round per minute)."""
    cols = ["from (UTC)", "to (UTC)", "minutes"]
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    seen = pd.DatetimeIndex(snaps["local_request_utc"].dt.floor("min").unique())
    every = pd.date_range(seen.min(), seen.max(), freq="min")
    missing = every.difference(seen)
    if missing.empty:
        return pd.DataFrame(columns=cols)
    s = pd.Series(missing)
    group = (s.diff() != pd.Timedelta(minutes=1)).cumsum()
    out = s.groupby(group).agg(["min", "max", "size"])
    out.columns = cols
    return out.reset_index(drop=True)


def log_events(log_path):
    """Log lines matching ``LOG_PATTERNS``, with their kind."""
    log_path = Path(log_path)
    rows = []
    if not log_path.exists():
        return pd.DataFrame(columns=["kind", "line"])
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for kind, pattern in LOG_PATTERNS.items():
            if re.search(pattern, line, flags=re.IGNORECASE):
                rows.append({"kind": kind, "line": line})
                break
    return pd.DataFrame(rows, columns=["kind", "line"])


def run_state(log_path):
    """'running', 'finished' or 'stopped' from the last start/end lines of the log.

    'running' only means the last start has no end line yet: if the process died (the window
    was closed, the computer restarted), the log looks the same; check the last file time too.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return "no log", None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if "  start: " in ln]
    if not starts:
        return "no start line", None
    last_start = lines[starts[-1]]
    after = lines[starts[-1] + 1:]
    if any("stopped by" in ln for ln in after):
        return "stopped", last_start
    if any("  end: " in ln for ln in after):
        return "finished", last_start
    return "running (or killed without a trace)", last_start


def disk_report(run_dir):
    """Size of the run on disk, newest file write, free space on that disk."""
    files = [f for f in Path(run_dir).rglob("*") if f.is_file()]
    parquet = [f for f in files if f.suffix == ".parquet"]
    newest = max(parquet, key=lambda f: f.stat().st_mtime) if parquet else None
    usage = shutil.disk_usage(run_dir)
    return {
        "run size (MB)": sum(f.stat().st_size for f in files) / 1e6,
        "parquet files": len(parquet),
        "newest file": newest.relative_to(run_dir).as_posix() if newest else None,
        "newest write (UTC)": pd.Timestamp(newest.stat().st_mtime, unit="s", tz="UTC")
        if newest else None,
        "free disk (GB)": usage.free / 1e9,
    }
