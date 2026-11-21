"""Tests for the long collector run: locked files, failed writes, keep-awake, defaults, status.

No network: fake exchanges and hand-made snapshots (helpers from tests/test_order_book.py).
"""

import pandas as pd
import pytest

import src.data.order_book as ob
from scripts import collect_order_book as script
from src.data.order_book import HourlyParquetStore, collect, load_snapshots
from src.data.run_status import (log_events, missing_minutes, read_snapshots, run_state,
                                 snapshots_per_hour)
from src.keep_awake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, keep_awake
from tests.test_order_book import PERP, SPOT, FakeClock, FakeExchange, frame

T_MS = int(pd.Timestamp("2026-11-21 23:58:10", tz="UTC").timestamp() * 1000)


# --- locked files (Windows) --------------------------------------------------------------------

def test_retry_if_locked_waits_then_succeeds():
    calls, pauses = [], []

    def action():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        return "ok"

    assert ob._retry_if_locked(action, pauses_s=(0.5, 1.0, 2.0), sleep=pauses.append) == "ok"
    assert pauses == [0.5, 1.0]


def test_retry_if_locked_gives_up_and_raises():
    pauses = []

    def action():
        raise PermissionError("in use")

    with pytest.raises(PermissionError):
        ob._retry_if_locked(action, pauses_s=(0.5, 1.0), sleep=pauses.append)
    assert pauses == [0.5, 1.0]


def test_atomic_write_leaves_no_tmp_when_rename_stays_locked(tmp_path, monkeypatch):
    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(ob.os, "replace", locked)
    with pytest.raises(PermissionError):
        ob._atomic_write_parquet(frame(), tmp_path / "x.parquet", sleep=lambda s: None)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_keeps_the_round_in_memory(tmp_path):
    class FlakyStore(HourlyParquetStore):
        def __init__(self, out_dir):
            super().__init__(out_dir)
            self.attempts = 0

        def append(self, df):
            self.attempts += 1
            if self.attempts == 1:
                raise PermissionError("file is open in another program")
            return super().append(df)

    clock, logs = FakeClock(T_MS), []
    summary = collect({"binance": FakeExchange()}, [SPOT], FlakyStore(tmp_path), interval_s=60,
                      n_rounds=3, clock=clock, sleep=clock.sleep, log=logs.append)
    assert summary["failed_flushes"] == 1
    assert any("writing to disk failed" in m for m in logs)
    assert sorted(load_snapshots(tmp_path)["round_id"].unique()) == [0, 1, 2]   # nothing lost


# --- keep-awake --------------------------------------------------------------------------------

def test_keep_awake_does_nothing_off_windows():
    calls = []
    with keep_awake(log=lambda m: None, platform="linux", set_state=calls.append) as on:
        assert on is False
    assert calls == []


def test_keep_awake_sets_and_restores_even_on_ctrl_c():
    calls = []
    with pytest.raises(KeyboardInterrupt):
        with keep_awake(log=lambda m: None, platform="win32", set_state=calls.append) as on:
            assert on is True
            raise KeyboardInterrupt
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


def test_keep_awake_failure_does_not_stop_the_job():
    def broken(flags):
        raise OSError("SetThreadExecutionState failed")

    logs = []
    with keep_awake(log=logs.append, platform="win32", set_state=broken) as on:
        assert on is False
    assert any("could not set it" in m for m in logs)


# --- script defaults ---------------------------------------------------------------------------

def test_script_defaults_are_the_planned_run():
    args = script.parse_args([])
    assert args.duration_min == 48 * 60 + 20 and args.interval_s == 60
    assert args.out_dir.startswith("data/collector/20")
    assert script.parse_args(["--duration-h", "1.5"]).duration_min == 90
    with pytest.raises(SystemExit):
        script.parse_args(["--duration-h", "1", "--duration-min", "60"])   # one or the other
    markets = ob.default_markets()
    assert [(m.label, m.limit) for m in markets] == [
        ("spot BTC/USDT", 5000), ("spot ETH/USDT", 5000), ("spot PENGU/USDT", 5000),
        ("perp BTC/USDT:USDT", 1000)]


# --- status of a run ---------------------------------------------------------------------------

def _run_folder(tmp_path):
    """Spot at 23:58, 23:59 and 00:01 UTC (00:00 missing); perp only at 23:58."""
    store = HourlyParquetStore(tmp_path)
    minute = 60_000
    store.append(pd.concat([frame(market=SPOT, request_ms=T_MS + k * minute, round_id=k)
                            for k in (0, 1, 3)] + [frame(market=PERP, request_ms=T_MS,
                                                         event_ms=T_MS)]))
    store.compact("20261121_23")       # one hour merged, the next one still in parts
    return tmp_path


def test_status_counts_snapshots_and_gaps(tmp_path):
    snaps = read_snapshots(_run_folder(tmp_path))
    assert len(snaps) == 4             # 3 spot + 1 perp, one row each
    per_hour = snapshots_per_hour(snaps, interval_s=60)
    h23 = per_hour.loc[pd.Timestamp("2026-11-21 23:00", tz="UTC")]
    assert h23["spot X/USDT"] == 2 and h23["perp X/USDT:USDT"] == 1 and h23["complete rounds"] == 1
    assert h23["expected"] == 2        # the run covered 23:58:10 to 00:00:00 of that hour
    gaps = missing_minutes(snaps)
    assert len(gaps) == 1 and gaps["minutes"].iloc[0] == 1
    assert gaps["from (UTC)"].iloc[0] == pd.Timestamp("2026-11-22 00:00", tz="UTC")


def test_log_events_and_state(tmp_path):
    log = tmp_path / "collector.log"
    log.write_text("\n".join([
        "2026-11-21 23:50:00 UTC  start: 2900 min ...",
        "2026-11-21 23:51:00 UTC  NetworkError on {...} (attempt 1): x; retrying in 2 s",
        "2026-11-21 23:52:00 UTC  round 418: spot BTC/USDT failed: RuntimeError: x",
        "2026-11-21 23:53:00 UTC  round 2 ended late: skipping 3 missed slot(s)",
        "2026-11-21 23:54:00 UTC  writing to disk failed (PermissionError: x); 4 snapshot(s) kept",
    ]) + "\n", encoding="utf-8")
    kinds = list(log_events(log)["kind"])
    assert kinds == ["retry", "request failed (snapshot missing)", "skipped slots",
                     "write/merge failed"]                 # "round 418" is not an IP ban
    assert run_state(log)[0].startswith("running")
    with log.open("a", encoding="utf-8") as f:
        f.write("2026-11-22 00:10:00 UTC  stopped by the user (Ctrl+C)\n")
    assert run_state(log)[0] == "stopped"
