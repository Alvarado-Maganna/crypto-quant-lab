"""Download the data.binance.vision files used by notebook 08 (checksums verified).

- USD-M BTCUSDT ``bookDepth``, one zip per UTC day: 2025-11-24 -> 2026-11-23 (the 12-month
  window), 2025-10-06 -> 2025-10-15 (the 10 Oct 2025 stress case) and 2026-11-19 (notebook 06).
- Spot BTCUSDT and ETHUSDT 1-minute klines over the same 12 months (monthly zips, daily zips for
  the last partial month); spot PENGUUSDT 1-minute klines of the collector's two full days.
- USD-M BTCUSDT 1-minute klines (trade prices) from 2026-08-01 to 2026-11-23, to turn bookDepth's
  USDT depth into BTC.

Files go to ``data/binance_vision/<same path as on the site>``; files already there are skipped,
so it can be re-run after an interruption. Usage (repo root, .venv active)::

    python scripts/download_binance_vision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.binance_vision import (book_depth_path, download,  # noqa: E402
                                     futures_klines_path, klines_periods, spot_klines_path)

OUT = REPO_ROOT / "data" / "binance_vision"
WINDOW = ("2025-11-24", "2026-11-23")
STRESS_10_10 = ("2025-10-06", "2025-10-15")
PERP_KLINES = ("2026-08-01", "2026-11-23")
PENGU_DAYS = ("2026-11-22", "2026-11-23")
EXTRA_DAYS = ["2026-11-19"]


def planned_paths():
    days = list(pd.date_range(*WINDOW, freq="D")) + list(pd.date_range(*STRESS_10_10, freq="D")) \
        + [pd.Timestamp(d) for d in EXTRA_DAYS]
    paths = sorted({book_depth_path("BTCUSDT", d) for d in days})
    for symbol in ("BTCUSDT", "ETHUSDT"):
        paths += [spot_klines_path(symbol, "1m", p) for p in klines_periods(*WINDOW)]
    paths += [spot_klines_path("PENGUUSDT", "1m", p) for p in klines_periods(*PENGU_DAYS)]
    paths += [futures_klines_path("BTCUSDT", "1m", p) for p in klines_periods(*PERP_KLINES)]
    return paths


def main():
    paths = planned_paths()
    missing, failed = [], []
    for i, path in enumerate(paths, 1):
        try:
            download(path, OUT / Path(path).parent)
        except FileNotFoundError:
            missing.append(path)
        except Exception as error:  # report and continue: a re-run picks it up
            failed.append(f"{path}: {type(error).__name__}: {error}")
        if i % 25 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} done", flush=True)
    print(f"not published ({len(missing)}): {missing}")
    print(f"failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
