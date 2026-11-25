"""Offline tests of src/data/binance_vision.py (no network)."""

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from src.data import binance_vision as bv


def make_zip(path, name, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    path.write_bytes(buf.getvalue())
    return path


def test_paths():
    assert bv.book_depth_path("BTCUSDT", "2026-11-22") == \
        "futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2026-11-22.zip"
    assert bv.spot_klines_path("ETHUSDT", "1m", "2026-10") == \
        "spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2026-10.zip"
    assert bv.spot_klines_path("ETHUSDT", "1m", "2026-11-03") == \
        "spot/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2026-11-03.zip"


def test_klines_periods_full_months_and_partial_edges():
    p = bv.klines_periods("2025-11-28", "2026-02-02")
    assert p == ["2025-11-28", "2025-11-29", "2025-11-30", "2025-12", "2026-01",
                 "2026-02-01", "2026-02-02"]
    # a month that ends exactly on the end day is a full month
    assert bv.klines_periods("2026-01-01", "2026-01-31") == ["2026-01"]


def test_timestamp_unit():
    assert bv.timestamp_unit([1761955200000000, 1761955260000000]) == "us"
    assert bv.timestamp_unit([1761955200000]) == "ms"
    with pytest.raises(ValueError):
        bv.timestamp_unit([1761955200000, 1761955260000000])     # mixed units
    with pytest.raises(ValueError):
        bv.timestamp_unit([1761955200])                          # seconds: not expected


def test_read_klines_zip_microseconds(tmp_path):
    rows = ("1761955200000000,100,110,90,105,2,1761955259999999,210,5,1,105,0\n"
            "1761955260000000,105,106,104,104,1,1761955319999999,104,3,0.5,52,0\n")
    z = make_zip(tmp_path / "k.zip", "k.csv", rows)
    df = bv.read_klines_zip(z)
    assert df.index[0] == pd.Timestamp("2025-11-01 00:00", tz="UTC")
    assert df.index[1] - df.index[0] == pd.Timedelta(minutes=1)
    assert df["close"].tolist() == [105.0, 104.0]
    assert "ignore" not in df.columns


def test_read_klines_zip_with_header_and_ms(tmp_path):
    rows = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
            "taker_buy_volume,taker_buy_quote_volume,ignore\n"
            "1761955200000,100,110,90,105,2,1761955259999,210,5,1,105,0\n")
    df = bv.read_klines_zip(make_zip(tmp_path / "k.zip", "k.csv", rows))
    assert len(df) == 1 and df.index[0] == pd.Timestamp("2025-11-01", tz="UTC")


def test_read_book_depth_zip(tmp_path):
    text = ("timestamp,percentage,depth,notional\n"
            "2026-11-22 00:00:01,-0.20,10.5,630000.0\n"
            "2026-11-22 00:00:01,0.20,9.5,571000.0\n")
    df = bv.read_book_depth_zip(make_zip(tmp_path / "b.zip", "b.csv", text))
    assert list(df.columns) == ["timestamp_utc", "percentage", "depth", "notional"]
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2026-11-22 00:00:01", tz="UTC")


def test_read_book_depth_rejects_other_columns(tmp_path):
    z = make_zip(tmp_path / "b.zip", "b.csv", "time,pct,qty\n1,2,3\n")
    with pytest.raises(ValueError):
        bv.read_book_depth_zip(z)


def test_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"zip bytes"
    good = hashlib.sha256(payload).hexdigest()
    answers = {}

    def fake_get(url, timeout=60):
        return answers[url.endswith(".CHECKSUM")]

    monkeypatch.setattr(bv, "_get", fake_get)
    answers.update({False: payload, True: f"{good}  f.zip".encode()})
    out = bv.download("x/f.zip", tmp_path)
    assert out.read_bytes() == payload

    answers.update({True: f"{'0' * 64}  g.zip".encode()})
    with pytest.raises(ValueError):
        bv.download("x/g.zip", tmp_path)
    assert not (tmp_path / "g.zip").exists() and not (tmp_path / "g.zip.tmp").exists()
