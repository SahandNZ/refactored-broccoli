import pandas as pd
import pytest

from src.data import _alpaca


def test_bars_from_pages_joins_pages_and_renames():
    first = {"t": "2020-01-02T14:30:00Z",
             "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "n": 3, "vw": 1.2}  # fmt: skip
    second = {"t": "2020-01-02T14:45:00Z",
              "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 5, "n": 2, "vw": 1.6}  # fmt: skip
    pages = [{"bars": {"META": [first]}}, {"bars": {"META": [second]}}, {"bars": None}]
    frame = _alpaca.bars_from_pages(pages, "META")
    assert list(frame.columns) == ["t", "open", "high", "low", "close", "volume", "trades", "vwap"]
    assert len(frame) == 2
    assert frame["close"].tolist() == pytest.approx([1.5, 1.8])


def test_build_bars_indexes_by_utc_and_cuts_the_period(tmp_path):
    pages = [{"bars": {"META": [
        {"t": "2019-12-31T23:45:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1},
        {"t": "2020-01-02T14:30:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 1, "n": 1, "vw": 2},
    ]}}]  # fmt: skip
    _alpaca.bars_from_pages(pages, "META").to_parquet(tmp_path / "2020.parquet")
    bars = _alpaca.build_bars(tmp_path, start="2020-01-01", end="2020-02-01")
    assert list(bars.index) == [pd.Timestamp("2020-01-02 14:30", tz="UTC")]
    assert (bars["n_bars"] == 1).all()


def test_load_keys_reads_the_file_when_the_environment_has_none(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    path = tmp_path / "alpaca.env"
    path.write_text("APCA_API_KEY_ID=id\nAPCA_API_SECRET_KEY=secret\n")
    assert _alpaca.load_keys(path) == {"APCA-API-KEY-ID": "id", "APCA-API-SECRET-KEY": "secret"}


def test_load_keys_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(KeyError):
        _alpaca.load_keys(tmp_path / "absent.env")
