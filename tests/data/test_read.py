import json

import pandas as pd
import pytest

from src.data._read import bars_file, read_bars, read_manifest
from src.enums import Market, Timeframe


def test_layout_and_round_trip(tmp_path):
    path = bars_file(tmp_path, Market.FX, "EURUSD", Timeframe.H1)
    assert path == tmp_path / Market.FX.value / "EURUSD" / "ohlcv" / "1h.pq"
    frame = pd.DataFrame(
        {"close": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")], name="open_time"),
    )
    path.parent.mkdir(parents=True)
    frame.to_parquet(path)
    assert read_bars(tmp_path, Market.FX, "EURUSD", Timeframe.H1).equals(frame)


def test_strings_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        bars_file(tmp_path, "forex", "EURUSD", Timeframe.H1)  # type: ignore[arg-type]


def test_read_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"instruments": [{"symbol": "BTCUSDT", "market": "crypto", "passed": True}]})
    )
    assert read_manifest(tmp_path).loc["BTCUSDT", "market"] == "crypto"
