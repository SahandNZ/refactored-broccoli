import numpy as np
import pandas as pd

from src.data._read import bars_file
from src.data._store import write_all_timeframes
from src.enums import Market, Timeframe


def _hourly_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC", name="open_time")
    close = 100 + np.arange(48.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
            "n_bars": 1,
        },
        index=index,
    )


def test_a_1h_base_writes_1h_and_coarser_only(tmp_path):
    write_all_timeframes(
        _hourly_bars(), data_dir=tmp_path, market=Market.STOCK, symbol="SAP.DE", base=Timeframe.H1
    )
    written = {tf for tf in Timeframe if bars_file(tmp_path, Market.STOCK, "SAP.DE", tf).exists()}
    assert Timeframe.M15 not in written
    assert Timeframe.M30 not in written
    assert {Timeframe.H1, Timeframe.H4, Timeframe.D1} <= written
    hourly = pd.read_parquet(bars_file(tmp_path, Market.STOCK, "SAP.DE", Timeframe.H1))
    assert len(hourly) == 48
