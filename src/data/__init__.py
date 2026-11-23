"""Market data: download, bars of every timeframe and the manifest.

Downloads come from Dukascopy, Binance, Alpaca and Yahoo (references). Bars live at
``data/<market>/<symbol>/ohlcv/<timeframe>.pq``.
"""

from src.data._config import DataConfig, Instrument, QualityLimits, load_data_config
from src.data._download import build_instrument, download_all, refresh_manifest
from src.data._manifest import describe_bars
from src.data._quality import check_quality
from src.data._read import bars_file, read_bars, read_manifest
from src.data._resample import add_mid_and_spread, resample_bars
from src.data._store import write_all_timeframes

__all__ = [
    "DataConfig",
    "Instrument",
    "QualityLimits",
    "add_mid_and_spread",
    "bars_file",
    "build_instrument",
    "check_quality",
    "describe_bars",
    "download_all",
    "load_data_config",
    "read_bars",
    "read_manifest",
    "refresh_manifest",
    "resample_bars",
    "write_all_timeframes",
]
