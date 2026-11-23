import json
from pathlib import Path

import pandas as pd

from src.enums import Market, Timeframe

KIND = "ohlcv"
"""The data-kind folder of price bars: ``data/<market>/<symbol>/ohlcv/<timeframe>.pq``."""


def bars_file(data_dir: Path, market: Market, symbol: str, timeframe: Timeframe) -> Path:
    """Return the file of one instrument's bars, e.g. ``data/forex/EURUSD/ohlcv/1h.pq``.

    Raises
    ------
    TypeError
        If `market` or `timeframe` is not the enum.
    """
    if not isinstance(market, Market) or not isinstance(timeframe, Timeframe):
        raise TypeError(f"market and timeframe must be enums, got {market!r}, {timeframe!r}")
    return data_dir / market.value / symbol / KIND / f"{timeframe.value}.pq"


def read_bars(data_dir: Path, market: Market, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
    """Read one instrument's bars (UTC index of open times).

    Raises
    ------
    FileNotFoundError
        If the instrument or timeframe is not on disk.
    """
    return pd.read_parquet(bars_file(data_dir, market, symbol, timeframe))


def read_manifest(data_dir: Path) -> pd.DataFrame:
    """Read ``data/manifest.json``: one row per instrument with its source, quality and splits.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol.
    """
    with (data_dir / "manifest.json").open() as file:
        entries = json.load(file)["instruments"]
    return pd.DataFrame(entries).set_index("symbol")
