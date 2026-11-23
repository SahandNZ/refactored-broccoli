from pathlib import Path

import pandas as pd

from src.data._read import bars_file
from src.data._resample import resample_bars
from src.enums import Market, Timeframe
from src.utils import write_parquet_atomic


def write_all_timeframes(
    bars: pd.DataFrame,
    *,
    data_dir: Path,
    market: Market,
    symbol: str,
    base: Timeframe = Timeframe.M15,
) -> None:
    """Write the bars of one instrument and every coarser timeframe built from them.

    Files go to ``data/<market>/<symbol>/ohlcv/<timeframe>.pq`` (`bars_file`).

    Parameters
    ----------
    bars : pd.DataFrame
        Bars of the `base` timeframe with a UTC index of open times.
    data_dir : Path
        The lab's ``data/`` folder.
    market : Market
        The instrument's market.
    symbol : str
        The instrument's symbol (Yahoo ticker for stocks).
    base : Timeframe, optional
        The timeframe of `bars`; finer timeframes are not written.
    """
    for timeframe in Timeframe:
        if timeframe.minutes < base.minutes:
            continue
        frame = bars if timeframe is base else resample_bars(bars, timeframe)
        write_parquet_atomic(frame, bars_file(data_dir, market, symbol, timeframe))
