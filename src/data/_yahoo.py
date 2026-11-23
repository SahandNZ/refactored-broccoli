import logging
import time
from pathlib import Path

import pandas as pd

from src.utils import to_utc, write_parquet_atomic

logger = logging.getLogger(__name__)


def download_reference(ticker: str, *, start: str, end: str, folder: Path, retries: int) -> Path:
    """Download Yahoo daily closes and the split list of one stock, unless already on disk.

    They are a reference only: the splits adjust the Dukascopy stock prices (which are not
    split-adjusted) and the daily closes check our own daily closes. ``Close`` with
    ``auto_adjust=False`` is adjusted for splits only, not for dividends.

    Parameters
    ----------
    ticker : str
        Yahoo ticker, e.g. ``"NVDA"`` or ``"SAP.DE"``.
    start, end : str
        UTC dates of the daily closes, ``[start, end)``.
    folder : Path
        ``data/stock/<ticker>/reference/yahoo``; it receives ``1d.pq`` and ``splits.pq``.
    retries : int
        Attempts after the first one, with exponential pauses.

    Returns
    -------
    Path
        The folder.

    Raises
    ------
    RuntimeError
        If Yahoo returns no daily data after every attempt.
    """
    import yfinance as yf  # noqa: PLC0415 (heavy optional dependency, python.md 12)

    if (folder / "1d.pq").exists() and (folder / "splits.pq").exists():
        return folder
    for attempt in range(retries + 1):
        stock = yf.Ticker(ticker)
        closes = stock.history(start=start, end=end, interval="1d", auto_adjust=False)
        if len(closes):
            splits = stock.splits
            daily = pd.DataFrame({"close": closes["Close"].to_numpy()}, index=closes.index.date)
            daily.index.name = "date"
            table = pd.DataFrame({"ratio": splits.to_numpy()}, index=splits.index.date)
            table.index.name = "date"
            table = table[
                (table.index >= to_utc(start).date()) & (table.index < to_utc(end).date())
            ]
            write_parquet_atomic(table, folder / "splits.pq")
            write_parquet_atomic(daily, folder / "1d.pq")
            return folder
        pause = 10 * 2**attempt
        logger.warning("Yahoo returned nothing for %s, retrying in %d s", ticker, pause)
        time.sleep(pause)
    raise RuntimeError(f"Yahoo has no daily data for {ticker}")
