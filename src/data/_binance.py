import hashlib
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import days_of_month, in_window, ms_to_utc, write_parquet_atomic

logger = logging.getLogger(__name__)

URL_ROOT = "https://data.binance.vision/data/futures/um"
"""Binance public data, USD-M futures."""

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "amount", "trades",
    "taker_buy_volume", "taker_buy_amount", "ignore",
]  # fmt: skip
_KEEP = ["open", "high", "low", "close", "volume", "amount", "trades", "taker_buy_volume"]


def _fetch(url: str, *, retries: int) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https host)
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        pause = 10 * 2**attempt
        logger.warning("%s failed (%s), retrying in %d s", url, reason, pause)
        time.sleep(pause)
    raise RuntimeError(f"download failed after {retries + 1} attempts: {url}")


def _fetch_verified(url: str, *, retries: int) -> pd.DataFrame | None:
    archive = _fetch(url, retries=retries)
    if archive is None:
        return None
    checksum = _fetch(url + ".CHECKSUM", retries=retries)
    if checksum is not None and hashlib.sha256(archive).hexdigest() != checksum.split()[0].decode():
        raise RuntimeError(f"checksum mismatch: {url}")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        raw = zipped.read(zipped.namelist()[0])
    # newer files start with a header row, older ones do not
    has_header = raw[:9] == b"open_time"
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=_COLUMNS)


def download_month(symbol: str, month: str, *, parts_dir: Path, retries: int) -> Path:
    """Download one month of 15m klines (checksum verified), unless already on disk.

    Whole days missing inside the month are taken from the daily files. A month without a
    monthly file (before the listing or after a delisting) gives an empty part, so it is not
    asked for again.

    Parameters
    ----------
    symbol : str
        Binance symbol, e.g. ``"BTCUSDT"``.
    month : str
        ``"YYYY-MM"``.
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``).
    retries : int
        Attempts after the first one for each file.

    Returns
    -------
    Path
        The part file (parquet, raw kline columns).
    """
    target = parts_dir / f"{month}.parquet"
    if target.exists():
        return target
    name = f"{symbol}-15m-{month}"
    frame = _fetch_verified(f"{URL_ROOT}/monthly/klines/{symbol}/15m/{name}.zip", retries=retries)
    if frame is None:
        # no monthly file: the symbol was not listed (or already delisted) that month
        write_parquet_atomic(pd.DataFrame(columns=_COLUMNS), target)
        return target
    present = set(ms_to_utc(frame["open_time"]).normalize())
    frames = [frame]
    for day in days_of_month(month):
        if day in present or day < min(present) or day > max(present):
            continue
        url = f"{URL_ROOT}/daily/klines/{symbol}/15m/{symbol}-15m-{day:%Y-%m-%d}.zip"
        daily = _fetch_verified(url, retries=retries)
        if daily is not None:
            logger.info("%s: filled %s from the daily file", symbol, f"{day:%Y-%m-%d}")
            frames.append(daily)
    result = pd.concat(frames)
    write_parquet_atomic(result, target)
    return target


def build_bars(parts_dir: Path, *, start: str, end: str) -> pd.DataFrame:
    """Join the monthly parts of one symbol into 15m bars.

    Parameters
    ----------
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``) with ``YYYY-MM.parquet`` files.
    start, end : str
        UTC dates; bars opening in ``[start, end)`` are kept.

    Returns
    -------
    pd.DataFrame
        15m bars indexed by UTC open time: open, high, low, close, volume, amount (quote
        volume), trades, taker_buy_volume and ``n_bars`` (1). A bar without a trade (volume
        and trades 0) is left out: Binance repeats the last price in such bars, during an
        outage and for months after a coin is delisted, and a closed market has no bar.
    """
    frames = [pd.read_parquet(file) for file in sorted(parts_dir.glob("*.parquet"))]
    raw = pd.concat([frame for frame in frames if len(frame)])
    raw = raw.drop_duplicates("open_time").sort_values("open_time")
    bars = raw[_KEEP].astype("float64")
    times = ms_to_utc(raw["open_time"])
    bars.index = times
    bars["n_bars"] = 1
    traded = (bars["volume"] > 0) | (bars["trades"] > 0)
    return bars[traded & in_window(times, start=start, end=end)]
