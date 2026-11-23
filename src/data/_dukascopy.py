import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from src import LAB_DIR
from src.data._resample import add_mid_and_spread
from src.enums import Timeframe
from src.utils import ms_to_utc

logger = logging.getLogger(__name__)

CLI = LAB_DIR / "vendor" / "dukascopy" / "node_modules" / ".bin" / "dukascopy-node"
"""The pinned dukascopy-node command line tool (see ``vendor/dukascopy/package.json``)."""

SIDES = ("bid", "ask")

CLI_TIMEFRAMES = {Timeframe.M15: "m15", Timeframe.H1: "h1"}
"""The tool's name of each timeframe we download (15m needs one request per day, 1h one per
month)."""


def part_path(parts_dir: Path, side: str, year: int) -> Path:
    """Return the raw CSV of one side and year in an instrument's raw folder (``raw/<source>``)."""
    return parts_dir / f"{side}-{year}.csv"


def download_part(
    code: str,
    side: str,
    year: int,
    *,
    date_from: str,
    date_to: str,
    parts_dir: Path,
    retries: int,
    timeframe: Timeframe = Timeframe.M15,
) -> Path:
    """Download one year of candles of one price side, unless already on disk.

    The tool writes into a temporary folder and the file is moved into place only when
    complete, so an interrupted run is resumed by running it again.

    Parameters
    ----------
    code : str
        Dukascopy instrument code, e.g. ``"eurusd"``.
    side : str
        ``"bid"`` or ``"ask"``.
    year : int
        The calendar year of the part (its file name).
    date_from, date_to : str
        UTC dates of the part, ``[date_from, date_to)`` (from ``src.utils.years_between``).
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``).
    retries : int
        Attempts after the first one, with an exponential pause (10 s, 20 s, ...).
    timeframe : Timeframe, optional
        Candle length, 15m or 1h (`CLI_TIMEFRAMES`).

    Returns
    -------
    Path
        The CSV file (columns timestamp in ms, open, high, low, close, volume).

    Raises
    ------
    RuntimeError
        If every attempt failed.
    """
    target = part_path(parts_dir, side, year)
    if target.exists():
        return target
    command = [
        str(CLI), "-i", code, "-from", date_from, "-to", date_to,
        "-t", CLI_TIMEFRAMES[timeframe], "-p", side,
        "-v", "-vu", "units", "-f", "csv", "-s", "-r", "3", "-bs", "5", "-bp", "1000",
        "--waf-mode", "off", "--rate-limit-retries", "8", "--request-interval", "250",
        "-fn", "part",
    ]  # fmt: skip
    parts_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory(dir=parts_dir) as folder:
            done = subprocess.run(  # noqa: S603 (fixed command, no shell, no user input)
                [*command, "-dir", folder], capture_output=True, text=True, check=False
            )
            produced = Path(folder) / "part.csv"
            if done.returncode == 0 and produced.exists():
                shutil.move(produced, target)
                return target
        pause = 10 * 2**attempt
        logger.warning(
            "%s %s %d failed (attempt %d, exit %d): %s; retrying in %d s",
            code, side, year, attempt + 1, done.returncode, done.stderr.strip()[-200:], pause,
        )  # fmt: skip
        time.sleep(pause)
    raise RuntimeError(f"Dukascopy download failed for {code} {side} {year}")


def _read_side(parts_dir: Path, side: str) -> pd.DataFrame:
    files = sorted(parts_dir.glob(f"{side}-*.csv"))
    if not files:
        raise FileNotFoundError(f"no {side} parts in {parts_dir}")
    frames = [pd.read_csv(file) for file in files if file.stat().st_size > 0]
    frame = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    frame.index = ms_to_utc(frame.pop("timestamp"))
    return frame.add_prefix(f"{side}_")


def build_bars(parts_dir: Path) -> pd.DataFrame:
    """Join the downloaded bid and ask parts of one instrument into bars (15m or 1h).

    Parameters
    ----------
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``) with ``bid-<year>.csv`` and
        ``ask-<year>.csv``.

    Returns
    -------
    pd.DataFrame
        Bars indexed by UTC open time with ``bid_*`` and ``ask_*`` OHLC and volumes,
        ``mid_*`` OHLC, ``spread`` and ``n_bars`` (1). Only bars with both sides are kept;
        bars without volume (a closed market) are not in the source at all.
    """
    bars = _read_side(parts_dir, "bid").join(_read_side(parts_dir, "ask"), how="inner")
    bars["n_bars"] = 1
    bars.index.name = "open_time"
    return add_mid_and_spread(bars)
