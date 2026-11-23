import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils import iso_to_utc, to_utc, write_parquet_atomic

logger = logging.getLogger(__name__)

URL = "https://data.alpaca.markets/v2/stocks/bars"
"""Alpaca market data API, historical stock bars (data only, no trading endpoint)."""

KEY_FILE = Path.home() / ".config" / "anonymous-benchmark" / "alpaca.env"
"""Where the API keys live, outside the repository (``APCA_API_KEY_ID=...`` lines)."""

_RENAME = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "n": "trades",
    "vw": "vwap",
}


def load_keys(path: Path = KEY_FILE) -> dict[str, str]:
    """Return the Alpaca key headers, from the environment or else from `path`.

    Parameters
    ----------
    path : Path, optional
        A file of ``NAME=value`` lines with ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY``.

    Returns
    -------
    dict of str to str
        The two request headers.

    Raises
    ------
    KeyError
        If either key is found nowhere.
    """
    values = dict(os.environ)
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            values.setdefault(name.strip(), value.strip())
    return {
        "APCA-API-KEY-ID": values["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": values["APCA_API_SECRET_KEY"],
    }


def bars_from_pages(pages: list[dict[str, object]], symbol: str) -> pd.DataFrame:
    """Turn the JSON pages of a bars request into one table.

    Parameters
    ----------
    pages : list of dict
        Decoded responses, each with ``{"bars": {symbol: [bar, ...]}}``.
    symbol : str
        The requested symbol.

    Returns
    -------
    pd.DataFrame
        Columns t (ISO time), open, high, low, close, volume, trades, vwap.
    """
    rows: list[dict[str, object]] = []
    for page in pages:
        bars = page.get("bars") or {}
        rows.extend(bars.get(symbol, []))  # type: ignore[union-attr]
    return pd.DataFrame(rows, columns=["t", *_RENAME]).rename(columns=_RENAME)


def _get(url: str, headers: dict[str, str], *, retries: int) -> dict[str, object]:
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)  # noqa: S310 (fixed https host)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
            if error.code in (401, 403, 404, 422):
                raise RuntimeError(f"Alpaca refused the request ({reason}): {url}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        pause = 10 * 2**attempt
        logger.warning("Alpaca request failed (%s), retrying in %d s", reason, pause)
        time.sleep(pause)
    raise RuntimeError(f"Alpaca request failed after {retries + 1} attempts")


def download_year(
    symbol: str, year: int, *, date_from: str, date_to: str, parts_dir: Path, retries: int
) -> Path:
    """Download one year of 15m bars (SIP feed, split-adjusted only), unless already on disk.

    Parameters
    ----------
    symbol : str
        US stock symbol, e.g. ``"META"``.
    year : int
        The calendar year of the part (its file name).
    date_from, date_to : str
        UTC dates of the part, ``[date_from, date_to)``.
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``).
    retries : int
        Attempts after the first one for each page.

    Returns
    -------
    Path
        The part file (parquet).
    """
    target = parts_dir / f"{year}.parquet"
    if target.exists():
        return target
    headers = load_keys()
    query = {
        "symbols": symbol, "timeframe": "15Min", "feed": "sip", "adjustment": "split",
        "start": f"{to_utc(date_from):%Y-%m-%dT%H:%M:%SZ}",
        "end": f"{to_utc(date_to):%Y-%m-%dT%H:%M:%SZ}", "limit": "10000",
    }  # fmt: skip
    pages: list[dict[str, object]] = []
    while True:
        page = _get(f"{URL}?{urllib.parse.urlencode(query)}", headers, retries=retries)
        pages.append(page)
        token = page.get("next_page_token")
        if not token:
            break
        query["page_token"] = str(token)
    write_parquet_atomic(bars_from_pages(pages, symbol), target)
    return target


def build_bars(parts_dir: Path, *, start: str, end: str) -> pd.DataFrame:
    """Join the yearly parts of one symbol into 15m bars.

    Parameters
    ----------
    parts_dir : Path
        The instrument's raw folder (``raw/<source>``) with ``<year>.parquet`` files.
    start, end : str
        UTC dates; bars opening in ``[start, end)`` are kept.

    Returns
    -------
    pd.DataFrame
        15m bars indexed by UTC open time: open, high, low, close, volume, trades, vwap and
        ``n_bars`` (1). Alpaca bars have no bid or ask, so there is no spread. Pre- and
        after-market bars of the SIP feed are kept.
    """
    frames = [pd.read_parquet(file) for file in sorted(parts_dir.glob("*.parquet"))]
    raw = pd.concat([frame for frame in frames if len(frame)]).drop_duplicates("t")
    bars = raw.drop(columns="t").astype("float64")
    bars.index = iso_to_utc(raw["t"])
    bars = bars.sort_index()
    bars["n_bars"] = 1
    return bars[(bars.index >= to_utc(start)) & (bars.index < to_utc(end))]
