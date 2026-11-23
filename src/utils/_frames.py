import os
from pathlib import Path

import pandas as pd

OHLC_FIELDS = ("open", "high", "low", "close")
"""The four price fields of a bar, in order."""


def ohlc_columns(prefix: str = "") -> list[str]:
    """Return the OHLC column names with an optional prefix.

    Examples
    --------
    >>> ohlc_columns("bid_")
    ['bid_open', 'bid_high', 'bid_low', 'bid_close']
    """
    return [f"{prefix}{field}" for field in OHLC_FIELDS]


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise if `frame` misses any of `columns`.

    Raises
    ------
    ValueError
        Naming the missing columns.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing}, got {list(frame.columns)}")


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write `frame` to `path` as parquet through a temporary file.

    A crash never leaves a half-written file behind, so a run can be resumed by checking which
    files exist.

    Parameters
    ----------
    frame : pd.DataFrame
        The table to write (its index is kept).
    path : Path
        The final file; its folder is created when missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)
