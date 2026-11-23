"""Write the SHA-256 checksum of every data file into configs/data_checksums.csv.

The release cannot redistribute the raw prices (Dukascopy, Alpaca, Binance and Yahoo
Finance terms), so it gives these checksums instead: a user who rebuilds data/ with the
download scripts compares each file with its line. Most files are parquet written by our code
(pyarrow, pandas), so their bytes depend on the library versions of the locked environment
(uv.lock) as well as on the source; a mismatch flags a revised source or another library
version. Only hashes and sizes are written, never data. The download logs are left out.

Usage: .venv/bin/python scripts/write_checksums.py [--workers N]
"""

import argparse
import csv
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from src import LAB_DIR

logger = logging.getLogger(__name__)

DATA_DIR = LAB_DIR / "data"
"""The lab's data folder (read only)."""

OUT = LAB_DIR / "configs" / "data_checksums.csv"
"""The written checksum table (in git)."""

SKIPPED_SUFFIXES = (".log",)
"""File endings left out: the download logs, which change with every run."""

CHUNK = 1 << 20
"""Bytes read at a time."""


def file_checksum(path: Path, *, root: Path) -> tuple[str, int, str]:
    """Return a file's path relative to `root`, its size in bytes and its SHA-256 hex digest.

    Parameters
    ----------
    path : Path
        A file under `root`.
    root : Path
        The data folder the path is given relative to.

    Returns
    -------
    tuple of (str, int, str)
        The relative path (POSIX), the size and the digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest()


def data_files(root: Path) -> list[Path]:
    """Return every file under `root` except the skipped ones, in path order."""
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix not in SKIPPED_SUFFIXES
    )


def main() -> None:
    """Hash every data file and write the table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    workers = parser.parse_args().workers
    files = data_files(DATA_DIR)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(file_checksum, root=DATA_DIR), files, chunksize=16))
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)
    logger.info("wrote %d checksums to %s", len(rows), OUT)


if __name__ == "__main__":
    main()
