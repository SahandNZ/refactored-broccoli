"""Download every candidate instrument and build all timeframes.

Dukascopy 15m bid/ask and Binance USD-M 15m klines go to
``data/<source>/<instrument>/15m.parquet``; 30m .. 1w are built from them; the quality of each
instrument goes to ``data/manifest.csv``; the log to ``data/download.log``. Resumable: run it
again after an interruption.

Usage: .venv/bin/python scripts/download_data.py [--config configs/data.toml] [--workers N]
       [--groups NAME ...] [--select START:STOP] [--timeframe 15m] [--no-build]

``--groups`` keeps only those groups of the config; ``--select`` then keeps a slice of the
remaining instruments (for example ``0:55``), so two machines can split one download.
``--timeframe`` downloads another base timeframe than the config's (its raw parts go to their
own folder); ``--no-build`` only downloads, so already built bars and the manifest stay as they
are (e.g. the slow 15m parts of the non-US stocks, built from 1h).
"""

import argparse
import dataclasses
import logging
from pathlib import Path

from src import LAB_DIR
from src.data import download_all, load_data_config
from src.enums import Timeframe


def main() -> None:
    """Parse the arguments, configure logging and run the download."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=LAB_DIR / "configs" / "data.toml")
    parser.add_argument("--workers", type=int, help="override the config's number of workers")
    parser.add_argument("--groups", nargs="+", help="only these groups of the config")
    parser.add_argument("--select", help="slice START:STOP of the chosen instruments")
    parser.add_argument("--timeframe", help="download this base timeframe instead (e.g. 15m)")
    parser.add_argument("--no-build", action="store_true", help="download only, build nothing")
    arguments = parser.parse_args()
    data_dir = LAB_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(data_dir / "download.log"), logging.StreamHandler()],
    )
    config = load_data_config(arguments.config, data_dir=data_dir, groups=arguments.groups)
    if arguments.select is not None:
        start, stop = (int(part) for part in arguments.select.split(":"))
        config = dataclasses.replace(config, instruments=config.instruments[start:stop])
    if arguments.timeframe is not None:
        timeframe = Timeframe(arguments.timeframe)
        instruments = tuple(dataclasses.replace(i, timeframe=timeframe) for i in config.instruments)
        config = dataclasses.replace(config, instruments=instruments)
    if arguments.workers is not None:
        config = dataclasses.replace(config, workers=arguments.workers)
    download_all(config, build=not arguments.no_build)


if __name__ == "__main__":
    main()
