import logging
import multiprocessing
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import pandas as pd

from src.data import _alpaca, _binance, _dukascopy, _yahoo
from src.data._config import DataConfig, Instrument
from src.data._manifest import describe_bars, write_manifest
from src.data._quality import adjust_for_splits, check_quality, stock_calendar
from src.data._read import bars_file
from src.data._store import write_all_timeframes
from src.enums import Market, Source, Timeframe
from src.utils import months_between, years_between

logger = logging.getLogger(__name__)

RETRIES = 4
"""Attempts after the first one for every file, with exponential pauses."""

type Task = tuple[str, Callable[..., Path], dict[str, object]]


def _instrument_dir(config: DataConfig, instrument: Instrument) -> Path:
    return config.data_dir / instrument.market.value / instrument.name


def _parts_dir(config: DataConfig, instrument: Instrument) -> Path:
    folder = _instrument_dir(config, instrument) / "raw" / instrument.source.value
    if instrument.timeframe is not Timeframe.M15:
        folder = folder / instrument.timeframe.value  # keep 15m and coarser parts apart
    return folder


def _reference_dir(config: DataConfig, instrument: Instrument) -> Path:
    return _instrument_dir(config, instrument) / "reference" / "yahoo"


def _tasks(config: DataConfig, instrument: Instrument) -> list[Task]:
    parts_dir = _parts_dir(config, instrument)
    tasks: list[Task] = []
    if instrument.source is Source.DUKASCOPY:
        for year, date_from, date_to in years_between(config.start, config.end):
            for side in _dukascopy.SIDES:
                arguments = {
                    "code": instrument.code, "side": side, "year": year, "date_from": date_from,
                    "date_to": date_to, "parts_dir": parts_dir, "retries": RETRIES,
                    "timeframe": instrument.timeframe,
                }  # fmt: skip
                tasks.append(("dukascopy", _dukascopy.download_part, arguments))
    elif instrument.source is Source.ALPACA:
        for year, date_from, date_to in years_between(config.start, config.end):
            arguments = {
                "symbol": instrument.code, "year": year, "date_from": date_from,
                "date_to": date_to, "parts_dir": parts_dir, "retries": RETRIES,
            }  # fmt: skip
            tasks.append(("alpaca", _alpaca.download_year, arguments))
    else:
        for month in months_between(config.start, config.end):
            arguments = {
                "symbol": instrument.code,
                "month": month,
                "parts_dir": parts_dir,
                "retries": RETRIES,
            }
            tasks.append(("binance", _binance.download_month, arguments))
    if instrument.market is Market.STOCK and instrument.has_reference:
        arguments = {
            "ticker": instrument.name, "start": config.start, "end": config.end,
            "folder": _reference_dir(config, instrument), "retries": RETRIES,
        }  # fmt: skip
        tasks.append(("yahoo", _yahoo.download_reference, arguments))
    return tasks


def build_instrument(
    config: DataConfig, instrument: Instrument
) -> tuple[dict[str, object], pd.DataFrame]:
    """Build one instrument from its downloaded parts.

    Adjust stock splits, check the quality and write every timeframe.

    Parameters
    ----------
    config : DataConfig
        The data configuration.
    instrument : Instrument
        The instrument to build.

    Returns
    -------
    tuple of (dict, pd.DataFrame)
        Its ``data/manifest.json`` entry and the splits applied (symbol, date, ratio, source).
    """
    parts_dir = _parts_dir(config, instrument)
    if instrument.source is Source.DUKASCOPY:
        bars = _dukascopy.build_bars(parts_dir)
    elif instrument.source is Source.ALPACA:
        bars = _alpaca.build_bars(parts_dir, start=config.start, end=config.end)
    else:
        bars = _binance.build_bars(parts_dir, start=config.start, end=config.end)
    reference, applied = None, pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    if instrument.market is Market.STOCK and instrument.has_reference:
        folder = _reference_dir(config, instrument)
        reference = pd.read_parquet(folder / "1d.pq")["close"]
        splits = pd.read_parquet(folder / "splits.pq")["ratio"]
        if instrument.source is Source.DUKASCOPY and len(splits):
            bars = adjust_for_splits(bars, splits, timezone=str(stock_calendar(instrument).tz))
            applied = pd.DataFrame({
                "symbol": instrument.name, "date": splits.index, "ratio": splits.to_numpy(),
                "source": "yahoo",
            })  # fmt: skip
    write_all_timeframes(
        bars, data_dir=config.data_dir, market=instrument.market, symbol=instrument.name,
        base=instrument.timeframe,
    )  # fmt: skip
    row = describe_bars(bars, instrument)
    quality = check_quality(
        bars, instrument, limits=config.quality, reference=reference, end=config.end
    )
    row.update(quality)
    return row, applied


def _submit_downloads(
    config: DataConfig, pools: dict[str, ThreadPoolExecutor]
) -> tuple[dict[Future[object], str], dict[str, int]]:
    pending: dict[Future[object], str] = {}
    remaining: dict[str, int] = {}
    for instrument in config.instruments:
        tasks = _tasks(config, instrument)
        remaining[instrument.name] = len(tasks)
        for source, function, arguments in tasks:
            pending[pools[source].submit(function, **arguments)] = instrument.name
    return pending, remaining


def _wait_for_downloads(
    config: DataConfig,
    downloads: dict[Future[object], str],
    tasks_per_instrument: dict[str, int],
    builder: ProcessPoolExecutor | None,
) -> tuple[dict[Future[object], str], dict[str, str]]:
    """Wait for every download and submit an instrument's build once all its files are in.

    Nothing is built when `builder` is None. Returns the builds started (future: instrument
    name) and the failed instruments (name: reason).
    """
    pending = dict(downloads)
    remaining = dict(tasks_per_instrument)
    by_name = {instrument.name: instrument for instrument in config.instruments}
    builds: dict[Future[object], str] = {}
    failed: dict[str, str] = {}
    done_downloads = 0
    while pending:
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            name = pending.pop(future)
            done_downloads += 1
            if future.exception() is not None:
                failed[name] = f"download failed: {future.exception()}"
                logger.error("%s: %s", name, failed[name])
            remaining[name] -= 1
            if builder is not None and remaining[name] == 0 and name not in failed:
                builds[builder.submit(build_instrument, config, by_name[name])] = name
            if done_downloads % 200 == 0:
                logger.info(
                    "%d downloads left, %d builds started, %d failed",
                    len(pending),
                    len(builds),
                    len(failed),
                )
    return builds, failed


def _collect_builds(
    builds: dict[Future[object], str],
) -> tuple[list[dict[str, object]], list[pd.DataFrame], dict[str, str]]:
    rows: list[dict[str, object]] = []
    splits: list[pd.DataFrame] = []
    failed: dict[str, str] = {}
    for future, name in builds.items():
        if future.exception() is not None:
            failed[name] = f"build failed: {future.exception()}"
            logger.error("%s: %s", name, failed[name])
            continue
        row, applied = future.result()
        rows.append(row)
        splits.append(applied)
    return rows, splits, failed


def download_all(config: DataConfig, *, build: bool = True) -> pd.DataFrame:
    """Download every instrument, build it as soon as its files are in, and write the manifest.

    Downloads are I/O bound and run in one thread pool per source (sizes in
    ``config.workers``); every task skips files already on disk and writes atomically, so an
    interrupted run is resumed by running it again. When all files of an instrument are in, a
    process pool adjusts stock splits, checks the quality and writes every timeframe.
    ``data/manifest.json`` (one entry per instrument: source, quality, ``passed`` and
    ``reason``, and the splits applied) is written at the end.

    Parameters
    ----------
    config : DataConfig
        What to download, the period, the workers and the quality limits.
    build : bool, optional
        False downloads the files only: nothing is built and the manifest is not written (for
        a slow download kept beside already built bars, e.g. 15m parts of 1h instruments).

    Returns
    -------
    pd.DataFrame
        The manifest (empty when `build` is False).
    """
    sources = ("binance", "dukascopy", "alpaca", "yahoo")
    pools = {name: ThreadPoolExecutor(max_workers=config.workers[name]) for name in sources}
    # spawn, not fork: forking a process that runs download threads can deadlock
    builder = ProcessPoolExecutor(
        max_workers=config.workers["build"], mp_context=multiprocessing.get_context("spawn")
    )
    downloads, tasks_per_instrument = _submit_downloads(config, pools)
    logger.info(
        "%d instruments, %d download tasks, workers %s",
        len(tasks_per_instrument),
        len(downloads),
        config.workers,
    )
    selected_builder = builder if build else None
    builds, failed = _wait_for_downloads(config, downloads, tasks_per_instrument, selected_builder)
    for pool in pools.values():
        pool.shutdown()
    if not build:
        builder.shutdown()
        logger.info("downloaded only (no build): %d failed %s", len(failed), sorted(failed))
        return pd.DataFrame()
    rows, splits, build_failed = _collect_builds(builds)
    builder.shutdown()
    failed.update(build_failed)
    by_name = {instrument.name: instrument for instrument in config.instruments}
    for name, reason in failed.items():
        instrument = by_name[name]
        rows.append({
            "source": instrument.source.value, "market": instrument.market.value,
            "symbol": name, "code": instrument.code, "passed": False, "reason": reason,
        })  # fmt: skip
    manifest = pd.DataFrame(rows).sort_values(["market", "symbol"])
    applied = pd.concat(splits)
    write_manifest(config.data_dir / "manifest.json", manifest, applied)
    passed = int(manifest["passed"].sum())
    logger.info("built %d instruments, %d passed, %d failed", len(rows), passed, len(failed))
    return manifest


def _describe_on_disk(
    config: DataConfig, instrument: Instrument
) -> tuple[dict[str, object], pd.DataFrame]:
    path = bars_file(config.data_dir, instrument.market, instrument.name, instrument.timeframe)
    empty = pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    if not path.exists():
        row = {
            "source": instrument.source.value, "market": instrument.market.value,
            "symbol": instrument.name, "code": instrument.code, "passed": False,
            "reason": "not downloaded yet",
        }  # fmt: skip
        return row, empty
    bars = pd.read_parquet(path)
    reference, applied = None, empty
    if instrument.market is Market.STOCK and instrument.has_reference:
        folder = _reference_dir(config, instrument)
        if not (folder / "1d.pq").exists():
            row = describe_bars(bars, instrument)
            row.update({"passed": False, "reason": "no Yahoo reference yet"})
            return row, empty
        reference = pd.read_parquet(folder / "1d.pq")["close"]
        splits = pd.read_parquet(folder / "splits.pq")["ratio"]
        if instrument.source is Source.DUKASCOPY and len(splits):
            applied = pd.DataFrame({
                "symbol": instrument.name, "date": splits.index, "ratio": splits.to_numpy(),
                "source": "yahoo",
            })  # fmt: skip
    row = describe_bars(bars, instrument)
    quality = check_quality(
        bars, instrument, limits=config.quality, reference=reference, end=config.end
    )
    row.update(quality)
    return row, applied


def refresh_manifest(config: DataConfig) -> pd.DataFrame:
    """Write ``data/manifest.json`` from the bars already on disk, while downloads still run.

    Instruments that are fully built get their description and quality checks (their base bars
    are already split-adjusted); the others get ``passed = False`` and the reason "not
    downloaded yet" (or "no Yahoo reference yet" for a stock whose Yahoo reference is still
    missing), so a study run on a partial download excludes them explicitly.

    Parameters
    ----------
    config : DataConfig
        The data configuration (instruments, quality limits, workers).

    Returns
    -------
    pd.DataFrame
        The manifest.
    """
    rows, splits = [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config.workers["build"], mp_context=context) as pool:
        for row, applied in pool.map(
            _describe_on_disk, [config] * len(config.instruments), config.instruments
        ):
            rows.append(row)
            splits.append(applied)
    manifest = pd.DataFrame(rows).sort_values(["market", "symbol"])
    write_manifest(config.data_dir / "manifest.json", manifest, pd.concat(splits))
    logger.info("manifest: %d instruments, %d passed", len(manifest), int(manifest["passed"].sum()))
    return manifest
