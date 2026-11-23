from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.data._config import DataConfig, Instrument, QualityLimits
from src.data._download import (
    _collect_builds,
    _describe_on_disk,
    _parts_dir,
    _tasks,
    _wait_for_downloads,
    download_all,
)
from src.data._read import bars_file
from src.enums import Market, Source, Timeframe


def _config(tmp_path, instrument):
    limits = QualityLimits(0.95, 10, 0.001, 10, 0.4, 30, 300, 0.25)
    return DataConfig("2024-01-01", "2025-01-01", {"dukascopy": 1}, limits, tmp_path, (instrument,))


def test_a_1h_instrument_downloads_1h_parts_into_their_own_folder(tmp_path):
    hourly = Instrument("SAP.DE", Source.DUKASCOPY, Market.FX, "sapdeeur", Timeframe.H1)
    config = _config(tmp_path, hourly)
    tasks = _tasks(config, hourly)
    assert {task[2]["timeframe"] for task in tasks} == {Timeframe.H1}
    assert _parts_dir(config, hourly).name == "1h"
    plain = Instrument("EURUSD", Source.DUKASCOPY, Market.FX, "eurusd")
    assert _parts_dir(_config(tmp_path, plain), plain).name == "dukascopy"


def test_download_only_builds_nothing_and_keeps_the_manifest(tmp_path):
    config = DataConfig(
        "2024-01-01",
        "2025-01-01",
        {"binance": 1, "dukascopy": 1, "alpaca": 1, "yahoo": 1, "build": 1},
        QualityLimits(0.95, 10, 0.001, 10, 0.4, 30, 300, 0.25),
        tmp_path,
        (),
    )
    assert download_all(config, build=False).empty
    assert not (tmp_path / "manifest.json").exists()


def test_a_stock_without_reference_gets_no_yahoo_task(tmp_path):
    with_reference = Instrument("SAP.DE", Source.DUKASCOPY, Market.STOCK, "sapdeeur")
    without = Instrument("ROG.SW", Source.DUKASCOPY, Market.STOCK, "rogchchf", has_reference=False)
    with_tasks = _tasks(_config(tmp_path, with_reference), with_reference)
    without_tasks = _tasks(_config(tmp_path, without), without)
    assert "yahoo" in {task[0] for task in with_tasks}
    assert "yahoo" not in {task[0] for task in without_tasks}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def test_a_failed_download_is_reported_and_starts_no_build(tmp_path: Path):
    instrument = Instrument("EURUSD", Source.DUKASCOPY, Market.FX, "eurusd")
    config = _config(tmp_path, instrument)
    with ThreadPoolExecutor(max_workers=1) as pool:
        downloads: dict[Future[object], str] = {
            pool.submit(str, "ok"): "EURUSD",
            pool.submit(_fail, "no file"): "EURUSD",
            pool.submit(str, "ok"): "GBPUSD",
        }
        tasks_per_instrument = {"EURUSD": 2, "GBPUSD": 1}
        builds, failed = _wait_for_downloads(config, downloads, tasks_per_instrument, None)
    assert builds == {}
    assert failed == {"EURUSD": "download failed: no file"}
    assert len(downloads) == 3
    assert tasks_per_instrument == {"EURUSD": 2, "GBPUSD": 1}


def test_collect_builds_keeps_rows_and_reports_failed_builds():
    applied = pd.DataFrame(columns=["symbol", "date", "ratio", "source"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        builds: dict[Future[object], str] = {
            pool.submit(tuple, [{"symbol": "EURUSD"}, applied]): "EURUSD",
            pool.submit(_fail, "bad bars"): "GBPUSD",
        }
        rows, splits, failed = _collect_builds(builds)
    assert rows == [{"symbol": "EURUSD"}]
    assert len(splits) == 1
    assert splits[0] is applied
    assert failed == {"GBPUSD": "build failed: bad bars"}


def test_a_stock_missing_its_yahoo_reference_fails_on_disk(tmp_path: Path):
    stock = Instrument("SAP.DE", Source.DUKASCOPY, Market.STOCK, "sapdeeur")
    config = _config(tmp_path, stock)
    index = pd.date_range("2024-01-02 08:00", periods=40, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": 100.0}, index=index)
    path = bars_file(tmp_path, Market.STOCK, "SAP.DE", Timeframe.M15)
    path.parent.mkdir(parents=True)
    bars.to_parquet(path)
    row, applied = _describe_on_disk(config, stock)
    assert row["passed"] is False
    assert row["reason"] == "no Yahoo reference yet"
    assert row["symbol"] == "SAP.DE"
    assert row["n_bars"] == 40
    assert applied.empty
