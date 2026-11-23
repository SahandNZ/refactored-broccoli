from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import read_cache, study_cache
from src.benchmark._cache import FORECAST_COLUMNS, LABEL_COLUMNS, QUANTILE_COLUMNS
from src.experiment import load_study


def test_read_cache_parses_dates_and_keeps_every_table(cache_folder: Path) -> None:
    cache = read_cache(cache_folder)
    assert cache.models["model"].tolist() == ["random-walk", "climatology", "skilled", "sampled"]
    assert cache.models["release_date"].isna().tolist() == [True, True, False, False]
    assert str(cache.models["release_date"].dt.tz) == "UTC"
    assert set(cache.labels["window"]) == {"l1", "d1", "t1"}


def test_forecasts_join_labels_on_time_asset_and_horizon(cache_folder: Path) -> None:
    cache = read_cache(cache_folder)
    frame = cache.forecasts("skilled")
    assert set(frame.columns) == set(LABEL_COLUMNS) | set(FORECAST_COLUMNS)
    assert len(frame) == len(cache.labels)
    np.testing.assert_allclose(frame["q50"], 0.5 * frame["label"], atol=1e-6)
    assert (frame[list(QUANTILE_COLUMNS)].dtypes == "float32").all()


def test_forecasts_keep_one_row_per_seed_and_only_forecast_decisions(cache_folder: Path) -> None:
    frame = read_cache(cache_folder).forecasts("sampled")
    assert sorted(frame["seed"].unique()) == [0, 1]
    assert (frame["time"].dt.hour % 4 == 0).all()
    assert not frame.duplicated(["time", "asset", "horizon", "seed"]).any()


def test_unknown_model_and_missing_column_raise(cache_folder: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'nope' is not in"):
        read_cache(cache_folder).forecasts("nope")
    for name in ("labels.parquet", "models.csv", "assets.csv"):
        (tmp_path / name).write_bytes((cache_folder / name).read_bytes())
    pd.read_parquet(tmp_path / "labels.parquet").drop(columns="sigma").to_parquet(
        tmp_path / "labels.parquet"
    )
    with pytest.raises(ValueError, match="missing columns \\['sigma'\\]"):
        read_cache(tmp_path)


def test_a_label_that_is_not_finite_raises(cache_folder: Path, tmp_path: Path) -> None:
    for name in ("labels.parquet", "models.csv", "assets.csv"):
        (tmp_path / name).write_bytes((cache_folder / name).read_bytes())
    labels = pd.read_parquet(tmp_path / "labels.parquet")
    labels.loc[:9, "label"] = np.nan
    labels.to_parquet(tmp_path / "labels.parquet")
    with pytest.raises(ValueError, match="labels must be finite, got 10 other values"):
        read_cache(tmp_path)


def test_a_study_reads_the_cache_its_options_name(cache_folder: Path, tmp_path: Path) -> None:
    runs = tmp_path / "experiments" / "02-forecasts" / "runs"
    runs.mkdir(parents=True)
    (runs / "synthetic").symlink_to(cache_folder)
    study = tmp_path / "experiments" / "05-rq3-net"
    config = load_study(LAB_DIR / "experiments" / "05-rq3-net" / "study.toml")
    named = replace(config, options={**config.options, "forecast_cache": "synthetic"})
    assert study_cache(study, named).folder == runs / "synthetic"
    absent = replace(config, options={**config.options, "forecast_cache": "absent"})
    with pytest.raises(FileNotFoundError, match=r"no forecast cache in .*absent"):
        study_cache(study, absent)
