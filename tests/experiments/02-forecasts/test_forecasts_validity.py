"""The validity table of 02-forecasts (its lib's `validity_table`, Q1) on the synthetic lab."""

import shutil
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.benchmark import ForecastCache, build_cache, cache_settings, on_model_grid, read_cache
from src.benchmark._settings import BASELINES
from src.models import Forecaster
from tests.benchmark.synthetic_lab import lab_config


def test_validity_counts_every_labelled_decision_of_each_model(
    forecasts: ModuleType, run: Path
) -> None:
    table = forecasts.validity_table(read_cache(run), lab_config())
    totals = table[table["window"] == "all"].set_index("model")
    assert set(table["window"]) == {"d1", "d2", "d3", "d4", "all"}
    assert totals.loc["fake", "decisions"] > 0
    for model in BASELINES:
        assert totals.loc[model, "share"] == pytest.approx(1.0)
    assert totals.loc["fake-kronos", "decisions"] < totals.loc["fake", "decisions"] / 50
    assert (totals["share"] == 1.0).all()
    per_window = table[table["window"] != "all"].groupby("model")[["decisions", "valid"]].sum()
    pd.testing.assert_frame_equal(per_window, totals[["decisions", "valid"]].loc[per_window.index])


def _expected_valid(cache: ForecastCache, model: str, labels: pd.DataFrame) -> pd.Series:
    # independent count: a decision (time, asset) is valid when each of its labelled horizons
    # has a row of nine finite quantiles that never decrease
    frame = cache.forecasts(model)
    values = frame[[f"q{level}" for level in range(10, 100, 10)]].to_numpy(dtype=float)
    frame = frame.assign(
        ok=np.isfinite(values).all(axis=1) & (np.diff(values, axis=1) >= 0).all(axis=1)
    )
    rows = labels.merge(frame[["time", "asset", "horizon", "ok"]], how="left").fillna({"ok": False})
    return rows.groupby(["time", "asset"])["ok"].all()


def test_validity_rejects_nan_and_unordered_quantiles_exactly(
    forecasts: ModuleType,
    lab: tuple[Path, Path],
    built: dict[str, list[Forecaster]],
    tmp_path: Path,
) -> None:
    config = lab_config(models=("broken",))
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("broken",))
    cache = read_cache(tmp_path)
    table = forecasts.validity_table(cache, config)
    released = cache.labels[cache.labels["time"] > pd.Timestamp("2025-01-01", tz="UTC")]
    expected = _expected_valid(cache, "broken", released)
    row = table.set_index("window").loc["all"]
    assert row["decisions"] == len(expected)
    assert row["valid"] == int(expected.sum())
    assert 0 < row["share"] < 1


def test_one_invalid_horizon_makes_the_whole_decision_invalid(
    forecasts: ModuleType, run: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    path = copy / "forecasts" / "fake.parquet"
    frame = pd.read_parquet(path)
    later = frame[(frame["time"] > pd.Timestamp("2025-02-01", tz="UTC")) & (frame["horizon"] == 24)]
    broken = later.index[:5]
    frame.loc[broken, "q50"] = np.nan
    frame.to_parquet(path)
    table = forecasts.validity_table(read_cache(copy), lab_config())
    totals = table[table["window"] == "all"].set_index("model")
    assert totals.loc["fake", "decisions"] - totals.loc["fake", "valid"] == 5


def test_only_decisions_after_the_release_count_and_a_late_model_has_no_window(
    forecasts: ModuleType, run: Path, tmp_path: Path
) -> None:
    cache = read_cache(run)
    table = forecasts.validity_table(cache, lab_config())
    fake = table[table["model"] == "fake"].set_index("window")
    labels = on_model_grid(cache.labels, "fake", cache_settings(lab_config()))
    after = labels[labels["time"] > pd.Timestamp("2025-01-01", tz="UTC")]
    assert fake.loc["all", "decisions"] == len(after.drop_duplicates(["time", "asset"]))
    assert (
        fake.loc["d1", "decisions"]
        < table.set_index(["model", "window"]).loc[("ewma-normal", "d1"), "decisions"]
    )
    # Kronos is released 2025-06-30: no decision in d1 and d2
    kronos = table[table["model"] == "fake-kronos"]
    assert set(kronos["window"]) == {"d3", "d4", "all"}
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    models = pd.read_csv(copy / "models.csv")
    models.loc[models["model"] == "fake", "release_date"] = "2026-01-01"
    models.to_csv(copy / "models.csv", index=False)
    late = forecasts.validity_table(read_cache(copy), lab_config())
    row = late[late["model"] == "fake"]
    assert row["window"].tolist() == ["all"]
    assert row["decisions"].tolist() == [0]
    assert row["share"].isna().all()


def test_a_single_seed_kronos_is_valid_and_an_unbuilt_model_raises_clearly(
    forecasts: ModuleType, run: Path, tmp_path: Path
) -> None:
    table = forecasts.validity_table(read_cache(run), replace(lab_config(), seeds=(0,)))
    assert table.set_index(["model", "window"]).loc[("fake-kronos", "all"), "share"] == 1.0
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    (copy / "forecasts" / "linear.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="linear is not built"):
        forecasts.validity_table(read_cache(copy), lab_config())


def test_a_listed_fit_that_did_not_converge_leaves_the_count(
    forecasts: ModuleType, run: Path, tmp_path: Path
) -> None:
    # garch-normal without one asset's d2 rows: invalid unless listed in not_fitted.csv, then
    # left out of the count, counted in not_fitted, and the share stays 1
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    cache = read_cache(copy)
    asset = str(cache.labels["asset"].iloc[0])
    keys = cache.labels[(cache.labels["window"] == "d2") & (cache.labels["asset"] == asset)]
    path = copy / "forecasts" / "garch-normal.parquet"
    frame = pd.read_parquet(path)
    gone = frame.merge(keys[["time", "asset", "horizon"]], how="left", indicator=True)
    frame[(gone["_merge"] == "left_only").to_numpy()].to_parquet(path)
    before = forecasts.validity_table(read_cache(copy), lab_config()).set_index(["model", "window"])
    assert before.loc[("garch-normal", "d2"), "share"] < 1
    listed = pd.DataFrame(
        {"model": ["garch-normal"], "window": ["d2"], "asset": [asset], "convergence_flag": [8]}
    )
    listed.to_csv(copy / "not_fitted.csv", index=False)
    after = forecasts.validity_table(read_cache(copy), lab_config()).set_index(["model", "window"])
    decisions = keys.drop_duplicates(["time", "asset"])
    assert after.loc[("garch-normal", "d2"), "share"] == 1.0
    assert after.loc[("garch-normal", "d2"), "not_fitted"] == len(decisions)
    assert after.loc[("garch-normal", "d1"), "not_fitted"] == 0
    assert after.loc[("ewma-normal", "d2"), "not_fitted"] == 0


def test_a_window_whose_decisions_are_all_listed_keeps_its_count(
    forecasts: ModuleType, run: Path, tmp_path: Path
) -> None:
    # every asset of d2 listed for garch-normal: d2 keeps a row with no decision to count and
    # its not_fitted count, which also enters the model's "all" row
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    cache = read_cache(copy)
    d2 = cache.labels[cache.labels["window"] == "d2"]
    path = copy / "forecasts" / "garch-normal.parquet"
    frame = pd.read_parquet(path)
    gone = frame.merge(d2[["time", "asset", "horizon"]], how="left", indicator=True)
    frame[(gone["_merge"] == "left_only").to_numpy()].to_parquet(path)
    assets = sorted(d2["asset"].astype(str).unique())
    listed = pd.DataFrame(
        {"model": "garch-normal", "window": "d2", "asset": assets, "convergence_flag": 8}
    )
    listed.to_csv(copy / "not_fitted.csv", index=False)
    table = forecasts.validity_table(read_cache(copy), lab_config()).set_index(["model", "window"])
    decisions = len(d2.drop_duplicates(["time", "asset"]))
    assert table.loc[("garch-normal", "d2"), "decisions"] == 0
    assert table.loc[("garch-normal", "d2"), "not_fitted"] == decisions
    assert table.loc[("garch-normal", "all"), "not_fitted"] == decisions
    assert table.loc[("garch-normal", "d1"), "share"] == 1.0
