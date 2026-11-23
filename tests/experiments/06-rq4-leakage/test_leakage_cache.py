"""rq4: a test cache and a leakage run of 02-forecasts, read as one cache."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import Bootstrap, StudySettings, read_cache
from src.benchmark._cache import write_forecasts
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS
from src.experiment import load_study

STUDY = LAB_DIR / "experiments" / "06-rq4-leakage"
MODELS = ["random-walk", "climatology", "skilled", "sampled", "leaky"]
ALREADY_IN_TEST_CACHE = ("random-walk", "climatology")
LEAKAGE = {fold.name for fold in LEAKAGE_FOLDS}
PLAN = {"data_version": "d", "seeds": [0, 1, 2], "kronos_commit": "k", "versions": {"torch": "2"}}
CONFIG = '[folds]\nuse = "test"\n\n[options]\nlookback = 512\nkronos_paths = 20\n'

BOOTSTRAP = Bootstrap(level=0.95, n_resamples=200, seed=0, block_hours=24, min_block=4)


def _keep_windows(folder: Path, model: str, *, is_leakage: bool) -> None:
    # rewrite a model's forecasts with only the leakage windows, or only the other windows
    labels = read_cache(folder).labels[["time", "asset", "horizon", "window"]]
    frame = pd.read_parquet(folder / "forecasts" / f"{model}.parquet")
    frame = frame.merge(labels, on=["time", "asset", "horizon"])
    write_forecasts(folder, model, frame[frame["window"].isin(LEAKAGE) == is_leakage])


def _split(full: Path, root: Path) -> tuple[Path, Path]:
    # the test cache: every window of random-walk and climatology, the others without l1-l12;
    # the leakage run: l1-l12 of the other models only
    test, leakage = root / "test", root / "leakage"
    shutil.copytree(full, test)
    shutil.copytree(full, leakage)
    for run in (test, leakage):
        (run / "meta.json").write_text(
            json.dumps(PLAN | {"git_commit": run.name, "plan_hash": run.name})
        )
        (run / "config.toml").write_text(CONFIG)
    for model in MODELS:
        if model in ALREADY_IN_TEST_CACHE:
            (leakage / "forecasts" / f"{model}.parquet").unlink()
            continue
        _keep_windows(test, model, is_leakage=False)
        _keep_windows(leakage, model, is_leakage=True)
    return test, leakage


@pytest.fixture(name="runs")
def _runs(leakage_cache_folder: Path, tmp_path: Path) -> tuple[Path, Path]:
    return _split(leakage_cache_folder, tmp_path)


def _cells(leakage: ModuleType, cache: object) -> pd.DataFrame:
    settings = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, BOOTSTRAP)
    options = leakage.LeakageOptions(
        grid_hours=4,
        placebo_models=("skilled",),
        placebo_split=pd.Timestamp("2024-01-01", tz="UTC"),
        not_built=(),
    )
    tables = leakage.leakage_tables(
        cache, MODELS, settings, before_folds=LEAKAGE_FOLDS, options=options
    )
    return tables["cells"]


def test_the_joined_runs_score_as_the_full_cache(
    leakage: ModuleType, leakage_cache_folder: Path, runs: tuple[Path, Path]
) -> None:
    joined = leakage.join_caches(read_cache(runs[0]), read_cache(runs[1]))
    for model in MODELS:
        full = read_cache(leakage_cache_folder).forecasts(model)
        both = joined.forecasts(model).sort_values(["time", "asset", "horizon", "seed"])
        expected = full.sort_values(["time", "asset", "horizon", "seed"])
        pd.testing.assert_frame_equal(both.reset_index(drop=True), expected.reset_index(drop=True))
    expected_cells = _cells(leakage, read_cache(leakage_cache_folder))
    pd.testing.assert_frame_equal(_cells(leakage, joined), expected_cells)


def test_runs_with_other_labels_models_or_assets_are_refused(
    leakage: ModuleType, runs: tuple[Path, Path]
) -> None:
    test, other = read_cache(runs[0]), read_cache(runs[1])
    changed = {
        "labels": other.labels.assign(label=other.labels["label"] + 1.0),
        "models": other.models.assign(every_hours=other.models["every_hours"] + 1),
        "assets": other.assets.assign(cost_per_side_bps=other.assets["cost_per_side_bps"] * 2),
    }
    for table, frame in changed.items():
        tables = {"labels": other.labels, "models": other.models, "assets": other.assets}
        tables[table] = frame
        different = type(other)(other.folder, tables["labels"], tables["models"], tables["assets"])
        with pytest.raises(ValueError, match=f"have different {table}"):
            leakage.join_caches(test, different)


def test_runs_of_another_plan_or_seeds_are_refused(
    leakage: ModuleType, runs: tuple[Path, Path]
) -> None:
    changes = (
        ("data_version", "other"),
        ("seeds", [0]),
        ("kronos_commit", "x"),
        ("versions", {"torch": "3"}),
    )
    for field, value in changes:
        (runs[1] / "meta.json").write_text(json.dumps(PLAN | {field: value}))
        with pytest.raises(ValueError, match=f"differ in \\['{field}'\\]"):
            leakage.join_caches(read_cache(runs[0]), read_cache(runs[1]))
    (runs[1] / "meta.json").write_text(json.dumps(PLAN))
    (runs[1] / "config.toml").write_text(CONFIG.replace("kronos_paths = 20", "kronos_paths = 5"))
    with pytest.raises(ValueError, match=r"differ in \['options'\]"):
        leakage.join_caches(read_cache(runs[0]), read_cache(runs[1]))


def test_labels_within_float_rounding_join_and_beyond_it_do_not(
    leakage: ModuleType, runs: tuple[Path, Path]
) -> None:
    test, other = read_cache(runs[0]), read_cache(runs[1])
    for change, joins in ((1 + 1e-12, True), (1 + 1e-6, False)):
        labels = other.labels.assign(sigma=other.labels["sigma"] * change)
        changed = type(other)(other.folder, labels, other.models, other.assets)
        if joins:
            leakage.join_caches(test, changed)
            continue
        with pytest.raises(ValueError, match="have different labels"):
            leakage.join_caches(test, changed)


def test_a_model_without_a_window_is_refused(leakage: ModuleType, runs: tuple[Path, Path]) -> None:
    test, other = runs
    _keep_windows(other, "skilled", is_leakage=False)
    joined = leakage.join_caches(read_cache(test), read_cache(other))
    with pytest.raises(ValueError, match=r"for \(model, window\) \[\('skilled', 'l1'\)"):
        leakage.require_windows(joined, MODELS, sorted(LEAKAGE))
    leakage.require_windows(joined, ["random-walk", "leaky"], sorted(LEAKAGE))


def test_leakage_cache_skips_the_leakage_quarters_of_a_model_not_built(
    leakage: ModuleType, runs: tuple[Path, Path], tmp_path: Path, plan_text: Callable[..., str]
) -> None:
    (runs[1] / "forecasts" / "leaky.parquet").unlink()
    study = _study(tmp_path / "joined", runs, "leakage", plan_text)
    config_path = study / "study.toml"
    text = plan_text(config_path.read_text(), locked=True, use="test", not_built=[])
    config_path.write_text(text)
    with pytest.raises(ValueError, match=r"\('leaky', 'l1'\)"):
        leakage.leakage_cache(study, load_study(config_path))
    config_path.write_text(plan_text(text, not_built=["leaky"]))
    cache = leakage.leakage_cache(study, load_study(config_path))
    assert set(cache.forecasts("leaky")["window"]).isdisjoint(LEAKAGE)


def test_a_window_in_both_runs_is_refused(leakage: ModuleType, runs: tuple[Path, Path]) -> None:
    test, other = runs
    shutil.copy(test / "forecasts" / "random-walk.parquet", other / "forecasts")
    joined = leakage.join_caches(read_cache(test), read_cache(other))
    with pytest.raises(ValueError, match=r"random-walk: windows \['d1', .*\] are in both"):
        joined.forecasts("random-walk")


def test_a_model_without_forecasts_in_either_run_is_refused(
    leakage: ModuleType, runs: tuple[Path, Path]
) -> None:
    test, other = runs
    (test / "forecasts" / "random-walk.parquet").unlink()
    joined = leakage.join_caches(read_cache(test), read_cache(other))
    with pytest.raises(ValueError, match=r"random-walk has no forecasts in .* nor in"):
        joined.forecasts("random-walk")
    with pytest.raises(ValueError, match="model 'absent' is not in"):
        joined.forecasts("absent")


def _study(
    tmp_path: Path, runs: tuple[Path, Path], leakage_run: str, plan_text: Callable[..., str]
) -> Path:
    study = tmp_path / "experiments" / "06-rq4-leakage"
    study.mkdir(parents=True)
    config = (STUDY / "study.toml").read_text()
    config = plan_text(
        config, locked=False, use="development", forecast_cache="test", leakage_cache=leakage_run
    )
    start, end = config.index("[models]"), config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in MODELS) + "]\n\n"
    config = config[:start] + names + config[end:]
    (study / "study.toml").write_text(config)
    for run in runs:
        shutil.copytree(run, tmp_path / "experiments" / "02-forecasts" / "runs" / run.name)
    return study


def test_leakage_cache_reads_one_run_or_joins_two(
    leakage: ModuleType,
    runs: tuple[Path, Path],
    tmp_path: Path,
    plan_text: Callable[..., str],
) -> None:
    study = _study(tmp_path / "joined", runs, "leakage", plan_text)
    joined = leakage.leakage_cache(study, load_study(study / "study.toml"))
    assert isinstance(joined, leakage.JoinedCache)
    assert joined.other.folder.name == "leakage"
    alone = _study(tmp_path / "alone", runs, "", plan_text)
    cache = leakage.leakage_cache(alone, load_study(alone / "study.toml"))
    assert not isinstance(cache, leakage.JoinedCache)
    assert cache.folder.name == "test"
    missing = _study(tmp_path / "missing", runs, "absent", plan_text)
    with pytest.raises(FileNotFoundError, match=r"check options\.leakage_cache"):
        leakage.leakage_cache(missing, load_study(missing / "study.toml"))


def test_not_fitted_reads_the_listed_fits_of_both_runs(
    leakage: ModuleType,
    runs: tuple[Path, Path],
    tmp_path: Path,
    plan_text: Callable[..., str],
) -> None:
    study = _study(tmp_path / "joined", runs, "leakage", plan_text)
    config = load_study(study / "study.toml")
    assert leakage.not_fitted(study, config).empty
    folder = tmp_path / "joined" / "experiments" / "02-forecasts" / "runs"
    fits = pd.DataFrame(
        {"model": ["garch-normal"], "window": ["l3"], "asset": ["NATGAS"], "convergence_flag": [8]}
    )
    fits.assign(extra=1).to_csv(folder / "leakage" / "not_fitted.csv", index=False)
    listed = leakage.not_fitted(study, config)
    pd.testing.assert_frame_equal(listed, fits)
    alone = _study(tmp_path / "alone", runs, "", plan_text)
    assert leakage.not_fitted(alone, load_study(alone / "study.toml")).empty
