"""07-rq5-off-hours end to end: run.py and analyze.py on synthetic bars with fake models."""

import re
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

import src
from src import LAB_DIR
from src.experiment import read_answers

STUDY = LAB_DIR / "experiments" / "07-rq5-off-hours"
"""The real study folder: its plan and scripts are run on a copy."""


@pytest.fixture
def fake_lib(
    study_lib_as_lib: ModuleType,
    off_hours_fake: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
):
    # the study's lib as `lib` (the scripts' `from lib import ...` finds it) with fake
    # models: nothing is loaded and no revision is looked up

    def build(kind: str, **kwargs: Any) -> Any:
        return off_hours_fake()

    def pin(name: str, revision: str) -> None:
        return None

    monkeypatch.setattr(study_lib_as_lib._reopening, "build_forecaster", build)
    monkeypatch.setattr(study_lib_as_lib._reopening, "require_revision", pin)
    return study_lib_as_lib


@pytest.fixture
def study(
    fake_lib: ModuleType,
    tmp_path: Path,
    off_hours_market: Callable[..., Any],
    off_hours_lab: Callable[..., Any],
    off_hours_universe: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
):
    # a lab with an asset that follows weekend crypto (in the universe), one that does not
    # (passed the data checks, not picked) and the crypto basket
    planted, crypto = off_hours_market(beta=0.8, seed=0, start="2023-06-01", end="2025-11-01")
    unrelated, _ = off_hours_market(beta=0.0, seed=1, start="2023-06-01", end="2025-11-01")
    (tmp_path / "data").mkdir()
    off_hours_lab(
        tmp_path / "data",
        {"PLANTED": ("forex", planted), "UNRELATED": ("index", unrelated)},
        crypto,
    )
    off_hours_universe(
        tmp_path,
        {
            "PLANTED": ("forex", True, True),
            "UNRELATED": ("index", False, True),
            "BROKEN": ("commodity", False, False),
        },
    )
    monkeypatch.setattr(src, "LAB_DIR", tmp_path)
    folder = tmp_path / "experiments" / "07-rq5-off-hours"
    folder.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", folder / "README.md")
    # the synthetic bars end in 2025-10: the copy scores the development folds
    config = (STUDY / "study.toml").read_text()
    config = re.sub(r'(?m)^forecast_cache = ".*"$', 'forecast_cache = "synthetic"', config)
    config = re.sub(r'(?m)^use = ".*"$', 'use = "development"', config)
    (folder / "study.toml").write_text(config)
    # the 02-forecasts run whose config.toml gives the context length, the model builds and
    # the 01-universe run
    run = folder.parent / "02-forecasts" / "runs" / "synthetic"
    run.mkdir(parents=True)
    shutil.copy(STUDY.parent / "02-forecasts" / "study.toml", run / "config.toml")
    return folder


def _main(script: str, study: Path, monkeypatch: pytest.MonkeyPatch, *extra: str):
    arguments = [script, "--study-dir", str(study), "--run-id", "synthetic-run", *extra]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(STUDY / script), run_name="__main__")


def test_rq5_forecasts_the_universe_regresses_the_broad_set_and_writes_every_table(
    study: Path, monkeypatch: pytest.MonkeyPatch
):
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    tables = study / "runs" / "synthetic-run" / "tables"
    gaps = pd.read_csv(tables / "gap_assets.csv").set_index("asset")
    assert list(gaps.index) == ["PLANTED", "UNRELATED"]
    assert gaps.loc["PLANTED", "is_yes"]
    assert not gaps.loc["UNRELATED", "is_yes"]
    assert gaps.loc["PLANTED", "beta"] == pytest.approx(0.8, abs=0.05)
    cells = pd.read_csv(tables / "reopening_cells.csv")
    assert set(cells["market"]) == {"forex"}
    assert len(cells) == 3
    assert (cells["status"] == "scored").all()
    tests = pd.read_csv(tables / "reopening_tests.csv")
    assert len(tests) == 3 * 6
    assert {"skill", "skill_low", "skill_high", "adjusted_p"} <= set(tests.columns)
    use = pd.read_csv(tables / "covariate_use.csv")
    assert use["use_share"].notna().all()
    forecasts = pd.read_parquet(study / "runs" / "synthetic-run" / "forecasts.parquet")
    assert set(forecasts["arm"]) == {"event", "calendar", "covariate", "correction", "model-free"}
    assert set(forecasts["asset"]) == {"PLANTED"}
    sets = pd.read_csv(tables / "asset_sets.csv").set_index("asset")
    assert sets["is_forecast"].to_dict() == {"PLANTED": True, "UNRELATED": False}
    for name in ("gap_half_years", "gap_directions", "weekends"):
        assert (tables / f"{name}.csv").exists()
    readme = study / "results" / "README.md"
    answers = read_answers(readme)
    # the planted gap is linear in crypto, so the correction and the model-free arm are as
    # good as the fake's covariate arm: it beats calendar and event time, not them, and Q1
    # says no (the cells where the covariate arm wins are in test_comparisons.py)
    assert answers["Q1"] == "no"
    tests = tests.set_index(["model", "arm", "reference"])
    assert tests.loc[("chronos-2", "covariate", "calendar"), "is_better"]
    assert not tests.loc[("chronos-2", "covariate", "model-free"), "is_better"]
    assert answers["Q2"] == "yes"
    text = readme.read_text()
    assert "1 of 2 tested assets: PLANTED beta " in text
    assert "tested market and model cells" in text
    assert (study / "results" / "synthetic-run" / "tables" / "reopening_tests.csv").exists()


def test_rq5_dry_run_reads_no_data_and_writes_no_run(study: Path, monkeypatch: pytest.MonkeyPatch):
    shutil.rmtree(study.parent.parent / "data")
    _main("run.py", study, monkeypatch, "--dry-run")
    assert not (study / "runs").exists()


def test_rq5_records_a_failed_run(study: Path, monkeypatch: pytest.MonkeyPatch):
    shutil.rmtree(study.parent.parent / "data" / "forex")
    with pytest.raises(FileNotFoundError):
        _main("run.py", study, monkeypatch)
    meta = (study / "runs" / "synthetic-run" / "meta.json").read_text()
    assert '"status": "failed"' in meta


def test_rq5_refuses_a_subset_of_its_models(study: Path, monkeypatch: pytest.MonkeyPatch):
    # every test of every model is in one Holm family, so a run takes every model
    with pytest.raises(SystemExit, match="drop --models"):
        _main("run.py", study, monkeypatch, "--models", "chronos-2")
    assert not (study / "runs").exists()


def test_rq5_is_inconclusive_when_nothing_can_be_tested(
    study: Path, monkeypatch: pytest.MonkeyPatch
):
    config = (study / "study.toml").read_text()
    (study / "study.toml").write_text(config.replace("min_weekends = 20", "min_weekends = 1000"))
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    answers = read_answers(study / "results" / "README.md")
    assert answers == {"Q1": "inconclusive", "Q2": "inconclusive"}
