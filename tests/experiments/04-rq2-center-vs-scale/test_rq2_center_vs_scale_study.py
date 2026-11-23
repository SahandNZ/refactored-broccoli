"""04-rq2-center-vs-scale end to end: run.py and analyze.py on the synthetic cache."""

import re
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import LAB_DIR
from src.experiment import load_study, read_answers
from src.figures import MODEL_FAMILY

STUDY = LAB_DIR / "experiments" / "04-rq2-center-vs-scale"
BASELINES = ["climatology", "ewma-normal"]


# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


@pytest.fixture(autouse=True)
def _synthetic_model_styles(monkeypatch: pytest.MonkeyPatch) -> None:
    # the figure draws each model in its family's style; the synthetic models borrow two
    monkeypatch.setitem(MODEL_FAMILY, "centre-only", ("Chronos-2", ""))
    monkeypatch.setitem(MODEL_FAMILY, "scale-only", ("TimesFM 2.5", ""))


def _study(tmp_path: Path, cache_folder: Path, models: list[str]) -> Path:
    paper = tmp_path / "experiments"
    study = paper / "04-rq2-center-vs-scale"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    config = (STUDY / "study.toml").read_text()
    replacements = {
        "coverage_tolerance = 0.05": "coverage_tolerance = 0.1",
        "bootstrap_resamples = 2000": "bootstrap_resamples = 200",
    }
    # the study's own cache id (a real run) is replaced by the synthetic cache
    config = re.sub(r'forecast_cache = ".*"', 'forecast_cache = "synthetic"', config)
    config = re.sub(r"locked = \w+", "locked = true", config)
    config = re.sub(r'use = "\w+"', 'use = "test"', config)
    for old, new in replacements.items():
        assert old in config
        config = config.replace(old, new)
    start = config.index("[models]")
    end = config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in models) + "]\n\n"
    (study / "study.toml").write_text(config[:start] + names + config[end:])
    shutil.copytree(cache_folder, paper / "02-forecasts" / "runs" / "synthetic")
    return study


def _main(script: str, study: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    arguments = [script, "--study-dir", str(study), "--run-id", "synthetic-run", *extra]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(STUDY / script), run_name="__main__")


def _run(
    tmp_path: Path, cache_folder: Path, models: list[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    study = _study(tmp_path, cache_folder, models)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    return study


def test_the_study_options_are_the_ones_the_code_reads() -> None:
    options = load_study(STUDY / "study.toml").options
    assert set(options) == {
        "forecast_cache", "development_run", "scale_reference", "centre_reference", "alpha",
        "min_quarters", "main_family_min_quarters", "coverage_tolerance", "bootstrap_block_hours",
        "bootstrap_min_block", "bootstrap_resamples", "bootstrap_seed", "figure_horizon",
    }  # fmt: skip
    assert options["scale_reference"] == "ewma-normal"
    assert options["centre_reference"] == "climatology"


def test_rq2_tells_centre_skill_from_scale_skill(
    tmp_path: Path, centre_scale_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [*BASELINES, "centre-only", "scale-only"]
    study = _run(tmp_path, centre_scale_cache_folder, models, monkeypatch)
    tables = study / "runs" / "synthetic-run" / "tables"
    verdict = pd.read_csv(tables / "verdict.csv")
    assert set(verdict["model"]) == {"centre-only", "scale-only"}
    by_model = verdict.groupby("model")[["has_centre_skill", "is_calibrated"]].all()
    assert by_model.loc["centre-only", "has_centre_skill"]
    assert not by_model.loc["centre-only", "is_calibrated"]
    assert by_model.loc["scale-only", "is_calibrated"]
    assert not verdict.loc[verdict["model"] == "scale-only", "has_centre_skill"].any()
    cells = pd.read_csv(tables / "cells.csv")
    assert set(cells["model"]) == set(models)
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "yes", "Q2": "yes"}
    text = readme.read_text()
    assert (
        "9 of 18 cells (18 with a centre test) of the main family (models with at least 3 "
        "scored quarters wholly after release) have a median that predicts the sign better "
        "than climatology: centre-only crypto 1h"
    ) in text
    assert "Late family (its own Holm correction and verdict, no): 0 of 0 cells: none." in text
    assert "9 of 18 cells are calibrated and 9 track risk better than ewma-normal; both: " in text
    assert "both: scale-only crypto 1h" in text
    results = study / "results" / "synthetic-run"
    for name in ("tables/verdict.csv", "tables/rq2.csv", "figures/rq2.pdf"):
        assert (results / name).exists()
    figure = pd.read_csv(tables / "rq2.csv")
    assert list(dict.fromkeys(figure["model"])) == ["ewma-normal", "centre-only", "scale-only"]
    assert (figure["horizon"] == 1).all()


@pytest.mark.parametrize(
    ("model", "answers"),
    [("centre-only", {"Q1": "yes", "Q2": "no"}), ("scale-only", {"Q1": "no", "Q2": "yes"})],
)
def test_rq2_answers_each_question_on_its_own(
    tmp_path: Path,
    centre_scale_cache_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    answers: dict[str, str],
) -> None:
    study = _run(tmp_path, centre_scale_cache_folder, [*BASELINES, model], monkeypatch)
    assert read_answers(study / "results" / "README.md") == answers


def test_rq2_refuses_a_missing_cache(
    tmp_path: Path, centre_scale_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, centre_scale_cache_folder, [*BASELINES, "scale-only"])
    config = (study / "study.toml").read_text().replace('"synthetic"', '"absent"')
    (study / "study.toml").write_text(config)
    with pytest.raises(FileNotFoundError, match="no forecast cache in"):
        _main("run.py", study, monkeypatch)
    assert not (study / "runs").exists()


def test_run_scores_only_the_models_given_with_models(
    tmp_path: Path, centre_scale_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, centre_scale_cache_folder, [*BASELINES, "centre-only", "scale-only"])
    _main("run.py", study, monkeypatch, "--models", "centre-only")
    cells = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "cells.csv")
    assert set(cells["model"]) == {"centre-only", "ewma-normal"}


def test_q1_is_inconclusive_when_no_main_cell_has_a_centre_test(
    tmp_path: Path, centre_scale_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a zero-shot model whose median is always zero has no sign, so no cell is tested
    folder = tmp_path / "cache"
    shutil.copytree(centre_scale_cache_folder, folder)
    shutil.copy(
        folder / "forecasts" / "ewma-normal.parquet", folder / "forecasts" / "scale-only.parquet"
    )
    study = _run(tmp_path / "study", folder, [*BASELINES, "scale-only"], monkeypatch)
    answers = read_answers(study / "results" / "README.md")
    assert answers["Q1"] == "inconclusive"
    assert (
        "No cell of the main family has a centre test"
        in (study / "results" / "README.md").read_text()
    )
