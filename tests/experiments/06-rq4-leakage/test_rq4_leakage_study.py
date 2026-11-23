"""06-rq4-leakage end to end: run.py and analyze.py on the synthetic cache, in a temporary paper."""

import json
import runpy
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import cache_settings
from src.experiment import load_study, read_answers
from src.figures import MODEL_FAMILY

ALL_MODELS = ["random-walk", "climatology", "skilled", "sampled", "leaky"]
STUDY = LAB_DIR / "experiments" / "06-rq4-leakage"
CODE = [
    STUDY / "run.py",
    STUDY / "analyze.py",
    *(STUDY / "lib").glob("*.py"),
    *(LAB_DIR / "src" / "benchmark").glob("_*.py"),
]


# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


@pytest.fixture(autouse=True)
def _synthetic_model_styles(monkeypatch: pytest.MonkeyPatch) -> None:
    # fig:rq4 draws each model in its family's style; the synthetic models borrow three
    monkeypatch.setitem(MODEL_FAMILY, "skilled", ("Chronos-2", ""))
    monkeypatch.setitem(MODEL_FAMILY, "sampled", ("Kronos", "small"))
    monkeypatch.setitem(MODEL_FAMILY, "leaky", ("TimesFM 2.5", ""))


def _study(
    plan_text: Callable[..., str],
    tmp_path: Path,
    cache_folder: Path,
    models: list[str],
    placebo: tuple[str, ...] = (),
) -> Path:
    paper = tmp_path / "experiments"
    study = paper / "06-rq4-leakage"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    config = plan_text(
        (STUDY / "study.toml").read_text(),
        locked=True,
        use="test",
        forecast_cache="leakage",
        leakage_cache="",
        bootstrap_resamples=300,
        not_built=[],
        placebo_models=list(placebo),
    )
    start = config.index("[models]")
    end = config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in models) + "]\n\n"
    (study / "study.toml").write_text(config[:start] + names + config[end:])
    shutil.copytree(cache_folder, paper / "02-forecasts" / "runs" / "leakage")
    return study


def _main(
    script: str, study: Path, monkeypatch: pytest.MonkeyPatch, run_id: str = "synthetic-run"
) -> None:
    arguments = [script, "--study-dir", str(study), "--run-id", run_id]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(STUDY / script), run_name="__main__")


def test_every_option_of_the_plan_is_read_by_the_code() -> None:
    # the options are read by the study's scripts and lib or by the settings of src.benchmark
    options = load_study(STUDY / "study.toml").options
    code = "".join(path.read_text() for path in CODE)
    assert [name for name in options if f'options["{name}"]' not in code] == []


def test_the_after_grid_is_the_step_of_the_leakage_quarters() -> None:
    forecasts = cache_settings(load_study(STUDY.parent / "02-forecasts" / "study.toml"))
    options = load_study(STUDY / "study.toml").options
    assert options["grid_hours"] == forecasts.step_hours["leakage"] == 4


def test_rq4_answers_yes_for_the_leaky_model_only(
    tmp_path: Path,
    leakage_cache_folder: Path,
    plan_text: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(plan_text, tmp_path, leakage_cache_folder, ALL_MODELS, placebo=("skilled",))
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    tables = study / "runs" / "synthetic-run" / "tables"
    verdict = pd.read_csv(tables / "verdict.csv")
    # 9 cells each: leaky leakage, hourly, cutoff; skilled leakage, hourly, placebo; sampled
    # (on its own grid) leakage, cutoff
    assert len(verdict) == 8 * 9
    primary = verdict[verdict["test"] == "leakage"]
    assert len(primary) == 27
    assert primary.loc[primary["model"] == "leaky", "shows_leakage"].all()
    assert not verdict.loc[verdict["model"] != "leaky", "shows_leakage"].any()
    assert not verdict.loc[verdict["test"] != "leakage", "shows_leakage"].any()
    periods = pd.read_csv(tables / "periods.csv")
    leaky_periods = set(periods.loc[periods["model"] == "leaky", "period"])
    assert leaky_periods == {"before", "between", "after", "after-hourly"}
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "yes"}
    text = readme.read_text()
    assert "9 of 27 scored cells are better before the release than after" in text
    assert "leaky crypto 1h (did 0.2" in text
    assert (
        "Check placebo (skill early minus late, report only): Holm rejects 0 of 9 scored cells"
        in text
    )
    # fig:rq4: the leakage cells at 1 bar, one row per model and market, leaky filled
    figure = pd.read_csv(tables / "rq4.csv")
    assert len(figure) == 9  # 3 models x 3 markets of the synthetic cache
    assert set(figure["horizon"]) == {1}
    assert figure.loc[figure["model"] == "leaky", "shows_leakage"].all()
    assert not figure.loc[figure["model"] != "leaky", "shows_leakage"].any()
    copied = study / "results" / "synthetic-run"
    for name in ("tables/rq4.csv", "tables/not_fitted.csv", "figures/rq4.pdf"):
        assert (copied / name).exists(), name
    assert "Check cutoff (skill before minus between, report only):" in text
    assert "one-sided test" in text
    assert "did not converge, left out (02-forecasts not_fitted.csv): none" in text
    assert (tables / "not_fitted.csv").exists()
    assert (study / "results" / "synthetic-run" / "tables" / "periods.csv").exists()
    assert (study / "runs" / "test-log.csv").exists()


def test_rq4_answers_no_with_none_for_honest_models(
    tmp_path: Path,
    leakage_cache_folder: Path,
    plan_text: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(
        plan_text,
        tmp_path,
        leakage_cache_folder,
        ["random-walk", "climatology", "skilled"],
        ("skilled",),
    )
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "no"}
    assert "0 of 9 scored cells are better before the release than after" in readme.read_text()
    assert ": none." in readme.read_text()


def test_a_rerun_reproduces_the_cells(
    tmp_path: Path,
    leakage_cache_folder: Path,
    plan_text: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(plan_text, tmp_path, leakage_cache_folder, ["climatology", "leaky"])
    _main("run.py", study, monkeypatch, run_id="first")
    _main("run.py", study, monkeypatch, run_id="second")
    first = pd.read_csv(study / "runs" / "first" / "tables" / "cells.csv")
    second = pd.read_csv(study / "runs" / "second" / "tables" / "cells.csv")
    pd.testing.assert_frame_equal(first, second)


def test_rq4_refuses_a_missing_cache(
    tmp_path: Path,
    leakage_cache_folder: Path,
    plan_text: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(plan_text, tmp_path, leakage_cache_folder, ALL_MODELS)
    config = (study / "study.toml").read_text().replace('"leakage"', '"absent"')
    (study / "study.toml").write_text(config)
    with pytest.raises(FileNotFoundError, match="no forecast cache in"):
        _main("run.py", study, monkeypatch)
    assert not (study / "runs").exists()


def test_a_failed_run_is_recorded_as_failed(
    leakage: ModuleType,
    tmp_path: Path,
    leakage_cache_folder: Path,
    plan_text: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study(plan_text, tmp_path, leakage_cache_folder, ALL_MODELS)

    def broken(*arguments: object, **options: object) -> None:
        raise RuntimeError("broken scoring")

    monkeypatch.setattr(leakage, "leakage_tables", broken)
    with pytest.raises(RuntimeError, match="broken scoring"):
        _main("run.py", study, monkeypatch)
    meta = json.loads((study / "runs" / "synthetic-run" / "meta.json").read_text())
    assert meta["status"] == "failed"
    assert "broken scoring" in meta["exit_reason"]
