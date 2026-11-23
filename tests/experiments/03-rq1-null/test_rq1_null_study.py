"""03-rq1-null end to end: run.py and analyze.py on the synthetic cache, in a temporary paper."""

import re
import runpy
import shutil
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.experiment import read_answers

ALL_MODELS = ["random-walk", "climatology", "skilled", "sampled"]
STUDY = LAB_DIR / "experiments" / "03-rq1-null"


# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


def test_the_scripts_find_this_studys_lib_as_lib(
    study_lib_as_lib: ModuleType, load_study_lib: Callable[[Path], ModuleType]
) -> None:
    assert sys.modules["lib"] is study_lib_as_lib
    assert study_lib_as_lib is load_study_lib(STUDY)


def _study(tmp_path: Path, cache_folder: Path, models: list[str]) -> Path:
    paper = tmp_path / "experiments"
    study = paper / "03-rq1-null"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    config = (STUDY / "study.toml").read_text()
    config = config.replace("locked = false", "locked = true")
    config = config.replace('use = "development"', 'use = "test"')
    config = re.sub(r'forecast_cache = ".*"', 'forecast_cache = "synthetic"', config)
    config = config.replace("bootstrap_resamples = 1000", "bootstrap_resamples = 100")
    config = config.replace("skill_resamples = 2000", "skill_resamples = 200")
    start = config.index("[models]")
    end = config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in models) + "]\n\n"
    (study / "study.toml").write_text(config[:start] + names + config[end:])
    shutil.copytree(cache_folder, paper / "02-forecasts" / "runs" / "synthetic")
    # the size check of 09-mde-threshold, which the Q1 limitations cite
    size = paper / "09-mde-threshold" / "results" / "synthetic-size" / "tables"
    size.mkdir(parents=True)
    (size.parent.parent / "README.md").write_text("# Results\n\nRuns: synthetic-size.\n")
    pd.DataFrame(
        {
            "length": [600, 2400],
            "floor": [4, 4],
            "median_blocks": [6.0, 26.0],
            "mean_test_size": [0.07, 0.05],
            "difference_test_size": [0.06, 0.065],
            "draws": [300, 300],
            "nominal": [0.05, 0.05],
        }
    ).to_csv(size / "bootstrap_size.csv", index=False)
    return study


def _main(script: str, study: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = [script, "--study-dir", str(study), "--run-id", "synthetic-run"]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(STUDY / script), run_name="__main__")


def test_rq1_answers_yes_for_the_skilled_model_only(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    verdict = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "verdict.csv")
    assert len(verdict) == 18
    beats = verdict.groupby("model")["beats_baseline"].all()
    assert beats["skilled"]
    assert not verdict.loc[verdict["model"] == "sampled", "beats_baseline"].any()
    readme = study / "results" / "README.md"
    answers = read_answers(readme)
    assert answers["Q1"] == "yes"
    assert answers["Q2"] in {"yes", "no"}
    text = readme.read_text()
    assert "09-mde-threshold synthetic-size" in text
    assert "the difference test 6.0%-6.5% at a nominal 5%" in text
    assert "exceed ten times it" in text
    assert (
        "Main verdict, main family (sampled, skilled): 9 of 18 scored cells beat the best " in text
    )
    assert "skilled crypto 1 bars (skill " in text
    assert "Separate verdict, late family (none): 0 of 0 scored cells" in text
    assert "no late model has a scored cell" in text
    assert "scored cells at 1 bar(s)" in text
    assert "against K = 80% (sensitivity, not deciding: K = 50%: " in text
    assert "its MDE (80 % power) is" in text
    for name in ("quarters.csv", "skill.csv", "verdict.csv"):
        assert (study / "results" / "synthetic-run" / "tables" / name).exists()
    skill = pd.read_csv(study / "results" / "synthetic-run" / "tables" / "skill.csv")
    assert {"skill", "skill_low", "skill_high", "family", "beats_baseline"} <= set(skill.columns)
    assert (study / "runs" / "test-log.csv").exists()


def test_rq1_answers_no_with_none_for_a_model_without_skill(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ["random-walk", "climatology", "sampled"])
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    readme = study / "results" / "README.md"
    assert read_answers(readme)["Q1"] == "no"
    assert set(read_answers(readme)) == {"Q1", "Q2"}
    assert "main family (sampled): 0 of 9 scored cells beat the best baseline: none" in (
        readme.read_text()
    )


def test_a_late_model_gets_its_own_verdict_and_mde(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    models = study.parent / "02-forecasts" / "runs" / "synthetic" / "models.csv"
    table = pd.read_csv(models)
    table.loc[table["model"] == "skilled", "release_date"] = "2026-03-01"
    table.to_csv(models, index=False)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    readme = study / "results" / "README.md"
    text = readme.read_text()
    assert read_answers(readme)["Q1"] == "no"
    assert "Main verdict, main family (sampled): 0 of 9 scored cells" in text
    assert "Separate verdict, late family (skilled): " in text
    assert (
        "the late models' MDE at 1 bar(s): skilled: 2 quarter(s), the rule cannot detect it" in text
    )


def test_rq1_refuses_a_missing_cache(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    config = (study / "study.toml").read_text().replace('"synthetic"', '"absent"')
    (study / "study.toml").write_text(config)
    with pytest.raises(FileNotFoundError, match="no forecast cache in"):
        _main("run.py", study, monkeypatch)
    assert not (study / "runs").exists()


def test_rq1_writes_the_power_tables_and_answers_q2(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    run = study / "runs" / "synthetic-run"
    power = pd.read_csv(run / "tables" / "power.csv")
    mde = pd.read_csv(run / "tables" / "mde.csv")
    assert len(mde) == 18
    assert len(power) == 18 * 21
    assert (power.loc[power["improvement_pct"] == 0, "detection_rate"] < 0.05).all()
    np.testing.assert_allclose(mde["false_positive_level"], 0.05 / 18)
    names = ("tables/mde.csv", "tables/mde_threshold.csv", "figures/mde_threshold.pdf",
             "tables/power.csv", "tables/power_rule.csv")  # fmt: skip
    for name in names:
        assert (study / "results" / "synthetic-run" / name).exists()
    assert not (run / "figures" / "mde.pdf").exists()
    at_one_hour = mde[mde["horizon"] == 1]
    resolved = at_one_hour["is_reached"] & (at_one_hour["mde_pct"] <= 2.0)
    expected = "yes" if resolved.mean() >= 0.8 else "no"
    assert read_answers(study / "results" / "README.md")["Q2"] == expected
    assert (
        f"{int(resolved.sum())} of the {len(at_one_hour)} scored cells at 1 bar(s)"
        in (study / "results" / "README.md").read_text()
    )
    rule = pd.read_csv(run / "tables" / "power_rule.csv")
    assert rule["share_needed"].tolist() == [0.8, 0.5, 0.9]
    assert rule["is_main"].tolist() == [True, False, False]
    assert (rule["is_powered"] == (resolved.mean() >= rule["share_needed"])).all()


def test_rq1_analyses_with_the_runs_own_configuration(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    config = (study / "study.toml").read_text()
    changed = config.replace("min_winning_quarters = 3", "min_winning_quarters = 9")
    (study / "study.toml").write_text(
        changed.replace("mde_threshold_pct = 2.0", "mde_threshold_pct = 10.0")
    )
    _main("analyze.py", study, monkeypatch)
    assert read_answers(study / "results" / "README.md")["Q1"] == "yes"
    readme = (study / "results" / "README.md").read_text()
    assert "have an MDE at or below X = 2 % of the best" in readme


def test_rq1_q2_answers_yes_when_the_threshold_is_above_every_mde(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    config = (study / "study.toml").read_text()
    config = config.replace("mde_threshold_pct = 2.0", "mde_threshold_pct = 10.0")
    (study / "study.toml").write_text(config)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    mde = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "mde.csv")
    assert mde.loc[mde["horizon"] == 1, "is_reached"].all()
    assert np.nanmax(mde.loc[mde["horizon"] == 1, "mde_pct"]) <= 10.0
    assert read_answers(study / "results" / "README.md")["Q2"] == "yes"


def test_every_option_of_rq1_is_read_by_its_code() -> None:
    # the options are read by the study's scripts, its lib or the settings of src.benchmark
    options = tomllib.loads((STUDY / "study.toml").read_text())["options"]
    sources = [
        STUDY / "run.py",
        STUDY / "analyze.py",
        *(STUDY / "lib").glob("*.py"),
        *(LAB_DIR / "src" / "benchmark").glob("_*.py"),
    ]
    code = "".join(path.read_text() for path in sources)
    assert [key for key in options if f'options["{key}"]' not in code] == []
