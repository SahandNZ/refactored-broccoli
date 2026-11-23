"""09-mde-threshold end to end: run.py and analyze.py on the synthetic cache, in a temp paper."""

import re
import runpy
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.experiment import read_answers

MODELS = ["random-walk", "climatology", "skilled", "sampled"]
STUDY = LAB_DIR / "experiments" / "09-mde-threshold"
RQ1 = STUDY.parent / "03-rq1-null"

# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")

SMALL_SIZE_CHECK = [
    ("size_lengths = [600, 2400]", "size_lengths = [200]"),
    ("size_draws = 300", "size_draws = 4"),
    ("size_resamples = 300", "size_resamples = 20"),
]
"""A size check small enough for the tests."""


def _study(tmp_path, cache_folder, replacements=(), rq1_replacements=()):
    # 09 in a temporary paper with a copy of 03-rq1-null's study.toml, whose rule it reads
    paper = tmp_path / "experiments"
    study = paper / "09-mde-threshold"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    config = (STUDY / "study.toml").read_text()
    config = re.sub(r'forecast_cache = ".*"', 'forecast_cache = "synthetic"', config)
    for old, new in [*SMALL_SIZE_CHECK, *replacements]:
        assert old in config
        config = config.replace(old, new)
    (study / "study.toml").write_text(config)
    rq1 = (
        (RQ1 / "study.toml")
        .read_text()
        .replace("bootstrap_resamples = 1000", "bootstrap_resamples = 100")
    )
    for old, new in rq1_replacements:
        assert old in rq1
        rq1 = rq1.replace(old, new)
    start = rq1.index("[models]")
    end = rq1.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in MODELS) + "]\n\n"
    rq1 = rq1[:start] + names + rq1[end:]
    (paper / "03-rq1-null").mkdir()
    (paper / "03-rq1-null" / "study.toml").write_text(rq1)
    shutil.copytree(cache_folder, paper / "02-forecasts" / "runs" / "synthetic")
    return study


def _main(script, study, monkeypatch):
    arguments = [script, "--study-dir", str(study), "--run-id", "synthetic-run"]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(STUDY / script), run_name="__main__")


def test_the_threshold_is_the_smallest_reaching_the_share_on_the_development_folds(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    run = study / "runs" / "synthetic-run"
    shares = pd.read_csv(run / "tables" / "mde_threshold.csv")
    assert shares["threshold_pct"].tolist() == pytest.approx(np.arange(0, 10.5, 0.5))
    for column in ["share", "share_1h", "share_4h", "share_24h"]:
        assert (np.diff(shares[column].to_numpy()) >= 0).all()
    assert shares["share_1h"].iloc[-1] >= 0.8
    chosen = shares.loc[shares["share_1h"] >= 0.8, "threshold_pct"].iloc[0]
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "yes"}
    assert f"threshold {chosen:g} % of the best baseline's CRPS" in readme.read_text()
    assert "cells at 1 bar(s)" in readme.read_text()
    assert (run / "rq1-config.toml").read_text() == (
        study.parent / "03-rq1-null" / "study.toml"
    ).read_text()
    for name in [
        "tables/mde_threshold.csv", "figures/mde_threshold.pdf", "tables/mde.csv",
        "figures/mde.pdf", "tables/power.csv", "tables/bootstrap_size.csv",
    ]:  # fmt: skip
        assert (study / "results" / "synthetic-run" / name).exists()
    mde = pd.read_csv(run / "tables" / "mde.csv")
    assert len(mde) == 2 * 3 * 3


def test_the_study_uses_the_development_folds_only(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder)
    _main("run.py", study, monkeypatch)
    power = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "power.csv")
    assert (power["times"] <= 3 * 200).all()
    study = _study(
        tmp_path / "test", quarters_cache_folder, [('use = "development"', 'use = "test"')]
    )
    with pytest.raises(SystemExit, match="development folds only, got test"):
        _main("run.py", study, monkeypatch)
    assert not (study / "runs").exists()


def test_no_threshold_when_no_cell_is_resolved(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(
        tmp_path, quarters_cache_folder,
        rq1_replacements=[("power_improvements_pct = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, "
                           "4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]",
                           "power_improvements_pct = [0.0, 0.01]")],
    )  # fmt: skip
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "no"}
    assert "short of 80%: no threshold of the grid powers the rule." in readme.read_text()
    assert (study / "runs" / "synthetic-run" / "figures" / "mde.pdf").exists()


def test_the_rule_and_its_settings_live_only_in_rq1() -> None:
    options = tomllib.loads((STUDY / "study.toml").read_text())["options"]
    assert set(options) == {
        "forecast_cache", "size_lengths", "size_floors", "size_draws", "size_resamples",
        "size_seed",
    }  # fmt: skip
    rq1 = tomllib.loads((RQ1 / "study.toml").read_text())["options"]
    assert rq1["mde_threshold_grid_pct"] == pytest.approx(np.arange(0, 10.5, 0.5).tolist())
    assert rq1["power_share"] == 0.8
    assert rq1["power_improvements_pct"] == pytest.approx(np.arange(0, 10.5, 0.5).tolist())


def test_the_threshold_ignores_a_test_fold_setting_of_rq1(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 03-rq1-null itself is on the test folds since its plan was locked (2026-11-17)
    study = _study(
        tmp_path, quarters_cache_folder, rq1_replacements=[('use = "test"', 'use = "test"')]
    )
    _main("run.py", study, monkeypatch)
    power = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "power.csv")
    assert (power["times"] <= 3 * 200).all()


def test_the_size_check_rejects_a_share_of_the_null_draws(load_study_lib) -> None:
    lib = load_study_lib(STUDY)
    table = lib.bootstrap_size_table(lengths=(240,), floors=(2, 4), draws=20, resamples=50, seed=0)
    assert table["floor"].tolist() == [2, 4]
    assert (table["median_blocks"] >= 2).all()
    assert table.loc[1, "median_blocks"] <= table.loc[0, "median_blocks"]
    for column in ("mean_test_size", "difference_test_size"):
        assert ((table[column] >= 0) & (table[column] <= 0.3)).all()
    again = lib.bootstrap_size_table(lengths=(240,), floors=(2, 4), draws=20, resamples=50, seed=0)
    pd.testing.assert_frame_equal(table, again)
