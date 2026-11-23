"""02-forecasts end to end: run.py and analyze.py on the synthetic lab, in a temporary lab."""

import json
import logging
import runpy
from pathlib import Path

import pandas as pd
import pytest

import src
from src import LAB_DIR
from src.benchmark import _build, cache_settings
from src.benchmark._settings import BASELINES
from src.experiment import load_study, read_answers
from src.models import Forecaster

STUDY = LAB_DIR / "experiments" / "02-forecasts"


# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


def _study(lab_dir: Path, lab: tuple[Path, Path]) -> Path:
    study = lab_dir / "experiments" / "02-forecasts"
    study.mkdir(parents=True)
    (study / "README.md").write_text((STUDY / "README.md").read_text())
    config = (STUDY / "study.toml").read_text()
    for old, new in [("lookback = 512", "lookback = 16"), ("train_days = 730", "train_days = 20"),
                     ('device = "cuda"', 'device = "cpu"'),
                     # the study is locked on the test folds; the synthetic lab has dev data only
                     ('use = "test"', 'use = "development"'),
                     ('spread_window = ["2020-01-01", "2024-01-01"]',
                      'spread_window = ["2024-10-01", "2024-11-01"]')]:  # fmt: skip
        assert old in config
        config = config.replace(old, new)
    start, end = config.index("[models]"), config.index("[seeds]")
    names = ", ".join(f'"{name}"' for name in (*BASELINES, "chronos-bolt-small"))
    (study / "study.toml").write_text(
        f"{config[:start]}[models]\nnames = [{names}]\n\n{config[end:]}"
    )
    universe = study.parent / "01-universe" / "runs" / "2026-11-17T0005-random-1000" / "tables"
    universe.mkdir(parents=True)
    (universe / "universe.csv").write_text(lab[1].read_text())
    (lab_dir / "data").symlink_to(lab[0])
    return study


def _main(script: str, study: Path, *arguments: str) -> None:
    argv = [script, "--study-dir", str(study), "--run-id", "synthetic", *arguments]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("sys.argv", argv)
        runpy.run_path(str(STUDY / script), run_name="__main__")


def test_run_and_analyze_answer_q1_and_a_rerun_resumes(
    lab: tuple[Path, Path],
    built: dict[str, list[Forecaster]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(src, "LAB_DIR", tmp_path)
    study = _study(tmp_path, lab)
    _main("run.py", study, "--models", "random-walk", "chronos-bolt-small")
    folder = study / "runs" / "synthetic"
    assert sorted(path.stem for path in (folder / "forecasts").iterdir()) == [
        "chronos-bolt-small",
        "random-walk",
    ]
    assert json.loads((folder / "meta.json").read_text())["status"] == "completed"
    assert set(pd.read_parquet(folder / "metrics.parquet")["model"]) == {
        "random-walk",
        "chronos-bolt-small",
    }
    # the resumed run builds with its own config.toml, not the changed study.toml
    toml = (study / "study.toml").read_text()
    (study / "study.toml").write_text(
        toml.replace("linear_lags = [1, 2, 4, 8, 24]", "linear_lags = [1]")
    )
    seen_lags = []
    real_linear = _build.linear

    def recording_linear(*arguments, **options):
        seen_lags.append(options["lags"])
        return real_linear(*arguments, **options)

    monkeypatch.setattr(_build, "linear", recording_linear)
    _main("run.py", study)
    assert len(built["fake"]) == 1
    assert len(list((folder / "forecasts").iterdir())) == 6
    assert "resuming run synthetic" in (folder / "run.log").read_text()
    assert set(seen_lags) == {(1, 2, 4, 8, 24)}
    meta = json.loads((folder / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps(meta | {"plan_hash": "changed"}))
    with pytest.raises(ValueError, match="started with plan_hash changed"):
        _main("run.py", study, "--models", "linear")
    (folder / "meta.json").write_text(json.dumps(meta))
    _main("analyze.py", study)
    table = pd.read_csv(folder / "tables" / "validity.csv")
    assert set(table["model"]) == {*BASELINES, "chronos-bolt-small"}
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "yes"}
    assert "lowest share of labelled decisions with a valid forecast" in readme.read_text()
    # a model with too many invalid forecasts in one window answers no, naming that window
    path = folder / "forecasts" / "chronos-bolt-small.parquet"
    frame = pd.read_parquet(path)
    in_d3 = frame["time"].between(
        pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC")
    )
    frame.loc[in_d3, "q10"] = float("nan")
    frame.to_parquet(path)
    _main("analyze.py", study)
    text = readme.read_text()
    assert read_answers(readme) == {"Q1": "no"}
    assert "chronos-bolt-small d3" in text
    assert "d1 yes, d2 yes, d3 no, d4 yes" in text


def test_the_study_configuration_names_how_to_build_every_model() -> None:
    config = load_study(STUDY / "study.toml")
    settings = cache_settings(config)
    assert set(settings.models) == set(BASELINES) | set(settings.zero_shot)
    assert set(settings.release_dates) == set(settings.zero_shot)
    assert settings.horizons == (1, 4, 24)
