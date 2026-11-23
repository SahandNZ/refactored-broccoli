"""05-rq3-net end to end: run.py and analyze.py on the synthetic cache, in a temporary paper."""

import json
import re
import runpy
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src import LAB_DIR
from src.experiment import load_study, read_answers

ALL_MODELS = ["random-walk", "climatology", "skilled", "sampled"]
STUDY = LAB_DIR / "experiments" / "05-rq3-net"
CODE = [
    STUDY / "run.py",
    STUDY / "analyze.py",
    STUDY / "make_tab_rq3_break_even.py",
    *(STUDY / "lib").glob("*.py"),
    *(LAB_DIR / "src" / "benchmark").glob("_*.py"),
]


# the scripts import the study's lib as `lib`, as when they run from their own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


@pytest.fixture(autouse=True)
def _paper_names(rq3: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    # the synthetic zero-shot models get names for the paper's table bodies
    monkeypatch.setitem(rq3._table.MODEL_NAMES, "skilled", "Skilled")
    monkeypatch.setitem(rq3._table.MODEL_NAMES, "sampled", "Sampled")


def _study(tmp_path: Path, cache_folder: Path, models: list[str]) -> Path:
    paper = tmp_path / "experiments"
    study = paper / "05-rq3-net"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    config = (STUDY / "study.toml").read_text()
    # the synthetic cache stands in for the study's own (whatever its plan and cache are)
    config = re.sub(r"(?m)^locked = .*$", "locked = true", config)
    config = re.sub(r'(?m)^use = ".*"$', 'use = "test"', config)
    config = re.sub(r'(?m)^forecast_cache = ".*"$', 'forecast_cache = "quarters"', config)
    config = config.replace("bootstrap_resamples = 2000", "bootstrap_resamples = 300")
    # the synthetic cache has no ridge regression: climatology stands in as the reference
    config = config.replace('reference = "linear"', 'reference = "climatology"')
    start = config.index("[models]")
    end = config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in models) + "]\n\n"
    (study / "study.toml").write_text(config[:start] + names + config[end:])
    shutil.copytree(cache_folder, paper / "02-forecasts" / "runs" / "quarters")
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


def test_rq3_answers_yes_for_the_skilled_model_only(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    tables = study / "runs" / "synthetic-run" / "tables"
    cells = pd.read_csv(tables / "cells.csv")
    assert len(cells) == 36 + 3
    assert set(pd.read_csv(tables / "reference.csv")["model"]) == {"climatology", "buy-and-hold"}
    verdict = pd.read_csv(tables / "verdict.csv")
    assert set(verdict["model"]) == {"skilled", "sampled"}
    assert set(verdict["family"]) == {"main"}
    assert verdict.loc[verdict["model"] == "skilled", "earns_after_costs"].all()
    assert not verdict.loc[verdict["model"] == "sampled", "earns_after_costs"].any()
    assert set(pd.read_csv(tables / "delayed.csv")["model"]) == {
        "random-walk",
        "climatology",
        "skilled",
    }
    # the sampled model holds its short at h = 4 (a 4-hour label every 4 hours): low turnover
    flags = verdict.set_index(["model", "market", "horizon"])["is_low_turnover"]
    assert flags[("sampled", "crypto", 4)]
    assert not flags[("skilled", "crypto", 4)]
    assert verdict["turnover_per_effective"].notna().any()
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "yes"}
    text = readme.read_text()
    assert "buy-and-hold (reference), break-even at 1h: crypto " in text
    assert "cells trade less than 0.05 per effective time (is_low_turnover" in text
    assert "Holm per family over all horizons and markets" in text
    assert "Highest break-even per market at 1h: crypto skilled " in text
    # skilled is scored in its 9 cells, sampled in 8; only skilled has a delayed book
    assert "One-bar delay (robustness, delayed.csv): in 9 scored cells with a delayed book" in text
    assert "Main family: 9 of 17 scored cells earn money after costs: skilled crypto 1h" in text
    assert "Late-release family (own verdict): 0 of 0 scored cells" in text
    assert "The break-even cost per side is above the cost paid in" in text
    for name in ("quarters.csv", "verdict.csv", "reference.csv", "delayed.csv", "costs.csv"):
        assert (study / "results" / "synthetic-run" / "tables" / name).exists()
    assert (study / "runs" / "test-log.csv").exists()


def test_the_table_script_writes_both_bodies_from_the_run(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    out = tmp_path / "generated"
    arguments = ["make_tab_rq3_break_even.py", "--study-dir", str(study), "--run-id"]
    monkeypatch.setattr(sys, "argv", [*arguments, "synthetic-run", "--out-dir", str(out)])
    runpy.run_path(str(STUDY / "make_tab_rq3_break_even.py"), run_name="__main__")
    body = (out / "rq3_body.tex").read_text()
    assert body.startswith(
        "% Generated by experiments/05-rq3-net/make_tab_rq3_break_even.py from "
        "experiments/05-rq3-net/runs/synthetic-run/tables/"
    )
    assert "Skilled & \\textbf{" in body
    assert "Climatology & -- & -- & -- \\\\" in body
    assert "\nBuy and hold & " not in body
    assert "\nMean cost per side & 5.0 & 0.4 & 1.0 \\\\" in body
    main = (out / "rq3_appendix_main_body.tex").read_text()
    assert "95\\% CI" in main
    assert "Skilled & Crypto & " in main
    assert "Climatology" not in main
    late = (out / "rq3_appendix_late_body.tex").read_text()
    assert "Skilled" not in late
    assert "Climatology & Crypto & " in late
    assert "Buy and hold & Crypto & " in late
    assert sorted(path.name for path in out.iterdir()) == [
        "rq3_appendix_late_body.tex",
        "rq3_appendix_main_body.tex",
        "rq3_body.tex",
    ]


def test_the_table_script_writes_into_the_paper_generated_tables() -> None:
    script = runpy.run_path(str(STUDY / "make_tab_rq3_break_even.py"), run_name="not_main")
    assert script["OUT_DIR"] == LAB_DIR / "paper" / "tables" / "generated"


def test_the_tables_are_at_horizon_one(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    config = study / "runs" / "synthetic-run" / "config.toml"
    config.write_text(config.read_text().replace("table_horizon = 1", "table_horizon = 4"))
    with pytest.raises(ValueError, match="table_horizon must be 1"):
        _main("analyze.py", study, monkeypatch)


def test_the_table_script_needs_the_analysed_run(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    _main("run.py", study, monkeypatch)
    arguments = ["make_tab_rq3_break_even.py", "--study-dir", str(study), "--run-id"]
    monkeypatch.setattr(sys, "argv", [*arguments, "synthetic-run", "--out-dir", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match=r"verdict\.csv"):
        runpy.run_path(str(STUDY / "make_tab_rq3_break_even.py"), run_name="__main__")


def test_rq3_answers_no_with_none_without_skill(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ["random-walk", "climatology", "sampled"])
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "no"}
    assert "Main family: 0 of 8 scored cells earn money after costs: none." in readme.read_text()


def test_a_late_release_that_earns_does_not_answer_yes(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the skilled model released mid-way through t2 has only 2 post-release quarters: it
    # earns in the late-release family, and Q1 (the main family) stays no
    study = _study(tmp_path, quarters_cache_folder, ["climatology", "skilled", "sampled"])
    models_file = study.parent / "02-forecasts" / "runs" / "quarters" / "models.csv"
    models = pd.read_csv(models_file)
    models.loc[models["model"] == "skilled", "release_date"] = "2026-03-01"
    models.to_csv(models_file, index=False)
    _main("run.py", study, monkeypatch)
    _main("analyze.py", study, monkeypatch)
    verdict = pd.read_csv(study / "runs" / "synthetic-run" / "tables" / "verdict.csv")
    skilled = verdict[verdict["model"] == "skilled"]
    assert (skilled["family"] == "late").all()
    assert skilled["earns_after_costs"].any()
    readme = study / "results" / "README.md"
    assert read_answers(readme) == {"Q1": "no"}
    text = readme.read_text()
    assert "Main family: 0 of 8 scored cells earn money after costs: none." in text
    assert "Late-release family (own verdict): " in text
    assert "skilled crypto 1h" in text


def test_a_rerun_reproduces_the_cells(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ["climatology", "skilled"])
    _main("run.py", study, monkeypatch, run_id="first")
    _main("run.py", study, monkeypatch, run_id="second")
    first = pd.read_csv(study / "runs" / "first" / "tables" / "cells.csv")
    second = pd.read_csv(study / "runs" / "second" / "tables" / "cells.csv")
    pd.testing.assert_frame_equal(first, second)


def test_rq3_refuses_a_missing_cache(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    config = (study / "study.toml").read_text().replace('"quarters"', '"absent"')
    (study / "study.toml").write_text(config)
    with pytest.raises(FileNotFoundError, match="no forecast cache in"):
        _main("run.py", study, monkeypatch)
    assert not (study / "runs").exists()


def test_a_dry_run_writes_nothing(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)
    monkeypatch.setattr(sys, "argv", ["run.py", "--study-dir", str(study), "--dry-run"])
    runpy.run_path(str(STUDY / "run.py"), run_name="__main__")
    assert not (study / "runs").exists()


def test_a_failed_run_is_recorded_as_failed(
    tmp_path: Path, quarters_cache_folder: Path, monkeypatch: pytest.MonkeyPatch, rq3: ModuleType
) -> None:
    study = _study(tmp_path, quarters_cache_folder, ALL_MODELS)

    def broken(*arguments: object, **options: object) -> None:
        raise RuntimeError("broken scoring")

    monkeypatch.setattr(rq3, "net_tables", broken)
    with pytest.raises(RuntimeError, match="broken scoring"):
        _main("run.py", study, monkeypatch)
    meta = json.loads((study / "runs" / "synthetic-run" / "meta.json").read_text())
    assert meta["status"] == "failed"
    assert "broken scoring" in meta["exit_reason"]
