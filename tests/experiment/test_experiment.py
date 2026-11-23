import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiment import (
    META_FIELDS,
    Answer,
    Placement,
    load_study,
    read_answers,
    read_questions,
    resume_run,
    seed_everything,
    start_run,
    write_results_readme,
)

PLAN = """# 99 Demo

## Questions

**Q1** (RQ-universe). Is the sky
blue?
Decision rule: yes if it is.
Primary metric: blueness.

**Q2** (RQ-profit). Does it pay?
Decision rule: yes if net Sharpe > 0.
Primary metric: net Sharpe.

## Hypothesis
"""

STUDY = """
[study]
name = "99-demo"
question = "Demo?"
[plan]
locked = false
[data]
tier = 1
markets = ["crypto"]
timeframes = ["1h"]
[folds]
use = "development"
[models]
names = ["naive"]
[seeds]
values = [0, 1]
[compute]
device = "cpu"
estimated_hours = 0.1
estimated_cost_usd = 0.0
[options]
answer = 42
"""


@pytest.fixture
def study(tmp_path):
    folder = tmp_path / "experiments" / "99-demo"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(PLAN)
    (folder / "study.toml").write_text(STUDY)
    return folder


def test_questions_span_lines_and_keep_their_rules(study):
    questions = read_questions(study / "README.md")
    assert [q.id for q in questions] == ["Q1", "Q2"]
    assert questions[0].text == "Is the sky blue?"
    assert questions[1].research_question == "RQ-profit"


def test_question_without_question_mark_is_rejected(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(PLAN.replace("blue?", "blue."))
    with pytest.raises(ValueError, match="'\\?'"):
        read_questions(path)


def test_load_study_and_bad_folds(study, tmp_path):
    config = load_study(study / "study.toml")
    assert config.seeds == (0, 1)
    assert config.options == {"answer": 42}
    bad = tmp_path / "bad.toml"
    bad.write_text(STUDY.replace('use = "development"', 'use = "train"'))
    with pytest.raises(ValueError, match=r"folds\.use"):
        load_study(bad)


def test_run_records_metadata_refuses_overwrite_and_feeds_the_readme(study, tmp_path):
    config = load_study(study / "study.toml")
    placement = Placement("local", None, None, 1)
    run = start_run(
        study,
        "r1",
        config,
        data_dir=tmp_path,
        placement=placement,
    )
    run.finish(pd.DataFrame({"x": [1.0]}))
    meta = json.loads((run.folder / "meta.json").read_text())
    assert set(META_FIELDS) <= set(meta)
    assert meta["status"] == "completed"
    assert meta["seeds"] == [0, 1]
    assert (run.folder / "config.toml").read_text() == STUDY
    with pytest.raises(FileExistsError):
        start_run(
            study,
            "r1",
            config,
            data_dir=tmp_path,
            placement=placement,
        )
    (run.folder / "tables" / "a.csv").write_text("x\n1\n")
    answers = [
        Answer("Q1", "yes", "It is.", "one sky", ("r1/tables/a.csv",)),
        Answer("Q2", "inconclusive", "Not run.", "not run", ()),
    ]
    path = write_results_readme(study, answers, run_ids=["r1"])
    text = path.read_text()
    assert "| r1 |" in text
    assert read_answers(path) == {"Q1": "yes", "Q2": "inconclusive"}
    assert (study / "results" / "r1" / "tables" / "a.csv").read_text() == "x\n1\n"


def test_answers_must_match_the_plan(study, tmp_path):
    with pytest.raises(ValueError, match="do not match"):
        write_results_readme(study, [Answer("Q1", "yes", "", "none", ())], run_ids=[])


def test_finish_rejects_unknown_status(study, tmp_path):
    config = load_study(study / "study.toml")
    run = start_run(study, "r2", config, data_dir=tmp_path,
                    placement=Placement("local", None, None, 1))  # fmt: skip
    with pytest.raises(ValueError, match="status"):
        run.finish(None, status="done")
    run.finish(None, status="failed", reason="test")


def test_test_folds_need_a_locked_plan_and_are_logged(study, tmp_path):
    placement = Placement("local", None, None, 1)
    (study / "study.toml").write_text(STUDY.replace('use = "development"', 'use = "test"'))
    with pytest.raises(PermissionError, match="locked"):
        start_run(
            study,
            "t1",
            load_study(study / "study.toml"),
            data_dir=tmp_path,
            placement=placement,
        )
    (study / "study.toml").write_text(
        STUDY.replace('use = "development"', 'use = "test"').replace(
            "locked = false", "locked = true"
        )
    )
    run = start_run(
        study,
        "t1",
        load_study(study / "study.toml"),
        data_dir=tmp_path,
        placement=placement,
    )
    run.finish(None)
    log = pd.read_csv(study / "runs" / "test-log.csv")
    assert log["run_id"].tolist() == ["t1"]


def test_unknown_location_is_rejected():
    with pytest.raises(ValueError, match="vast-ai"):
        Placement("local-gpu", None, None, 1)


def test_readme_rules_primary_metric_limitations_figures_and_data(study, tmp_path):
    assert read_questions(study / "README.md")[1].primary_metric == "net Sharpe"
    config = load_study(study / "study.toml")
    placement = Placement("local", None, None, 1)
    run = start_run(study, "r1", config, data_dir=tmp_path, placement=placement)
    run.finish(None)
    for figure in ("r1/figures/f.pdf", "r1/figures/f.png"):
        figure_only = [Answer("Q1", "yes", "x", "y", (figure,)), Answer("Q2", "no", "x", "y", ())]
        with pytest.raises(ValueError, match="needs its data"):
            write_results_readme(study, figure_only, run_ids=["r1"])
    no_limits = [Answer("Q1", "yes", "x", " ", ()), Answer("Q2", "no", "x", "y", ())]
    with pytest.raises(ValueError, match="limitations"):
        write_results_readme(study, no_limits, run_ids=["r1"])
    (tmp_path / "manifest.json").write_text("{}")
    second = start_run(study, "r2", config, data_dir=tmp_path, placement=placement)
    second.finish(None)
    good = [Answer("Q1", "yes", "x", "y", ()), Answer("Q2", "no", "x", "y", ())]
    with pytest.raises(ValueError, match="data versions"):
        write_results_readme(study, good, run_ids=["r1", "r2"])


def _meta(run_folder):
    return json.loads((run_folder / "meta.json").read_text())


def test_a_run_as_a_context_manager_records_failures_and_interrupts(study, tmp_path):
    config = load_study(study / "study.toml")
    placement = Placement("local", None, None, 1)
    options = {"data_dir": tmp_path, "placement": placement}
    with pytest.raises(RuntimeError, match="broken"), start_run(study, "f1", config, **options):
        raise RuntimeError("broken")
    assert _meta(study / "runs" / "f1")["status"] == "failed"
    assert "broken" in _meta(study / "runs" / "f1")["exit_reason"]
    with pytest.raises(KeyboardInterrupt), start_run(study, "f2", config, **options):
        raise KeyboardInterrupt
    assert _meta(study / "runs" / "f2")["status"] == "stopped"
    with start_run(study, "f3", config, **options) as run:
        run.write_tables({"cells": pd.DataFrame({"x": [1, 2]})})
        run.finish(None)
    assert _meta(run.folder)["status"] == "completed"
    assert (run.folder / "tables" / "cells.csv").read_text() == "x\n1\n2\n"


def test_a_run_resumes_with_its_own_configuration_from_the_same_code_plan_and_data(study, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    config = load_study(study / "study.toml")
    run = start_run(
        study, "r3", config, data_dir=tmp_path, placement=Placement("local", None, None, 1)
    )
    run.finish(pd.DataFrame({"model": ["a"]}))
    (study / "study.toml").write_text(STUDY.replace("answer = 42", "answer = 7"))
    with resume_run(study, "r3", data_dir=tmp_path) as resumed:
        assert resumed.config.options == {"answer": 42}
        resumed.append_metrics(pd.DataFrame({"model": ["b"]}))
    assert pd.read_parquet(run.folder / "metrics.parquet")["model"].tolist() == ["a", "b"]
    assert _meta(run.folder)["status"] == "completed"
    (study / "README.md").write_text(PLAN + "\nChanged after the start.\n")
    with pytest.raises(ValueError, match=r"r3 started with plan_hash .*: start a new run"):
        resume_run(study, "r3", data_dir=tmp_path)
    (study / "README.md").write_text(PLAN)
    (tmp_path / "manifest.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match=r"r3 started with data_version .*: start a new run"):
        resume_run(study, "r3", data_dir=tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    meta = _meta(run.folder)
    for key, value in {"git_commit": "0" * 40, "git_dirty": "changed"}.items():
        (run.folder / "meta.json").write_text(json.dumps(meta | {key: value}))
        with pytest.raises(ValueError, match=f"r3 started with {key} {value}"):
            resume_run(study, "r3", data_dir=tmp_path)


def _generator_states():
    numpy_state = np.random.get_state()[1]
    return random.getstate(), numpy_state.tolist(), torch.random.get_rng_state().tolist()


def test_seed_everything_seeds_python_numpy_and_torch():
    seed_everything(3)
    first = _generator_states()
    random.seed(99)
    torch.rand(1)
    seed_everything(3)
    assert _generator_states() == first
    seed_everything(4)
    fourth = _generator_states()
    assert all(new != old for new, old in zip(fourth, first, strict=True))
