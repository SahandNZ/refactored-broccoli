import json
import re

import pytest

from src import LAB_DIR
from src.experiment import META_FIELDS, load_study, read_answers, read_questions

EXPERIMENTS = LAB_DIR / "experiments"
NAME = re.compile(r"\d\d-(rq\d+-)?[a-z0-9-]+")
GROUPS = [EXPERIMENTS / "00-template"]
"""The folder of studies that are not the benchmark's: the template."""
STUDIES = sorted(
    study
    for folder in [EXPERIMENTS, *GROUPS]
    for study in folder.iterdir()
    if study.is_dir() and NAME.fullmatch(study.name) and study not in GROUPS
)
FILES = {"README.md", "study.toml", "run.py", "analyze.py"}
SECTIONS = [
    "Questions",
    "Hypothesis",
    "Design",
    "Data",
    "Metrics",
    "Decision rule",
    "Compute",
    "Status",
]
RESEARCH_QUESTIONS = {
    question
    for readme in [EXPERIMENTS / "README.md", *(group / "README.md" for group in GROUPS)]
    for question in re.findall(r"\| (RQ-[a-z-]+) \|", readme.read_text())
}
"""Every research question: the benchmark's in experiments/README.md, a group's in its own."""


def _shared_files(study):
    # the files of a study folder besides its own lib/, runs/ and results/, its table
    # generators make_tab_<rq>_<content>.py and hidden entries (an
    # editor's cache such as Jupyter's .ipynb_checkpoints/ is not part of the study)
    return {
        path.name
        for path in study.iterdir()
        if path.name not in {"__pycache__", "results", "runs", "lib"}
        and not path.name.startswith(".")
        and not (path.name.startswith("make_tab_") and path.suffix == ".py")
    }


@pytest.mark.parametrize("study", STUDIES, ids=lambda path: path.name)
def test_study_folder_has_exactly_the_shared_files(study):
    assert _shared_files(study) == FILES


def test_a_table_generator_is_allowed_next_to_the_shared_files(tmp_path):
    study = tmp_path / "03-rq1-null"
    study.mkdir()
    for name in [*FILES, "make_tab_rq1_skill.py"]:
        (study / name).write_text("")
    assert _shared_files(study) == FILES
    (study / "notes.txt").write_text("")
    (study / "make_tab_rq1_skill.txt").write_text("")
    assert _shared_files(study) == FILES | {"notes.txt", "make_tab_rq1_skill.txt"}


def test_hidden_entries_are_not_part_of_a_study(tmp_path):
    study = tmp_path / "03-rq1-null"
    study.mkdir()
    for name in FILES:
        (study / name).write_text("")
    (study / ".ipynb_checkpoints").mkdir()
    (study / ".DS_Store").write_text("")
    assert _shared_files(study) == FILES


@pytest.mark.parametrize("study", STUDIES, ids=lambda path: path.name)
def test_study_lib_is_a_package(study):
    lib = study / "lib"
    if not lib.exists():
        pytest.skip("no study-owned code")
    assert lib.is_dir()
    assert (lib / "__init__.py").exists(), f"{lib} must be a package (with __init__.py)"


@pytest.mark.parametrize("study", STUDIES, ids=lambda path: path.name)
def test_readme_sections_and_questions(study):
    readme = study / "README.md"
    headings = re.findall(r"^## (.+)$", readme.read_text(), re.M)
    assert headings == SECTIONS
    for question in read_questions(readme):
        assert question.research_question in RESEARCH_QUESTIONS
        assert question.decision_rule


@pytest.mark.parametrize("study", STUDIES, ids=lambda path: path.name)
def test_study_toml_parses_and_names_its_folder(study):
    assert load_study(study / "study.toml").name == study.name


@pytest.mark.parametrize("study", STUDIES, ids=lambda path: path.name)
def test_results_answer_every_question_and_runs_record_their_metadata(study):
    results = study / "results"
    if not (results / "README.md").exists():
        pytest.skip("no results yet")
    questions = {question.id for question in read_questions(study / "README.md")}
    assert set(read_answers(results / "README.md")) == questions
    for meta in (study / "runs").glob("*/meta.json"):
        assert set(META_FIELDS) <= set(json.loads(meta.read_text()))


@pytest.mark.parametrize("group", GROUPS, ids=lambda path: path.name)
def test_every_group_folder_has_a_readme_and_only_studies(group):
    assert (group / "README.md").exists()
    for child in group.iterdir():
        assert child.name in {"README.md", "forecasts"} or (
            child.is_dir() and NAME.fullmatch(child.name)
        )


def test_experiments_holds_only_studies_and_groups():
    for child in EXPERIMENTS.iterdir():
        assert (
            child.name == "README.md"
            or child.name.startswith(".")
            or (child.is_dir() and NAME.fullmatch(child.name))
        )
