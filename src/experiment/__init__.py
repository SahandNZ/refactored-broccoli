"""The shared shape of every study.

Configuration, questions, command line, run folder, seeding and the results README.
"""

from src.experiment._answers import Answer, write_results_readme
from src.experiment._cli import Invocation, parse_invocation
from src.experiment._config import StudyConfig, load_study
from src.experiment._questions import Question, read_answers, read_questions
from src.experiment._runner import (
    META_FIELDS,
    Placement,
    Run,
    resume_run,
    start_run,
    study_key,
    study_results,
    study_runs,
)
from src.experiment._seed import seed_everything

__all__ = [
    "META_FIELDS",
    "Answer",
    "Invocation",
    "Placement",
    "Question",
    "Run",
    "StudyConfig",
    "load_study",
    "parse_invocation",
    "read_answers",
    "read_questions",
    "resume_run",
    "seed_everything",
    "start_run",
    "study_key",
    "study_results",
    "study_runs",
    "write_results_readme",
]
