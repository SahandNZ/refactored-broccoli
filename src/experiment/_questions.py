import re
from dataclasses import dataclass
from pathlib import Path

_QUESTION = re.compile(
    r"\*\*(Q\d+)\*\* \((RQ-[a-z-]+)\)\. (.*?)"
    r"\nDecision rule: (.*?)"
    r"\nPrimary metric: (.*?)(?=\n\n|\Z)",
    re.S,
)
_ANSWER = re.compile(r"^\*\*(Q\d+)\*\*: (yes|no|inconclusive)\b", re.M)


@dataclass(frozen=True, slots=True)
class Question:
    """One question of a study plan.

    Attributes
    ----------
    id : str
        ``"Q1"``, ``"Q2"``, ...
    research_question : str
        The paper research question it serves, e.g. ``"RQ-universe"``.
    text : str
        The question, ending in ``?``.
    decision_rule : str
        Which result answers yes and which no.
    primary_metric : str
        The one metric that decides, fixed before the run.
    """

    id: str
    research_question: str
    text: str
    decision_rule: str
    primary_metric: str


def read_questions(readme: Path) -> list[Question]:
    """Read the questions of a study plan (``experiments/NN-name/README.md``).

    Parameters
    ----------
    readme : Path
        The plan; its ``## Questions`` section holds the questions, each a ``**Qn**`` line
        followed by its ``Decision rule:`` and ``Primary metric:`` lines.

    Returns
    -------
    list of Question
        In order of appearance.

    Raises
    ------
    ValueError
        If there is no ``## Questions`` section, no question, a question not ending in ``?``
        or a repeated id.
    """
    text = readme.read_text()
    if "\n## Questions\n" not in text:
        raise ValueError(f"{readme} has no '## Questions' section")
    section = text.split("\n## Questions\n", 1)[1].split("\n## ", 1)[0]
    questions = [
        Question(
            identifier,
            research,
            " ".join(body.split()),
            " ".join(rule.split()),
            metric.strip().rstrip("."),
        )
        for identifier, research, body, rule, metric in _QUESTION.findall(section)
    ]
    if not questions:
        raise ValueError(f"{readme} has no question with a decision rule and a primary metric")
    for question in questions:
        if not question.text.endswith("?"):
            raise ValueError(f"{question.id} of {readme} does not end with '?'")
    identifiers = [question.id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{readme} repeats a question id: {identifiers}")
    return questions


def read_answers(readme: Path) -> dict[str, str]:
    """Read a results README's answers, ``{"Q1": "yes", ...}``, from its ``**Qn**: ...`` lines."""
    return dict(_ANSWER.findall(readme.read_text()))
