import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.enums import Market, Timeframe


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """The common configuration of a study, read from its ``study.toml``.

    Attributes
    ----------
    name : str
        The study folder name, ``NN-name``.
    question : str
        The question in one sentence.
    tier : int
        The universe tier (1, 2 or 3; 0 for all candidates).
    markets : tuple of Market
        The markets used.
    timeframes : tuple of Timeframe
        The input timeframes.
    folds : str
        ``"development"`` or ``"test"``.
    models : tuple of str
        The models compared.
    seeds : tuple of int
        Random seeds.
    plan_locked : bool
        Whether the plan is locked (required before any test-fold run).
    device : str
        Torch device.
    estimated_hours : float
        Estimated wall time; the runner stops a run that takes more than twice as long.
    estimated_cost_usd : float
        Estimated GPU cost (0 on the local machine).
    options : dict
        The study's own settings (``[options]`` table), read by its code.
    """

    name: str
    question: str
    tier: int
    markets: tuple[Market, ...]
    timeframes: tuple[Timeframe, ...]
    folds: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    plan_locked: bool
    device: str
    estimated_hours: float
    estimated_cost_usd: float
    options: dict[str, object]


def load_study(path: Path) -> StudyConfig:
    """Parse a ``study.toml`` into a `StudyConfig`.

    Parameters
    ----------
    path : Path
        The file; tables ``[study]``, ``[plan]``, ``[data]``, ``[folds]``, ``[models]``,
        ``[seeds]``, ``[compute]`` and an optional ``[options]``.

    Returns
    -------
    StudyConfig
        The configuration.

    Raises
    ------
    KeyError
        If a required value is missing (there are no defaults).
    ValueError
        If ``folds`` is neither ``"development"`` nor ``"test"``, or a market or timeframe is
        unknown.
    """
    with path.open("rb") as file:
        raw = tomllib.load(file)
    folds = raw["folds"]["use"]
    if folds not in ("development", "test"):
        raise ValueError(f"folds.use must be 'development' or 'test', got {folds!r}")
    return StudyConfig(
        name=raw["study"]["name"],
        question=raw["study"]["question"],
        tier=raw["data"]["tier"],
        markets=tuple(Market(value) for value in raw["data"]["markets"]),
        timeframes=tuple(Timeframe(value) for value in raw["data"]["timeframes"]),
        folds=folds,
        models=tuple(raw["models"]["names"]),
        seeds=tuple(raw["seeds"]["values"]),
        plan_locked=raw["plan"]["locked"],
        device=raw["compute"]["device"],
        estimated_hours=raw["compute"]["estimated_hours"],
        estimated_cost_usd=raw["compute"]["estimated_cost_usd"],
        options=dict(raw.get("options", {})),
    )
