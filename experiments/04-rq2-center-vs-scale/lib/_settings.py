"""The decision rule of rq2, read from the study's configuration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.experiment import StudyConfig


@dataclass(frozen=True, slots=True)
class CentreScaleRule:
    """The decision rule of rq2 (04-rq2-center-vs-scale).

    Attributes
    ----------
    scale_reference : str
        The baseline whose scale every model is compared with (Q2).
    centre_reference : str
        The baseline whose sign hit rate, on the same decisions, is the chance level of a
        model's hit rate (Q1).
    min_quarters : int
        The fewest scored quarters a rule must hold in.
    main_family_min_quarters : int
        A model is in the main Holm family when at least this many scored quarters start at
        or after its scoring start; the others form the late family, with their own Holm
        correction.
    coverage_tolerance : float
        The largest coverage error of a calibrated quarter.
    figure_horizon : int
        The horizon in bars of the figure and its table (``rq2``).
    """

    scale_reference: str
    centre_reference: str
    min_quarters: int
    main_family_min_quarters: int
    coverage_tolerance: float
    figure_horizon: int


def centre_scale_rule(config: "StudyConfig") -> CentreScaleRule:
    """Read the rule of rq2 from the configuration of 04-rq2-center-vs-scale.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    options: dict[str, Any] = config.options
    return CentreScaleRule(
        scale_reference=str(options["scale_reference"]),
        centre_reference=str(options["centre_reference"]),
        min_quarters=int(options["min_quarters"]),
        main_family_min_quarters=int(options["main_family_min_quarters"]),
        coverage_tolerance=float(options["coverage_tolerance"]),
        figure_horizon=int(options["figure_horizon"]),
    )
