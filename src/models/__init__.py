"""Zero-shot wrappers of time-series foundation models.

All share one interface: ``predict(context, horizon) -> Forecast``.

The wrappers are imported lazily, so a missing package only breaks its own model.
"""

from typing import Any

from src.models._capabilities import missing_value_reason, takes_covariates
from src.models._forecaster import QUANTILE_LEVELS, Forecast, Forecaster
from src.models._kronos_settings import KronosSettings

__all__ = [
    "QUANTILE_LEVELS",
    "Forecast",
    "Forecaster",
    "KronosSettings",
    "build_forecaster",
    "missing_value_reason",
    "require_revision",
    "takes_covariates",
]

NEW_MODELS = {
    "timesfm3": "TimesFM3Forecaster",
    "tirex": "TiRexForecaster",
    "tirex2": "TiRex2Forecaster",
    "toto2": "Toto2Forecaster",
    "t0": "T0Forecaster",
    "moirai2": "Moirai2Forecaster",
}
"""Kinds served by `src.models._new_models` and the class of each."""


def build_forecaster(kind: str, **kwargs: Any) -> Forecaster:
    """Create a forecaster by kind.

    Kinds: ``"chronos"`` (Chronos-Bolt, Chronos-2), ``"timesfm"`` (2.5), ``"timesfm3"``,
    ``"tirex"``, ``"tirex2"``, ``"toto2"``, ``"t0"``, ``"moirai2"`` and ``"kronos"``.

    Parameters
    ----------
    kind : str
        The model family.
    **kwargs
        Passed to the wrapper (e.g. ``name="amazon/chronos-2"``).

    Returns
    -------
    Forecaster
        The loaded model.
    """
    if kind == "chronos":
        from src.models._chronos import ChronosForecaster

        return ChronosForecaster(**kwargs)
    if kind == "timesfm":
        from src.models._timesfm import TimesFMForecaster

        return TimesFMForecaster(**kwargs)
    if kind in NEW_MODELS:
        from src.models import _new_models

        return getattr(_new_models, NEW_MODELS[kind])(**kwargs)
    if kind == "kronos":
        from src.models._kronos import KronosForecaster

        return KronosForecaster(**kwargs)
    raise ValueError(f"unknown forecaster kind {kind!r}")


def require_revision(repository: str, revision: str) -> None:
    """Check that a Hugging Face repository's ``main`` is the pinned commit `revision`.

    The wrappers load a repository by name, i.e. its ``main`` branch (from the local cache
    when it is there, ``.hf``); checking ``main`` against the commit recorded in the study's
    configuration before loading pins the weights every run uses.

    Parameters
    ----------
    repository : str
        The Hugging Face repository, e.g. ``"amazon/chronos-2"``.
    revision : str
        The pinned commit, 40 hexadecimal characters.

    Raises
    ------
    ValueError
        If `revision` is not a full commit hash, or ``main`` is at another commit.
    """
    from huggingface_hub import HfApi

    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(
            f"the revision of {repository} must be a full commit hash, got {revision!r}"
        )
    main = HfApi().model_info(repository).sha
    if main != revision:
        raise ValueError(f"{repository}: main is at {main}, the configuration pins {revision}")
