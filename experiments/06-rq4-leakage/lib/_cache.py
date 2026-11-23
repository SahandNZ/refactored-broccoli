"""The forecast cache of rq4: the shared test cache, plus a leakage run for l1-l12.

The test cache of 02-forecasts (``options.forecast_cache``) holds the development and test
folds of every model but the leakage quarters l1-l12 of some models only; the other models'
l1-l12 are built into a second 02-forecasts run of the same configuration and data
(``options.leakage_cache``, built with ``run.py --windows``). `leakage_cache` joins the two
into one cache that the scoring reads as if it were one run, and checks that every model the
study scores has forecasts in every window it reads.
"""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.benchmark import ForecastCache, read_cache, scored_folds, study_cache
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS
from src.experiment import StudyConfig, study_runs

from ._leakage import leakage_options

PLAN_FIELDS = ("data_version", "seeds", "kronos_commit", "versions")
"""The fields of a run's ``meta.json`` (data, seeds, Kronos code, package versions) that, with
the ``[folds]`` and ``[options]`` tables of its ``config.toml``, fix how its forecasts were
made: two joined runs must agree on all of them. The git commit and the plan's hash (of the
02-forecasts README) may differ; the forecast code between the two commits is checked by
hand and recorded in the README."""


def _has_forecasts(folder: Path, model: str) -> bool:
    return (folder / "forecasts" / f"{model}.parquet").exists()


@dataclass(frozen=True, slots=True)
class JoinedCache(ForecastCache):
    """Two 02-forecasts runs with the same plan, data and labels, read as one cache.

    The labels, models and assets are those of the first run (``folder``), which
    `join_caches` checked against ``other`` (a forecast row of ``other`` carries that run's
    copy of its label, equal to the first run's within the tolerance of `join_caches`). A
    model's forecasts are its rows of both runs; a window may come from one run only.

    Attributes
    ----------
    other : ForecastCache
        The second run.
    """

    other: ForecastCache

    def forecasts(self, model: str) -> pd.DataFrame:
        """Return the labelled decisions that `model` forecast in either run.

        Parameters
        ----------
        model : str
            A name from ``self.models["model"]``.

        Returns
        -------
        pd.DataFrame
            The rows of `ForecastCache.forecasts` of both runs.

        Raises
        ------
        ValueError
            If `model` is not in the cache, has no forecasts in either run, or has a window
            in both.
        """
        names = self.models["model"].tolist()
        if model not in names:
            raise ValueError(f"model {model!r} is not in {self.folder}, got {names}")
        parts: list[pd.DataFrame] = []
        if _has_forecasts(self.folder, model):
            # the parent's method by name: a slots dataclass has no zero-argument super()
            parts.append(ForecastCache.forecasts(self, model))
        if _has_forecasts(self.other.folder, model):
            parts.append(self.other.forecasts(model))
        if not parts:
            raise ValueError(
                f"{model} has no forecasts in {self.folder} nor in {self.other.folder}"
            )
        if len(parts) == 2:
            shared = sorted(set(parts[0]["window"]) & set(parts[1]["window"]))
            if shared:
                raise ValueError(
                    f"{model}: windows {shared} are in both {self.folder} and {self.other.folder}"
                )
        return pd.concat(parts, ignore_index=True)


def _plan(folder: Path) -> dict[str, object]:
    # how a run made its forecasts: the PLAN_FIELDS of its meta.json and the [folds] and
    # [options] tables of its config.toml (the windows, models, seeds and settings it ran)
    meta = json.loads((folder / "meta.json").read_text())
    config = tomllib.loads((folder / "config.toml").read_text())
    plan: dict[str, object] = {field: meta[field] for field in PLAN_FIELDS}
    return plan | {"folds": config["folds"], "options": config["options"]}


def _differ(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    # whether two tables differ beyond the float rounding of another CPU (relative 1e-9,
    # absolute 1e-12), in keys, columns, their order or values; times and names are compared
    # exactly, and a dtype that only changes the storage does not count
    try:
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            second.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-9,
            atol=1e-12,
        )
    except AssertionError:
        return True
    return False


def join_caches(first: ForecastCache, second: ForecastCache) -> JoinedCache:
    """Join two 02-forecasts runs into one cache (`JoinedCache`).

    Parameters
    ----------
    first, second : ForecastCache
        The runs, e.g. the test cache and the leakage run.

    Returns
    -------
    JoinedCache
        The cache with the labels, models and assets of `first`.

    Raises
    ------
    ValueError
        If the runs differ in their data version, seeds or Kronos commit (`PLAN_FIELDS` of
        ``meta.json``), in the ``[folds]`` or ``[options]`` of their ``config.toml``, or in
        their labels, models or assets (values within a relative 1e-9 and an absolute 1e-12
        count as equal), or if both name the same run. The ``status`` of ``meta.json`` is not
        checked: the test cache's is "failed" because its forecasts were built after its first
        start (02-forecasts README); `require_windows` checks what is there.
    """
    plans = _plan(first.folder), _plan(second.folder)
    if plans[0] != plans[1]:
        differ = [key for key in plans[0] if plans[0][key] != plans[1].get(key)]
        raise ValueError(f"{first.folder} and {second.folder} differ in {differ}")
    for table in ("labels", "models", "assets"):
        if _differ(getattr(first, table), getattr(second, table)):
            raise ValueError(
                f"{first.folder} and {second.folder} have different {table}: a leakage run must "
                "come from the same configuration and data as the test cache"
            )
    return JoinedCache(first.folder, first.labels, first.models, first.assets, other=second)


def require_windows(cache: ForecastCache, models: list[str], windows: list[str]) -> None:
    """Raise when a model of `models` has no forecast at all in a window of `windows`.

    A missing window (a model not built there, a lost shard) would otherwise only shrink a
    period; a single missing asset (a baseline fit that did not converge) is not caught here
    and is listed by 02-forecasts in ``not_fitted.csv``; a window whose forecasts are all
    invalid passes here and is reported by the scoring as without valid decisions.

    Parameters
    ----------
    cache : ForecastCache
        The cache.
    models : list of str
        The models to check.
    windows : list of str
        The windows each model must have.

    Raises
    ------
    ValueError
        With every (model, window) pair without forecasts.
    """
    missing = [
        (model, window)
        for model in models
        for window in sorted(set(windows) - set(cache.forecasts(model)["window"]))
    ]
    if missing:
        raise ValueError(f"no forecasts in {cache.folder} for (model, window) {missing}")


def not_fitted(study_dir: Path, config: StudyConfig) -> pd.DataFrame:
    """Return the baseline fits 02-forecasts left out in the runs rq4 reads (``not_fitted.csv``).

    Parameters
    ----------
    study_dir : Path
        The study folder; the runs are runs of its sibling ``02-forecasts``.
    config : StudyConfig
        The study's configuration (``options.forecast_cache`` and ``options.leakage_cache``).

    Returns
    -------
    pd.DataFrame
        ``model``, ``window``, ``asset``, ``convergence_flag``: one row per fit left out;
        empty when no run lists one.
    """
    runs = study_runs(study_dir.parent / "02-forecasts")
    names = [str(config.options[key]) for key in ("forecast_cache", "leakage_cache")]
    paths = [runs / name / "not_fitted.csv" for name in names if name]
    columns = ["model", "window", "asset", "convergence_flag"]
    parts = [pd.read_csv(path)[columns] for path in paths if path.exists()]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)


def leakage_cache(study_dir: Path, config: StudyConfig) -> ForecastCache:
    """Read the cache rq4 scores: ``forecast_cache``, joined with ``leakage_cache`` if set.

    Every model of the study must have forecasts in every window rq4 reads: the development
    folds and the scored folds, and with the locked plan the leakage quarters, except the
    models of ``options.not_built`` there (`require_windows`).

    Parameters
    ----------
    study_dir : Path
        The study folder; both runs are runs of its sibling ``02-forecasts``.
    config : StudyConfig
        The study's configuration (``options.forecast_cache`` and ``options.leakage_cache``;
        an empty ``leakage_cache`` reads the first run alone).

    Returns
    -------
    ForecastCache
        The cache.

    Raises
    ------
    FileNotFoundError
        If a named run has no ``labels.parquet``.
    ValueError
        If the two runs differ (`join_caches`) or a window is missing (`require_windows`).
    """
    cache = study_cache(study_dir, config)
    leakage_run = str(config.options["leakage_cache"])
    if leakage_run == str(config.options["forecast_cache"]):
        raise ValueError(
            f"leakage_cache must name another run than forecast_cache, got {leakage_run}"
        )
    if leakage_run:
        folder = study_runs(study_dir.parent / "02-forecasts") / leakage_run
        if not (folder / "labels.parquet").exists():
            raise FileNotFoundError(f"no forecast cache in {folder}: check options.leakage_cache")
        cache = join_caches(cache, read_cache(folder))
    folds = [*DEVELOPMENT_FOLDS, *scored_folds(config)]
    require_windows(cache, list(config.models), sorted({fold.name for fold in folds}))
    if config.folds == "test":
        not_built = leakage_options(config).not_built
        built = [name for name in config.models if name not in not_built]
        require_windows(cache, built, [fold.name for fold in LEAKAGE_FOLDS])
    return cache
