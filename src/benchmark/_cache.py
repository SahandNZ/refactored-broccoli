from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import StudyConfig, study_runs
from src.utils import require_columns, write_parquet_atomic

QUANTILE_COLUMNS = ("q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90")
"""The nine label quantiles every model writes, for the levels `src.models.QUANTILE_LEVELS`."""

KEYS = ("time", "asset", "horizon")
"""The columns a forecast row and a label row join on."""

FORECAST_COLUMNS = (*KEYS, "seed", *QUANTILE_COLUMNS)
LABEL_COLUMNS = (*KEYS, "window", "label_time", "label", "log_return", "sigma")
MODEL_COLUMNS = ("model", "kind", "release_date", "pretraining_cutoff", "every_hours")
ASSET_COLUMNS = ("asset", "market", "cost_per_side_bps")


@dataclass(frozen=True, slots=True)
class ForecastCache:
    """The forecasts of one 02-forecasts run, read by the benchmark studies (rq1-rq4).

    Attributes
    ----------
    folder : Path
        The run folder, ``runs/<run-id>/`` of 02-forecasts.
    labels : pd.DataFrame
        One row per labelled decision: ``LABEL_COLUMNS``; ``window`` names the fold (``d1``,
        ``t1``, ``l1``, ...) and ``label_time`` is the close time of the label's last bar.
    models : pd.DataFrame
        One row per model: ``MODEL_COLUMNS``; ``kind`` is ``zero-shot`` or ``baseline``, the
        dates are timezone-aware UTC or NaT, and ``every_hours`` is the hours between the
        grid points a model decides at (Kronos: `kronos_every_hours`), 0 for a model that
        forecasts every labelled decision.
    assets : pd.DataFrame
        One row per asset: ``ASSET_COLUMNS``.
    """

    folder: Path
    labels: pd.DataFrame
    models: pd.DataFrame
    assets: pd.DataFrame

    def forecasts(self, model: str) -> pd.DataFrame:
        """Return the labelled decisions that `model` forecast, with its raw quantiles.

        Parameters
        ----------
        model : str
            A name from ``self.models["model"]``.

        Returns
        -------
        pd.DataFrame
            ``LABEL_COLUMNS`` plus ``seed`` and ``QUANTILE_COLUMNS``, one row per decision and
            seed; decisions the model did not forecast are left out.

        Raises
        ------
        ValueError
            If `model` is not in the cache.
        """
        names = self.models["model"].tolist()
        if model not in names:
            raise ValueError(f"model {model!r} is not in {self.folder}, got {names}")
        frame = pd.read_parquet(self.folder / "forecasts" / f"{model}.parquet")
        require_columns(frame, list(FORECAST_COLUMNS))
        return self.labels.merge(frame, on=list(KEYS), how="inner", validate="one_to_many")


def read_cache(folder: Path) -> ForecastCache:
    """Read the labels, models and assets of a 02-forecasts run.

    Parameters
    ----------
    folder : Path
        The run folder, ``experiments/02-forecasts/runs/<run-id>/``.

    Returns
    -------
    ForecastCache
        The tables; forecasts are read per model with `ForecastCache.forecasts`.

    Raises
    ------
    ValueError
        If a table misses a column of the contract or a label is not finite.
    """
    labels = pd.read_parquet(folder / "labels.parquet")
    models = pd.read_csv(folder / "models.csv")
    assets = pd.read_csv(folder / "assets.csv")
    require_columns(labels, list(LABEL_COLUMNS))
    require_columns(models, list(MODEL_COLUMNS))
    require_columns(assets, list(ASSET_COLUMNS))
    is_finite = np.isfinite(labels[["label", "log_return", "sigma"]].to_numpy(dtype=float))
    if not is_finite.all():
        raise ValueError(f"{folder}: labels must be finite, got {(~is_finite).sum()} other values")
    for column in ("release_date", "pretraining_cutoff"):
        models[column] = pd.to_datetime(models[column], utc=True)
    return ForecastCache(folder, labels, models, assets)


def write_forecasts(folder: Path, model: str, frame: pd.DataFrame) -> Path:
    """Write one model's forecasts into a run folder, as the cache contract says.

    Parameters
    ----------
    folder : Path
        The run folder.
    model : str
        The model name; the file is ``forecasts/<model>.parquet``.
    frame : pd.DataFrame
        ``FORECAST_COLUMNS``: ``time`` (UTC), ``asset``, ``horizon`` (hours), ``seed`` and the
        nine quantiles of the vol-scaled label.

    Returns
    -------
    Path
        The written file.
    """
    require_columns(frame, list(FORECAST_COLUMNS))
    table = frame.loc[:, list(FORECAST_COLUMNS)].astype(dict.fromkeys(QUANTILE_COLUMNS, "float32"))
    path = folder / "forecasts" / f"{model}.parquet"
    write_parquet_atomic(table.reset_index(drop=True), path)
    return path


def forecast_run(study_dir: Path, config: StudyConfig) -> Path:
    """Return the 02-forecasts run a study names in ``options.forecast_cache``.

    Parameters
    ----------
    study_dir : Path
        The study folder, ``experiments/<study>``; the run is in its sibling
        ``02-forecasts``.
    config : StudyConfig
        The study's configuration.

    Returns
    -------
    Path
        ``02-forecasts/runs/<forecast_cache>/`` (not checked).
    """
    return study_runs(study_dir.parent / "02-forecasts") / str(config.options["forecast_cache"])


def study_cache(study_dir: Path, config: StudyConfig) -> ForecastCache:
    """Read the forecast cache a scoring study (03-06, 09) names in its options.

    Parameters
    ----------
    study_dir : Path
        The study folder, ``experiments/<study>``; the cache is the run
        ``options.forecast_cache`` of its sibling ``02-forecasts``.
    config : StudyConfig
        The study's configuration.

    Returns
    -------
    ForecastCache
        The cache (`read_cache`).

    Raises
    ------
    FileNotFoundError
        If that run has no ``labels.parquet``.
    """
    folder = forecast_run(study_dir, config)
    if not (folder / "labels.parquet").exists():
        raise FileNotFoundError(f"no forecast cache in {folder}: set options.forecast_cache")
    return read_cache(folder)
