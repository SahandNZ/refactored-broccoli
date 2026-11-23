import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark._cache import forecast_run
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS, Fold
from src.experiment import StudyConfig, load_study
from src.utils import hour_grid

BASELINES = ("random-walk", "climatology", "ewma-normal", "garch-normal", "linear")
"""The baselines of the forecast cache, fitted per window (``src.benchmark._baselines``)."""


@dataclass(frozen=True, slots=True)
class CacheSettings:
    """The configuration of a 02-forecasts run, read once from its ``study.toml``.

    Attributes
    ----------
    models : tuple of str
        Every model of the run: names from `BASELINES` or from `zero_shot`.
    seeds : tuple of int
        Seeds of the sampling model (Kronos), whose paths are pooled over all seeds; every
        model writes one forecast per decision, with seed 0.
    device : str
        Torch device of the zero-shot models.
    folds : str
        ``"development"`` (windows d1-d4) or ``"test"`` (every window: d1-d4, t1-t4, l1-l12).
    lookback : int
        Own 1h closes per context.
    horizons : tuple of int
        Label horizons in the asset's own 1h bars; one model call forecasts ``max(horizons)``
        steps.
    span_bars : int
        EWMA span of the label's volatility, in bars.
    train_days : int
        Days before each window start that fit the baselines.
    step_hours : dict of str to int
        Hours between decisions per fold kind (``development``, ``test``, ``leakage``).
    kronos_every_hours : int
        Hours between Kronos decisions, counted from each window start at 00:00 UTC (167: one
        week less one hour, so a quarter's ~13 grid points cover about 14 hours of the day,
        the same hours in every window).
    kronos_paths : int
        Sampled Kronos paths per context.
    kronos_tokenizer : str
        The Hugging Face name of Kronos's tokenizer.
    kronos_tokenizer_revision : str
        The commit of the tokenizer every run loads (`src.models.require_revision`).
    kronos_temperature, kronos_top_p : float
        Kronos's sampling temperature and nucleus threshold.
    chunk, moirai_chunk : int
        Contexts per model call (Moirai runs in a worker process, so it takes larger calls).
    linear_lags : tuple of int
        Lags of the linear baseline, in own bars.
    linear_ridge : float
        L2 penalty of the linear baseline.
    crypto_fee : str
        Binance USD-M fee of crypto, ``"maker"`` or ``"taker"``.
    us_stock_half_spread_bps : float
        Assumed half spread of US stocks without bid and ask (Alpaca).
    spread_window : tuple of (pd.Timestamp, pd.Timestamp)
        The UTC window ``[start, end)`` whose median spread sets each asset's cost: the
        universe's selection window, before every fold.
    zero_shot : dict of str to (str, str)
        Model name to its `src.models.build_forecaster` kind and Hugging Face repository.
    revisions : dict of str to str
        Zero-shot model name to the commit of its repository every run loads.
    release_dates : dict of str to str
        The scoring start (UTC date) per zero-shot model: the later of its release date
        (``options.release_dates``: the later of repository creation and first weight upload,
        owner decision A2) and the upload date of its pinned weights
        (``options.weight_dates``), so no model is scored on data before the weights it loads.
    pretraining_cutoffs : dict of str to str
        UTC dates; a cutoff only where the model card or paper states one.
    q1_min_share : float
        Q1's least share of labelled decisions with a valid forecast, per model.
    """

    models: tuple[str, ...]
    seeds: tuple[int, ...]
    device: str
    folds: str
    lookback: int
    horizons: tuple[int, ...]
    span_bars: int
    train_days: int
    step_hours: dict[str, int]
    kronos_every_hours: int
    kronos_paths: int
    kronos_tokenizer: str
    kronos_tokenizer_revision: str
    kronos_temperature: float
    kronos_top_p: float
    chunk: int
    moirai_chunk: int
    linear_lags: tuple[int, ...]
    linear_ridge: float
    crypto_fee: str
    us_stock_half_spread_bps: float
    spread_window: tuple[pd.Timestamp, pd.Timestamp]
    zero_shot: dict[str, tuple[str, str]]
    revisions: dict[str, str]
    release_dates: dict[str, str]
    pretraining_cutoffs: dict[str, str]
    q1_min_share: float


def cache_settings(config: StudyConfig) -> CacheSettings:
    """Read the settings of a 02-forecasts run from its configuration.

    Parameters
    ----------
    config : StudyConfig
        The parsed ``study.toml``.

    Returns
    -------
    CacheSettings
        The checked settings.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults), or a zero-shot model has no release
        date, weight date or pinned revision.
    ValueError
        If a model is neither a baseline nor in ``options.zero_shot``.
    """
    # the [options] table is untyped TOML; every value is converted to its type here
    options: dict[str, Any] = config.options
    zero_shot = {
        str(name): (str(kind), str(repo)) for name, (kind, repo) in options["zero_shot"].items()
    }
    unknown = [name for name in config.models if name not in BASELINES and name not in zero_shot]
    if unknown:
        raise ValueError(f"models {unknown} are neither in {BASELINES} nor in options.zero_shot")
    return CacheSettings(
        models=config.models,
        seeds=config.seeds,
        device=config.device,
        folds=config.folds,
        lookback=int(options["lookback"]),
        horizons=tuple(int(horizon) for horizon in options["horizons"]),
        span_bars=int(options["span_bars"]),
        train_days=int(options["train_days"]),
        step_hours={
            kind: int(options["step_hours"][kind]) for kind in ("development", "test", "leakage")
        },
        kronos_every_hours=int(options["kronos_every_hours"]),
        kronos_paths=int(options["kronos_paths"]),
        kronos_tokenizer=str(options["kronos_tokenizer"]),
        kronos_tokenizer_revision=str(options["kronos_tokenizer_revision"]),
        kronos_temperature=float(options["kronos_temperature"]),
        kronos_top_p=float(options["kronos_top_p"]),
        chunk=int(options["chunk"]),
        moirai_chunk=int(options["moirai_chunk"]),
        linear_lags=tuple(int(lag) for lag in options["linear_lags"]),
        linear_ridge=float(options["linear_ridge"]),
        crypto_fee=str(options["crypto_fee"]),
        us_stock_half_spread_bps=float(options["us_stock_half_spread_bps"]),
        spread_window=(
            pd.Timestamp(str(options["spread_window"][0]), tz="UTC"),
            pd.Timestamp(str(options["spread_window"][1]), tz="UTC"),
        ),
        zero_shot=zero_shot,
        revisions={
            name: str(options["revisions"][name]) for name in config.models if name in zero_shot
        },
        release_dates={
            name: max(str(options["release_dates"][name]), str(options["weight_dates"][name]))
            for name in config.models
            if name in zero_shot
        },
        pretraining_cutoffs={
            str(name): str(date) for name, date in options["pretraining_cutoffs"].items()
        },
        q1_min_share=float(options["q1_min_share"]),
    )


def benchmark_settings(study_dir: Path, config: StudyConfig) -> CacheSettings:
    """Read the settings of the 02-forecasts run a study names in ``options.forecast_cache``.

    The studies that make their own forecasts (07-rq5-off-hours, 08-closures) take the
    benchmark's decided values (context length, labels, model builds, release dates) from the
    run's ``config.toml``, kept in the run, so they are written in one place.

    Parameters
    ----------
    study_dir : Path
        The study folder.
    config : StudyConfig
        The study's configuration.

    Returns
    -------
    CacheSettings
        The run's settings.

    Raises
    ------
    FileNotFoundError
        If that run has no ``config.toml``.
    """
    folder = forecast_run(study_dir, config)
    if not (folder / "config.toml").exists():
        raise FileNotFoundError(f"no 02-forecasts run in {folder}: set options.forecast_cache")
    return cache_settings(load_study(folder / "config.toml"))


def cache_windows(settings: CacheSettings) -> list[tuple[Fold, int]]:
    """Return the windows of a run and the hours between their decisions.

    Parameters
    ----------
    settings : CacheSettings
        The run's settings.

    Returns
    -------
    list of (Fold, int)
        Development folds d1-d4; with ``folds = "test"`` also t1-t4 and l1-l12.
    """
    groups = {"development": DEVELOPMENT_FOLDS}
    if settings.folds == "test":
        groups |= {"test": TEST_FOLDS, "leakage": LEAKAGE_FOLDS}
    return [(fold, settings.step_hours[kind]) for kind, folds in groups.items() for fold in folds]


def model_kind(model: str, settings: CacheSettings) -> str:
    """Return ``"baseline"`` or the `src.models.build_forecaster` kind of a model."""
    return "baseline" if model in BASELINES else settings.zero_shot[model][0]


def on_model_grid(labels: pd.DataFrame, model: str, settings: CacheSettings) -> pd.DataFrame:
    """Keep the labelled rows a model forecasts: all; for Kronos one decision per grid point.

    Kronos (sampled, slow) has a grid every `kronos_every_hours` from each window start; at
    each grid point every asset takes its first labelled decision at or after it, before the
    next grid point. So Kronos's decisions do not depend on a window's step and a market that
    closes still gets a decision at almost every grid point.

    Parameters
    ----------
    labels : pd.DataFrame
        Label rows with ``time``, ``asset`` and ``window``.
    model : str
        The model.
    settings : CacheSettings
        The run's settings.

    Returns
    -------
    pd.DataFrame
        The rows on the model's decision grid, with a new index.
    """
    if model_kind(model, settings) != "kronos":
        return labels.reset_index(drop=True)
    every = pd.Timedelta(hours=settings.kronos_every_hours)
    folds = {fold.name: fold for fold, _ in cache_windows(settings)}
    chosen: list[pd.DataFrame] = []
    for (window, _), rows in labels.groupby(["window", "asset"]):
        fold = folds[str(window)]
        grid = hour_grid(
            fold.evaluation_start, fold.evaluation_end, step_hours=settings.kronos_every_hours
        )
        times = pd.DatetimeIndex(rows["time"].drop_duplicates().sort_values())
        first = times.searchsorted(grid)
        has_one = first < len(times)
        picked = times[first[has_one]]
        picked = picked[picked < grid[has_one] + every]
        chosen.append(rows[rows["time"].isin(picked)])
    if not chosen:
        return labels.iloc[:0].reset_index(drop=True)
    return pd.concat(chosen).sort_values(["time", "asset"], ignore_index=True)


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """The moving-block bootstrap of a scoring study (03-06, 09).

    Attributes
    ----------
    level : float
        Coverage of an interval, ``1 - alpha``.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    block_hours : float
        The block covers about this many hours of decisions.
    min_block : int
        The shortest block, in decision times.
    """

    level: float
    n_resamples: int
    seed: int
    block_hours: float
    min_block: int

    def block_length(self, spacing_hours: float) -> int:
        """Return the block in decision times of a series spaced `spacing_hours` apart.

        ``max(ceil(block_hours / spacing_hours), min_block)``: 24 for hourly decisions, 6 for
        the leakage quarters (every 4 hours) and `min_block` for Kronos (every 167 hours), with
        ``block_hours = 24`` and ``min_block = 4``.
        """
        return max(math.ceil(self.block_hours / spacing_hours), self.min_block)


@dataclass(frozen=True, slots=True)
class StudySettings:
    """The options every scoring study (03-06, 09) shares, read once from its ``study.toml``.

    Attributes
    ----------
    choice_folds : tuple of Fold
        The folds the best baseline is chosen on: always the development folds d1-d4.
    scored_folds : tuple of Fold
        The scored folds: d1-d4 with ``folds = "development"``, t1-t4 with ``"test"``.
    alpha : float
        The significance level of every test.
    bootstrap : Bootstrap
        The circular-block bootstrap, ``level = 1 - alpha``.
    """

    choice_folds: tuple[Fold, ...]
    scored_folds: tuple[Fold, ...]
    alpha: float
    bootstrap: Bootstrap


def study_settings(config: StudyConfig) -> StudySettings:
    """Read the shared options of a scoring study from its configuration.

    Parameters
    ----------
    config : StudyConfig
        The parsed ``study.toml`` (``alpha``, ``bootstrap_resamples``, ``bootstrap_seed``,
        ``bootstrap_block_hours`` and ``bootstrap_min_block`` in ``[options]``).

    Returns
    -------
    StudySettings
        The settings.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    alpha = float(str(config.options["alpha"]))
    return StudySettings(DEVELOPMENT_FOLDS, scored_folds(config), alpha, read_bootstrap(config))


def read_bootstrap(config: StudyConfig) -> Bootstrap:
    """Read the moving-block bootstrap of a study (``alpha`` and the ``bootstrap_*`` options).

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    options: dict[str, Any] = config.options
    return Bootstrap(
        level=1 - float(options["alpha"]),
        n_resamples=int(options["bootstrap_resamples"]),
        seed=int(options["bootstrap_seed"]),
        block_hours=float(options["bootstrap_block_hours"]),
        min_block=int(options["bootstrap_min_block"]),
    )


def scored_folds(config: StudyConfig) -> tuple[Fold, ...]:
    """Return the scored folds of a study: d1-d4 while the plan is open, t1-t4 when locked."""
    return TEST_FOLDS if config.folds == "test" else DEVELOPMENT_FOLDS


def fold_names(times: pd.Series, folds: tuple[Fold, ...]) -> np.ndarray:
    """Return the name of the fold each time falls in, ``""`` outside every fold.

    Parameters
    ----------
    times : pd.Series
        UTC times, shape ``(n,)``.
    folds : tuple of Fold
        Folds whose evaluation quarters do not overlap.

    Returns
    -------
    np.ndarray
        Fold names (object array), shape ``(n,)``.
    """
    names = np.full(len(times), "", dtype=object)
    for fold in folds:
        is_inside = (times >= fold.evaluation_start) & (times < fold.evaluation_end)
        names[is_inside.to_numpy()] = fold.name
    return names


@dataclass(frozen=True, slots=True)
class NullRule:
    """The decision rule of rq1 and of its power (03-rq1-null; 09-mde-threshold reads it too).

    Attributes
    ----------
    min_winning_quarters : int
        The fewest scored quarters a model must win.
    main_family_min_quarters : int
        A model is in the main Holm family when at least this many scored quarters start
        after its scoring start; the others form the late family, with their own Holm
        correction and verdict.
    skill_resamples : int
        Moving-block bootstrap draws of the skill interval of a cell.
    improvements_pct : tuple of float
        The planted CRPS gains of the power analysis, in % of the best baseline's CRPS.
    power_target : float
        The detection rate that defines the minimum detectable effect (MDE).
    mde_horizon : int
        The horizon (in own bars) whose cells the power question judges.
    threshold_grid_pct : tuple of float
        The MDE thresholds X swept for the share of resolved cells, increasing.
    power_share : float
        The power rule: the Q1 rule is powerful when at least this share of the cells at
        `mde_horizon` has an MDE at or below the threshold (09 chooses the threshold as the
        smallest X of the grid that reaches this share on the development folds).
    power_share_sensitivity : tuple of float
        Other shares the power answer reports, for sensitivity only.
    mde_threshold_pct : float
        The threshold the power question judges, set from 09-mde-threshold before the test
        run.
    """

    min_winning_quarters: int
    main_family_min_quarters: int
    skill_resamples: int
    improvements_pct: tuple[float, ...]
    power_target: float
    mde_horizon: int
    threshold_grid_pct: tuple[float, ...]
    power_share: float
    power_share_sensitivity: tuple[float, ...]
    mde_threshold_pct: float


def null_rule(config: StudyConfig) -> NullRule:
    """Read the rule of rq1 and its power from the configuration of 03-rq1-null.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    options: dict[str, Any] = config.options
    return NullRule(
        min_winning_quarters=int(options["min_winning_quarters"]),
        main_family_min_quarters=int(options["main_family_min_quarters"]),
        skill_resamples=int(options["skill_resamples"]),
        improvements_pct=tuple(float(gain) for gain in options["power_improvements_pct"]),
        power_target=float(options["power_target"]),
        mde_horizon=int(options["mde_horizon"]),
        threshold_grid_pct=tuple(float(value) for value in options["mde_threshold_grid_pct"]),
        power_share=float(options["power_share"]),
        power_share_sensitivity=tuple(float(share) for share in options["power_share_sensitivity"]),
        mde_threshold_pct=float(options["mde_threshold_pct"]),
    )
