"""Study 07-rq5-off-hours: its settings and its two asset sets.

`off_hours_settings` reads the study's options and the benchmark's values it uses (context
length, label scale, model builds and their pinned revisions) once; `universe_assets` reads
which closed-market assets each question uses from the 01-universe run of the benchmark.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark import CacheSettings, forecast_run, scored_folds
from src.enums import Market
from src.evaluate import Fold
from src.experiment import StudyConfig, load_study, study_runs
from src.models import missing_value_reason, takes_covariates
from src.utils import require_columns, to_utc


@dataclass(frozen=True, slots=True)
class OffHoursSettings:
    """The options of 07-rq5-off-hours and the benchmark's values it uses, read once.

    Attributes
    ----------
    targets : tuple of Market
        The closed markets: the study's markets without crypto.
    crypto_basket : tuple of str
        The crypto assets whose mean log return is the crypto move.
    start : pd.Timestamp
        Q2 uses the weekends that reopen from here to the end of the scored folds.
    split_date : pd.Timestamp
        Q1 forecasts the weekends that reopen from here to the end of the scored folds.
    scored_folds : tuple of Fold
        d1-d4 with ``folds = "development"``, t1-t4 with ``"test"``; Q1 scores their weekends.
    min_closed_hours, max_closed_hours : float
        The shortest and the longest weekend closure (`weekend_closures`).
    friday_hours : int
        Friday's information covers this many hours before the close.
    hac_lags : int
        Newey-West lags of the regressions of Q2, in weekends.
    trim_largest : int
        The trimmed rank correlation leaves out this many weekends with the largest absolute
        crypto move.
    min_weekends : int
        A group of Q2 (asset, half-year, direction) or a cell of Q1 with fewer weekends is not
        tested.
    alpha : float
        The significance level of every test (one-sided in Q1, two-sided in Q2).
    interval_level : float
        Coverage of the skill intervals (Q1) and of the coefficient intervals (Q2).
    bootstrap_resamples : int
        Draws of the moving-block bootstrap of the skill intervals.
    bootstrap_seed : int
        Seed of those draws.
    lookback : int
        Calendar hours of every model context (the benchmark's, from the 02-forecasts run).
    span_bars : int
        EWMA span of the label's volatility, in bars (the benchmark's).
    min_fit_weekends : int
        The correction and the model-free arm of a weekend need at least this many earlier
        weekends.
    width_factors : np.ndarray
        The width factors the correction chooses from, shape ``(n_factors,)``.
    zero_shot : dict of str to (str, str)
        Each model to its `src.models.build_forecaster` kind and Hugging Face repository, as
        the 02-forecasts run builds it.
    revisions : dict of str to str
        Each model's pinned commit (`src.models.require_revision`), the benchmark's.
    device : str
        Torch device of the models.
    seed : int
        The one seed of the run (every model of the study is deterministic).
    """

    targets: tuple[Market, ...]
    crypto_basket: tuple[str, ...]
    start: pd.Timestamp
    split_date: pd.Timestamp
    scored_folds: tuple[Fold, ...]
    min_closed_hours: float
    max_closed_hours: float
    friday_hours: int
    hac_lags: int
    trim_largest: int
    min_weekends: int
    alpha: float
    interval_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    lookback: int
    span_bars: int
    min_fit_weekends: int
    width_factors: np.ndarray
    zero_shot: dict[str, tuple[str, str]]
    revisions: dict[str, str]
    device: str
    seed: int

    @property
    def end(self) -> pd.Timestamp:
        """The end of the last scored fold: no weekend reopens at or after it."""
        return self.scored_folds[-1].evaluation_end


def _check_models(config: StudyConfig, benchmark: CacheSettings) -> dict[str, tuple[str, str]]:
    # every model is a zero-shot model of the benchmark that takes past covariates and reads
    # NaN as missing (the calendar and covariate arms show the closed hours as NaN)
    unknown = [model for model in config.models if model not in benchmark.zero_shot]
    if unknown:
        raise ValueError(f"models {unknown} are not zero-shot models of the 02-forecasts run")
    builds = {model: benchmark.zero_shot[model] for model in config.models}
    plain = [model for model, (kind, name) in builds.items() if not takes_covariates(kind, name)]
    if plain:
        raise ValueError(f"models {plain} take no past covariates (src.models.takes_covariates)")
    filling = {model: missing_value_reason(kind) for model, (kind, _) in builds.items()}
    filling = {model: reason for model, reason in filling.items() if reason}
    if filling:
        raise ValueError(
            f"models {sorted(filling)} cannot read NaN as missing, so they have no calendar "
            f"arm: {filling}"
        )
    return builds


def off_hours_settings(config: StudyConfig, benchmark: CacheSettings) -> OffHoursSettings:
    """Read the options of 07-rq5-off-hours from its configuration and the benchmark's.

    Parameters
    ----------
    config : StudyConfig
        The study's configuration.
    benchmark : CacheSettings
        The 02-forecasts run's settings (`benchmark_settings`): the context length, the label's
        EWMA span, how each model is built and its pinned revision.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    ValueError
        If the study has no closed market or more than one seed, or a model is not a zero-shot
        model of the benchmark, takes no past covariates or cannot read NaN as missing.
    """
    options: dict[str, Any] = config.options
    targets = tuple(market for market in config.markets if market is not Market.CRYPTO)
    if not targets:
        raise ValueError(f"07-rq5-off-hours needs a closed market, got {config.markets}")
    if len(config.seeds) != 1:
        raise ValueError(f"the models are deterministic: one seed, got {config.seeds}")
    builds = _check_models(config, benchmark)
    widths: dict[str, Any] = options["width_factors"]
    return OffHoursSettings(
        targets=targets,
        crypto_basket=tuple(str(asset) for asset in options["crypto_basket"]),
        start=to_utc(str(options["start"])),
        split_date=to_utc(str(options["split_date"])),
        scored_folds=scored_folds(config),
        min_closed_hours=float(options["min_closed_hours"]),
        max_closed_hours=float(options["max_closed_hours"]),
        friday_hours=int(options["friday_hours"]),
        hac_lags=int(options["hac_lags"]),
        trim_largest=int(options["trim_largest"]),
        min_weekends=int(options["min_weekends"]),
        alpha=float(options["alpha"]),
        interval_level=float(options["interval_level"]),
        bootstrap_resamples=int(options["bootstrap_resamples"]),
        bootstrap_seed=int(options["bootstrap_seed"]),
        lookback=benchmark.lookback,
        span_bars=benchmark.span_bars,
        min_fit_weekends=int(options["min_fit_weekends"]),
        width_factors=np.linspace(
            float(widths["first"]), float(widths["last"]), int(widths["count"])
        ),
        zero_shot=builds,
        revisions={model: benchmark.revisions[model] for model in builds},
        device=config.device,
        seed=config.seeds[0],
    )


# --- the asset sets -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssetSets:
    """The closed-market assets of each question, from the benchmark's 01-universe run.

    Attributes
    ----------
    forecast : dict of str to Market
        Q1: the closed-market assets of the benchmark's universe (``in_universe``).
    gap : dict of str to Market
        Q2: every closed-market candidate that passed 01-universe's data filter on its
        selection window (2020-2023), in the universe or not.
    universe_run : str
        The 01-universe run they come from.
    """

    forecast: dict[str, Market]
    gap: dict[str, Market]
    universe_run: str


def _universe_table(study_dir: Path, run: str, columns: list[str]) -> pd.DataFrame:
    # the tables/universe.csv of one 01-universe run, with at least `columns`
    folder = study_runs(study_dir.parent / "01-universe") / run
    table = pd.read_csv(folder / "tables" / "universe.csv")
    require_columns(table, columns)
    return table


def _universe(table: pd.DataFrame) -> set[str]:
    # the symbols of a universe table that are in the universe
    return {str(symbol) for symbol in table.loc[table["in_universe"].astype(bool), "symbol"]}


def universe_assets(study_dir: Path, config: StudyConfig, settings: OffHoursSettings) -> AssetSets:
    """Read the asset sets of both questions from the study's 01-universe run.

    The run is the study's own ``options.universe_run``. Its universe (``in_universe``) must be
    the benchmark's: the one of the ``universe_run`` of the 02-forecasts run in
    ``options.forecast_cache``, so the forecasts of Q1 use the paper's closed-market assets,
    as 08-closures does. Q2 takes every closed-market candidate with ``passed_data_filter`` in
    the study's run (01-universe's data filter: no pegged currency, bars from the start and
    a passed quality check): its data passed the checks of the selection window, which ends
    before the weekends Q1 scores, so the set does not require the data to survive to 2026.

    Parameters
    ----------
    study_dir : Path
        The study folder (the runs of 02-forecasts and 01-universe are its siblings').
    config : StudyConfig
        The study's configuration.
    settings : OffHoursSettings
        The study's settings (``targets``).

    Returns
    -------
    AssetSets
        Both sets; the forecast set is inside the gap set.

    Raises
    ------
    ValueError
        If the study's universe run lists another universe than the benchmark's, or has no
        asset of a closed market in the universe.
    """
    run = str(config.options["universe_run"])
    table = _universe_table(
        study_dir, run, ["symbol", "market", "in_universe", "passed_data_filter"]
    )
    benchmark = str(
        load_study(forecast_run(study_dir, config) / "config.toml").options["universe_run"]
    )
    if benchmark != run:
        # an older pinned run may lack later columns: only its universe is compared
        pinned = _universe(_universe_table(study_dir, benchmark, ["symbol", "in_universe"]))
        if _universe(table) != pinned:
            differ = sorted(_universe(table) ^ pinned)
            raise ValueError(
                f"01-universe run {run} has another universe than the benchmark's run "
                f"{benchmark} (assets {differ}): Q1 must forecast the paper's universe"
            )
    names = [market.value for market in settings.targets]
    table = table[table["market"].isin(names)]
    markets = dict(zip(table["symbol"].astype(str), table["market"].map(Market), strict=True))
    forecast = [str(symbol) for symbol in table.loc[table["in_universe"].astype(bool), "symbol"]]
    if not forecast:
        raise ValueError(f"the universe of 01-universe run {run} has no asset of {names}")
    passed = table["passed_data_filter"].astype(bool)
    gap = [str(symbol) for symbol in table.loc[passed, "symbol"]]
    return AssetSets(
        {symbol: markets[symbol] for symbol in forecast},
        {symbol: markets[symbol] for symbol in gap},
        run,
    )
