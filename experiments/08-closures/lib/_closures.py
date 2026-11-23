"""Study 08-closures: forecasts after a reopening in every arm, scored per closure day.

`closure_settings` and `benchmark_assets` read the study's options and assets;
`closure_losses` forecasts the decisions of `._closure_contexts` with every model and arm and
scores them, and `closure_day_means` turns the losses into the series of closure days that
the summary, the tests and the intervals (`._verdict`) all use.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.benchmark import (
    QUANTILE_COLUMNS,
    RELEASED_AFTER,
    SCORED,
    CacheSettings,
    benchmark_settings,
    ewma_normal,
    fold_names,
    forecast_run,
    full_quarters,
    is_after_release,
    label_quantiles,
    scored_folds,
    valid_quantiles,
)
from src.enums import Market
from src.evaluate import Fold, quantile_crps
from src.experiment import StudyConfig, seed_everything
from src.models import (
    QUANTILE_LEVELS,
    Forecaster,
    build_forecaster,
    missing_value_reason,
    require_revision,
)
from src.utils import require_columns, to_utc

from ._closure_contexts import (
    ARMS,
    FORWARD_FILLED,
    MISSING,
    REPRESENTATIONS,
    TRADING_HOURS,
    ClosureContexts,
    closure_contexts,
    cut_weekends,
    reopening_decisions,
)

logger = logging.getLogger(__name__)

CLOSED = "closed markets"
CONTROL = "crypto, weekends cut"
"""The two groups of decisions: real closures (Q1) and artificial ones in crypto (Q2)."""

BASELINE = "ewma-normal"
"""The baseline of the study: a standard normal in label units, the same in every arm (it
reads no context)."""

LENGTH_MULTIPLES = {"toto2": 32}
"""Model kinds whose context length must be a multiple of their patch: Toto 2.0 (patch 32,
both sizes' ``config.json``; its wrapper marks every value observed, so a short
trading-hours context is left-padded with its first value, not NaN). Every other kind takes
any length (checked: t0 on CPU; Chronos, TimesFM, TiRex and Moirai on the vast.ai runs)."""

ALL = "all"
"""The pooled split of the summary, the primary one; the others are descriptive."""


@dataclass(frozen=True, slots=True)
class ClosureSettings:
    """The options of 08-closures and the benchmark's values it uses, read once.

    Attributes
    ----------
    models : tuple of str
        Zero-shot models of the 02-forecasts run, and `BASELINE`.
    builds : dict of str to (str, str)
        Each zero-shot model's `src.models.build_forecaster` kind and repository.
    chunks : dict of str to int
        Each zero-shot model's contexts per predict call: the benchmark's ``chunk``, or
        ``moirai_chunk`` for Moirai (a worker process), so a call never needs more GPU
        memory than in 02-forecasts.
    length_multiples : dict of str to int
        Each zero-shot model's required multiple of the context length (`LENGTH_MULTIPLES`,
        else 1).
    revisions : dict of str to str
        Each zero-shot model's pinned commit (`src.models.require_revision`).
    release_dates : dict of str to pd.Timestamp
        Each zero-shot model's scoring start (UTC): the benchmark's, the later of its release
        and the upload of its pinned weights.
    missing_not_applicable : dict of str to str
        Models whose wrapper cannot read NaN as missing (`src.models.missing_value_reason`),
        with the reason; their ``missing`` arm is reported as not applicable.
    lookback : int
        Calendar hours per context, and own bars of the benchmark input (the benchmark's).
    span_bars : int
        EWMA span of the label's volatility (the benchmark's).
    device : str
        Torch device of the zero-shot models.
    seed : int
        The one seed of the run (every model of the study is deterministic).
    scored_folds : tuple of Fold
        d1-d4 with ``folds = "development"``, t1-t4 with ``"test"``.
    min_closure_hours : float
        The shortest closure, in hours without a bar.
    min_own_closes : int
        The fewest own closes a trading-hours context must hold; a decision with fewer (a
        data gap) is dropped from every arm.
    reopen_bars : int
        Target bars per reopening.
    alpha : float
        The one-sided significance level of the tests; the intervals cover ``1 - alpha``.
    min_closure_days : int
        The fewest closure days a model needs to be tested (and to get intervals).
    small_sample_days : int
        Below this many closure days the one-sided test is too liberal with dependent days;
        the results' limitations say so.
    min_quarters : int
        A model with fewer scored quarters wholly after its scoring start is late: it is
        tested in its own Holm family, with its own verdict.
    n_resamples : int
        Bootstrap draws of the intervals.
    bootstrap_seed : int
        Seed of the bootstrap draws.
    """

    models: tuple[str, ...]
    builds: dict[str, tuple[str, str]]
    chunks: dict[str, int]
    length_multiples: dict[str, int]
    revisions: dict[str, str]
    release_dates: dict[str, pd.Timestamp]
    missing_not_applicable: dict[str, str]
    lookback: int
    span_bars: int
    device: str
    seed: int
    scored_folds: tuple[Fold, ...]
    min_closure_hours: float
    min_own_closes: int
    reopen_bars: int
    alpha: float
    min_closure_days: int
    small_sample_days: int
    min_quarters: int
    n_resamples: int
    bootstrap_seed: int

    def post_release_quarters(self, model: str) -> int:
        """Return how many scored quarters start at or after the model's scoring start.

        The quarters wholly after it, as in rq1 (`src.benchmark.full_quarters`).
        """
        return full_quarters(self.release_dates[model], self.scored_folds)


def closure_settings(config: StudyConfig, benchmark: CacheSettings) -> ClosureSettings:
    """Read the options of 08-closures and the benchmark's values it uses.

    Parameters
    ----------
    config : StudyConfig
        The study's configuration.
    benchmark : CacheSettings
        The 02-forecasts run's settings (`benchmark_settings`): the context length, the label,
        how each model is built, its pinned commit and its scoring start.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    ValueError
        If the study has more than one seed, or a model is neither `BASELINE` nor a zero-shot
        model of the benchmark, or is Kronos (its candle contexts in calendar time are not
        coded for this study).
    """
    options: dict[str, Any] = config.options
    if len(config.seeds) != 1:
        raise ValueError(
            f"08-closures runs deterministic models once: one seed, got {config.seeds}"
        )
    zero_shot = [model for model in config.models if model != BASELINE]
    for model in zero_shot:
        if model not in benchmark.zero_shot:
            raise ValueError(
                f"model {model!r} is neither {BASELINE} nor a model of the 02-forecasts run"
            )
        if benchmark.zero_shot[model][0] == "kronos":
            raise ValueError(f"model {model!r}: Kronos is not coded for 08-closures")
    builds = {model: benchmark.zero_shot[model] for model in zero_shot}
    reasons = {model: missing_value_reason(kind) for model, (kind, _) in builds.items()}
    return ClosureSettings(
        models=config.models,
        builds=builds,
        length_multiples={
            model: LENGTH_MULTIPLES.get(kind, 1) for model, (kind, _) in builds.items()
        },
        chunks={
            model: benchmark.moirai_chunk if kind == "moirai2" else benchmark.chunk
            for model, (kind, _) in builds.items()
        },
        revisions={model: benchmark.revisions[model] for model in zero_shot},
        release_dates={model: to_utc(benchmark.release_dates[model]) for model in zero_shot},
        missing_not_applicable={model: reason for model, reason in reasons.items() if reason},
        lookback=benchmark.lookback,
        span_bars=benchmark.span_bars,
        device=config.device,
        seed=config.seeds[0],
        scored_folds=scored_folds(config),
        min_closure_hours=float(options["min_closure_hours"]),
        min_own_closes=int(options["min_own_closes"]),
        reopen_bars=int(options["reopen_bars"]),
        alpha=float(options["alpha"]),
        min_closure_days=int(options["min_closure_days"]),
        small_sample_days=int(options["small_sample_days"]),
        min_quarters=int(options["min_quarters"]),
        n_resamples=int(options["bootstrap_resamples"]),
        bootstrap_seed=int(options["bootstrap_seed"]),
    )


def analysis_settings(
    study_dir: Path, run_config: StudyConfig, summary: pd.DataFrame
) -> ClosureSettings:
    """Read the settings of a run's analysis: the models the run built, from its summary.

    A development run may have run a subset (``run.py --models``), and a model released
    after the scored folds is not built; both leave the analysis, which reads the run's own
    ``config.toml`` and the 02-forecasts run it names.

    Parameters
    ----------
    study_dir : Path
        The study folder.
    run_config : StudyConfig
        The run's ``config.toml``.
    summary : pd.DataFrame
        The run's ``tables/summary.csv``.

    Returns
    -------
    ClosureSettings
        The settings with the built models, in the study's order.
    """
    built = set(summary.loc[summary["status"] != RELEASED_AFTER, "model"])
    models = tuple(model for model in run_config.models if model in built)
    benchmark = benchmark_settings(study_dir, run_config)
    return closure_settings(replace(run_config, models=models), benchmark)


def benchmark_assets(study_dir: Path, config: StudyConfig) -> dict[str, Market]:
    """Return the assets of the 02-forecasts run (its ``assets.csv``) in the study's markets.

    Parameters
    ----------
    study_dir : Path
        The study folder; the run is ``options.forecast_cache`` of its sibling 02-forecasts.
    config : StudyConfig
        The study's configuration.

    Returns
    -------
    dict of str to Market
        Each asset and its market.
    """
    assets = pd.read_csv(forecast_run(study_dir, config) / "assets.csv")
    require_columns(assets, ["asset", "market"])
    markets: dict[str, Market] = {}
    for asset, name in zip(assets["asset"], assets["market"], strict=True):
        if Market(name) in config.markets:
            markets[str(asset)] = Market(name)
    return markets


def closure_types(closed_hours: pd.Series) -> np.ndarray:
    """Name the kind of each closure: night (under 24 hours), weekend (44 or more) or other.

    A forex weekend is 48 hours, 47 at the spring daylight-saving switch of New York and 49 in
    autumn; a one-day US stock holiday 41 (16:00 to the 09:00 decision). So 44 hours separates them.

    Parameters
    ----------
    closed_hours : pd.Series
        Closure lengths in hours, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        ``"night"``, ``"weekend"`` or ``"other"`` (24 to 44 hours, e.g. a one-day holiday),
        shape ``(n,)``.
    """
    hours = closed_hours.to_numpy(dtype=float)
    return np.where(hours < 24, "night", np.where(hours >= 44, "weekend", "other"))


def _group_contexts(
    closes: dict[str, pd.Series], settings: ClosureSettings
) -> ClosureContexts | None:
    # the decisions after every reopening in the scored folds, with their fold as `window`;
    # None for a group without any (e.g. no data in the folds), which is then left out
    parts: list[pd.DataFrame] = []
    for asset, series in closes.items():
        decisions = reopening_decisions(
            series, min_closure_hours=settings.min_closure_hours, reopen_bars=settings.reopen_bars
        )
        windows = fold_names(decisions["time"], settings.scored_folds)
        is_scored: np.ndarray = windows != ""
        parts.append(decisions.assign(asset=asset, window=windows).loc[is_scored])
    found = pd.concat(parts, ignore_index=True)
    if found.empty:
        logger.warning("no reopening in the scored folds for %s", sorted(closes))
        return None
    return closure_contexts(
        closes,
        found,
        lookback=settings.lookback,
        span_bars=settings.span_bars,
        min_own_closes=settings.min_own_closes,
    )


def _left_pad(windows: np.ndarray, multiple: int) -> np.ndarray:
    # contexts of one length, shape (n, length), left-padded with their first value to the
    # next multiple of `multiple` (unchanged when the length already is one)
    pad = -windows.shape[1] % multiple
    return np.concatenate([np.repeat(windows[:, :1], pad, axis=1), windows], axis=1)


def _label_forecasts(
    forecaster: Forecaster | None,
    contexts: ClosureContexts,
    arm: str,
    *,
    chunk: int,
    multiple: int,
) -> np.ndarray:
    # quantiles of the label, shape (n, 9), one step ahead (the next own bar in every arm);
    # the trading-hours contexts are as long as their own bars, so the calls go by length,
    # each in calls of at most `chunk` contexts (the benchmark's, so the GPU memory of a call
    # is that of 02-forecasts), each context left-padded to a multiple of `multiple` (Toto's
    # patch); no forecaster means the baseline, which reads no context
    decisions = contexts.decisions
    if forecaster is None:
        return ewma_normal(decisions)
    windows = contexts.windows[arm]
    lengths = np.full(len(decisions), windows.shape[1])
    if arm == TRADING_HOURS:
        lengths = contexts.own_bars
    last_close, sigma = decisions["last_close"].to_numpy(), decisions["sigma"].to_numpy()
    quantiles = np.empty((len(decisions), len(QUANTILE_COLUMNS)))
    for length in np.unique(lengths):
        same_length = np.flatnonzero(lengths == length)
        for first in range(0, len(same_length), chunk):
            rows = same_length[first : first + chunk]
            context = _left_pad(windows[rows, windows.shape[1] - length :], multiple)
            forecast = forecaster.predict(context, 1)
            prices = forecast.checked_quantiles(len(rows), 1)
            quantiles[rows] = label_quantiles(prices, last_close[rows], sigma[rows], (1,))[:, 0]
    return quantiles


def _scores(quantiles: np.ndarray, labels: np.ndarray, suffix: str) -> dict[str, np.ndarray]:
    # the CRPS, the 10-90 % coverage and the validity of label quantiles, shape (n, 9)
    return {
        f"crps{suffix}": quantile_crps(quantiles, np.asarray(QUANTILE_LEVELS), labels),
        f"is_covered{suffix}": (labels >= quantiles[:, 0]) & (labels <= quantiles[:, -1]),
        f"is_valid{suffix}": valid_quantiles(quantiles),
    }


def _scored(scores: dict[str, dict[str, np.ndarray]], column: str) -> dict[str, np.ndarray]:
    # a decision is scored in the representations only when it is valid in all of them, and
    # in the descriptive benchmark input when it is also valid there
    is_valid = {arm: arm_scores[column] for arm, arm_scores in scores.items()}
    shared = np.logical_and.reduce([is_valid[arm] for arm in is_valid if arm in REPRESENTATIONS])
    return {arm: shared & valid for arm, valid in is_valid.items()}


def _model_losses(
    forecaster: Forecaster | None, model: str, contexts: ClosureContexts, settings: ClosureSettings
) -> pd.DataFrame:
    # every applicable arm forecasts every decision, scored twice: as the model gives its
    # quantiles (primary) and with the nine quantiles sorted (a monotone rearrangement,
    # descriptive robustness: a crossed but finite forecast stays valid)
    labels = contexts.decisions["label"].to_numpy()
    arms = [
        arm for arm in ARMS if not (arm == MISSING and model in settings.missing_not_applicable)
    ]
    chunk = settings.chunks[model] if forecaster is not None else 0
    multiple = settings.length_multiples[model] if forecaster is not None else 1
    scores: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        quantiles = _label_forecasts(forecaster, contexts, arm, chunk=chunk, multiple=multiple)
        scores[arm] = _scores(quantiles, labels, "") | _scores(
            np.sort(quantiles, axis=1), labels, "_sorted"
        )
    is_scored = _scored(scores, "is_valid")
    is_scored_sorted = _scored(scores, "is_valid_sorted")
    closed_share = contexts.is_closed.mean(axis=1)
    parts: list[pd.DataFrame] = []
    for arm in arms:
        has_closed_hours = arm in (FORWARD_FILLED, MISSING)
        parts.append(
            contexts.decisions.assign(
                representation=arm,
                **scores[arm],
                is_scored=is_scored[arm],
                is_scored_sorted=is_scored_sorted[arm],
                closed_share=closed_share if has_closed_hours else 0.0,
            )
        )
    return pd.concat(parts, ignore_index=True).assign(model=model)


def closure_day_means(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Average losses per closure day: over the assets at each time, then over the day's times.

    A closure day is the UTC day on which a closure ends (``closure_day``); every target bar of
    that closure counts on it. Each decision time weighs the same within its day, and each day
    weighs the same in a mean over the days, so the days are the units of the tests and of the
    bootstrap (the decisions of one reopening are not independent).

    Parameters
    ----------
    frame : pd.DataFrame
        Scored losses of one model, group and arm: ``closure_day``, ``time`` and `columns`.
    columns : list of str
        The values to average, e.g. ``["crps"]``.

    Returns
    -------
    pd.DataFrame
        One row per closure day in time order, indexed by ``closure_day``, with `columns`
        and ``times`` (the day's decision times).
    """
    times = frame.groupby(["closure_day", "time"])[columns].mean()
    days = times.groupby(level="closure_day")
    return days.mean().assign(times=days.size())


SUMMARY_COLUMNS = [
    "model",
    "group",
    "representation",
    "split",
    "status",
    "short_contexts",
    "decisions",
    "days",
    "times",
    "invalid",
    "crps",
    "coverage",
    "closed_share",
]
"""The columns of the ``"summary"`` table of `closure_losses`."""


def _split_row(arm_losses: pd.DataFrame) -> dict[str, object]:
    # the scored decisions of one split of an arm, its invalid forecasts, and its CRPS,
    # coverage and share of closed hours, each a mean over closure days (`closure_day_means`)
    scored = arm_losses.loc[arm_losses["is_scored"]]
    row: dict[str, object] = {
        "decisions": len(scored),
        "invalid": int((~arm_losses["is_valid"]).sum()),
    }
    if scored.empty:
        status = "not scored: no valid decision"
        if arm_losses["is_valid"].any():
            status = "not scored: invalid in another representation"
        return row | {"status": status, "days": 0, "times": 0}
    days = closure_day_means(scored, ["crps", "is_covered", "closed_share"])
    return row | {
        "status": SCORED,
        "days": len(days),
        "times": int(days["times"].sum()),
        "crps": days["crps"].mean(),
        "coverage": days["is_covered"].mean(),
        "closed_share": days["closed_share"].mean(),
    }


def _summary(
    losses: pd.DataFrame, model: str, settings: ClosureSettings
) -> list[dict[str, object]]:
    # one row per arm and split: all decisions (primary), each bar after the reopening and
    # each kind of closure (descriptive)
    splits = {ALL: np.ones(len(losses), dtype=bool)}
    for bar in range(settings.reopen_bars):
        splits[f"bar {bar}"] = (losses["bar"] == bar).to_numpy()
    kinds = closure_types(losses["closed_hours"])
    for kind in ("night", "weekend", "other"):
        splits[kind] = kinds == kind
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        row: dict[str, object] = {"model": model, "representation": arm}
        if arm == MISSING and model in settings.missing_not_applicable:
            reason = settings.missing_not_applicable[model]
            rows.append(row | {"split": ALL, "status": f"not applicable: {reason}"})
            continue
        is_arm = (losses["representation"] == arm).to_numpy()
        for split, is_inside in splits.items():
            if split == ALL or (is_arm & is_inside).any():
                arm_losses = losses.loc[is_arm & is_inside]
                rows.append(row | {"split": split} | _split_row(arm_losses))
    return rows


NO_DECISION_AFTER_RELEASE = "not scored: no decision after its scoring start"
"""The status of a built model's arms in a group whose decisions all precede its scoring
start (e.g. the Monday control for a model released late in the last week of the folds)."""


def _without_decisions(
    model: str, group: str, settings: ClosureSettings
) -> list[dict[str, object]]:
    # one pooled row per arm of a group where the model has no decision after its release;
    # a missing arm the model cannot read keeps its reason
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        status = NO_DECISION_AFTER_RELEASE
        if arm == MISSING and model in settings.missing_not_applicable:
            status = f"not applicable: {settings.missing_not_applicable[model]}"
        rows.append({"model": model, "group": group, "representation": arm, "split": ALL,
                     "status": status, "decisions": 0})  # fmt: skip
    return rows


def _released(
    contexts: dict[str, ClosureContexts], model: str, settings: ClosureSettings
) -> dict[str, ClosureContexts]:
    # a zero-shot model forecasts only the decisions after its release; the baseline all
    if model == BASELINE:
        return contexts
    release = settings.release_dates[model]
    return {
        group: found.select(is_after_release(found.decisions, release))
        for group, found in contexts.items()
    }


def _build(model: str, settings: ClosureSettings) -> Forecaster | None:
    # the zero-shot model from its pinned commit; None for the baseline
    seed_everything(settings.seed)
    if model == BASELINE:
        return None
    kind, repository = settings.builds[model]
    require_revision(repository, settings.revisions[model])
    return build_forecaster(kind, name=repository, device=settings.device)


def closure_losses(
    closes: dict[str, pd.Series], markets: dict[str, Market], settings: ClosureSettings
) -> dict[str, pd.DataFrame]:
    """Forecast and score every decision after a reopening, per model and arm.

    The closed markets (every market of `markets` but crypto) give the decisions of Q1; crypto
    with its weekends cut out (`cut_weekends`) those of the control, Q2. A zero-shot model
    forecasts only the decisions after its scoring start (00:00 UTC); a model without any is
    not built and is reported as released after the scored folds. Its ``missing`` arm is
    skipped when its wrapper cannot read NaN (``settings.missing_not_applicable``).

    Parameters
    ----------
    closes : dict of str to pd.Series
        1h closes per asset of `markets`, indexed by close times (UTC).
    markets : dict of str to Market
        The assets and their markets.
    settings : ClosureSettings
        The study's settings.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"losses"``: one row per model, group, arm and decision (``asset``, ``market``,
        ``window``, ``time``, ``bar``, ``closed_hours``, ``closure_day``, ``label`` ...,
        ``crps``, ``is_covered``, ``is_valid``, ``is_scored``, ``closed_share`` (of the
        context's hours) and the same four scores with the quantiles sorted, ``crps_sorted``,
        ``is_covered_sorted``, ``is_valid_sorted``, ``is_scored_sorted``). ``"summary"``: one
        row per model, group, arm and split (`SUMMARY_COLUMNS`); ``short_contexts`` is the
        group's count of decisions dropped for fewer than ``min_own_closes`` own closes in the
        trading-hours context.

    Raises
    ------
    ValueError
        If no model has a decision after its release (nothing to score).
    """
    is_crypto = {asset: market is Market.CRYPTO for asset, market in markets.items()}
    groups = {
        CLOSED: {asset: closes[asset] for asset in markets if not is_crypto[asset]},
        CONTROL: {asset: cut_weekends(closes[asset]) for asset in markets if is_crypto[asset]},
    }
    contexts: dict[str, ClosureContexts] = {}
    for group, series in groups.items():
        found = _group_contexts(series, settings) if series else None
        if found is not None:
            contexts[group] = found
    losses: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for model in settings.models:
        released = _released(contexts, model, settings)
        if all(found.decisions.empty for found in released.values()):
            logger.warning("%s: no decision after its release, not built", model)
            not_built: list[dict[str, object]] = [
                {"model": model, "group": group, "representation": arm, "split": ALL,
                 "status": RELEASED_AFTER}
                for group in contexts
                for arm in ARMS
            ]  # fmt: skip
            summary += not_built
            continue
        forecaster = _build(model, settings)
        for group, selected in released.items():
            if selected.decisions.empty:
                summary += _without_decisions(model, group, settings)
                continue
            frame = _model_losses(forecaster, model, selected, settings).assign(group=group)
            losses.append(frame)
            summary += [row | {"group": group} for row in _summary(frame, model, settings)]
            logger.info("%s, %s: %d decisions", model, group, len(selected.decisions))
        del forecaster
        torch.cuda.empty_cache()
    if not losses:
        raise ValueError("no model has a decision after its release in the scored folds")
    table = pd.concat(losses, ignore_index=True)
    table["market"] = table["asset"].map({asset: market.value for asset, market in markets.items()})
    frame = pd.DataFrame(summary, columns=SUMMARY_COLUMNS)
    short = {group: found.short_contexts for group, found in contexts.items()}
    frame["short_contexts"] = frame["group"].map(short)
    return {"losses": table, "summary": frame}
