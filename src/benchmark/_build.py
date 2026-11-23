import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch

from src.benchmark._baselines import climatology, ewma_normal, garch_normal, linear, random_walk
from src.benchmark._cache import KEYS, LABEL_COLUMNS, QUANTILE_COLUMNS, write_forecasts
from src.benchmark._contexts import context_ends
from src.benchmark._settings import (
    BASELINES,
    CacheSettings,
    cache_settings,
    cache_windows,
    model_kind,
    on_model_grid,
)
from src.enums import Market, Timeframe
from src.evaluate import cost_per_side_bps
from src.experiment import StudyConfig, seed_everything
from src.models import (
    QUANTILE_LEVELS,
    Forecast,
    Forecaster,
    KronosSettings,
    build_forecaster,
    require_revision,
)
from src.panel import ewma_volatility, hourly_closes, is_us_stock, load_panel_bars
from src.utils import hour_grid, last_at_or_before, write_parquet_atomic

if TYPE_CHECKING:
    from src.models._kronos import KronosForecaster

logger = logging.getLogger(__name__)

type Bars = dict[str, dict[Timeframe, pd.DataFrame]]
type Closes = dict[str, pd.Series]


@dataclass(frozen=True, slots=True)
class CachePart:
    """The part of a cache one `build_cache` call builds, when not the whole cache.

    Attributes
    ----------
    last_days : int or None
        A smoke run on development folds only: the labels keep only the decisions of the
        last `last_days` days of each fold (applied when the labels are first written).
        None keeps whole folds.
    seeds : tuple of int or None
        A shard of a sampling model (Kronos): only these seeds of the configuration are
        sampled, into ``samples/<model>/seed<s>.npz`` (each seed's samples are the same
        whether the seeds run together or apart); the forecasts are written once every seed
        is sampled. None samples every seed.
    windows : tuple of str or None
        Only these windows (fold names) are forecast; the labels of every window are written
        all the same. None forecasts every window.
    """

    last_days: int | None
    seeds: tuple[int, ...] | None
    windows: tuple[str, ...] | None = None


def build_cache(
    folder: Path,
    config: StudyConfig,
    *,
    data_dir: Path,
    universe: Path,
    models: tuple[str, ...],
    part: CachePart | None = None,
) -> pd.DataFrame:
    """Build the forecast cache of 02-forecasts into a run folder; resumable model by model.

    The first call writes ``labels.parquet``, ``models.csv`` and ``assets.csv``; every call then
    writes ``forecasts/<model>.parquet`` for each of `models` whose file does not exist yet.

    Parameters
    ----------
    folder : Path
        The run folder, ``runs/<run-id>/``.
    config : StudyConfig
        The run's configuration (options as in its ``config.toml``).
    data_dir : Path
        The lab's ``data/`` folder (read only).
    universe : Path
        ``tables/universe.csv`` of the 01-universe run; its assets with ``in_universe``
        are the assets of the cache.
    models : tuple of str
        The models to build, a subset of ``config.models``.
    part : CachePart or None, optional
        A part of the cache only (`CachePart`: a smoke run's last days, a shard of seeds);
        None builds the whole cache.

    Returns
    -------
    pd.DataFrame
        One row per model built by this call (``rows`` 0 for a sampling model whose other
        seeds are not sampled yet): ``model``, ``rows``, ``seconds``,
        ``peak_gpu_gb`` (memory of this process; Moirai runs in a worker process) and
        ``last_days`` (0 for whole folds).

    Raises
    ------
    ValueError
        If a model is not in the configuration, the universe has no asset, `last_days` is
        given with test folds, `seeds` is not a subset of the configuration's seeds, or a
        window of `part` is not a window of the run.
    """
    settings = cache_settings(config)
    part = part or CachePart(last_days=None, seeds=None, windows=None)
    seeds = _checked_seeds(part, settings)
    if part.windows is not None and part.seeds is not None:
        # seeds sampled apart pool only when every seed holds the run's windows; a shard of
        # seeds keeps every window, so its seed files can be pooled with the others (kept as
        # a rule, owner's decision of 2026-11-17, though the arrays are named by window)
        raise ValueError("--windows cannot be combined with --seeds (a sampling model's shards)")
    unknown = [model for model in models if model not in settings.models]
    if unknown:
        raise ValueError(f"models {unknown} are not in study.toml models {settings.models}")
    markets = _universe_markets(universe)
    bars = load_panel_bars(data_dir, markets, (Timeframe.H1,))
    closes = hourly_closes(bars)
    if not (folder / "labels.parquet").exists():
        _write_tables(folder, settings, markets, bars, closes, last_days=part.last_days)
    labels = _part_labels(pd.read_parquet(folder / "labels.parquet"), part)
    forecasts = folder / "forecasts"
    missing = [model for model in models if not (forecasts / f"{model}.parquet").exists()]
    logger.info("models already written, skipped: %s", sorted(set(models) - set(missing)))
    training: dict[str, pd.DataFrame] = {}
    if any(model in BASELINES for model in missing):
        for fold, _ in cache_windows(settings):
            training[fold.name] = training_labels(closes, fold.evaluation_start, settings)
    built: list[dict[str, object]] = []
    for model in missing:
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        not_fitted = pd.DataFrame()
        if model in BASELINES:
            frame, not_fitted = _baseline_forecasts(model, labels, training, closes, settings)
        else:
            panel = _Panel(bars, closes)
            samples = folder / "samples" / model
            frame = _zero_shot_forecasts(
                model, labels, panel, settings, samples=samples, seeds=seeds
            )
        if frame is not None:
            # listed before the forecasts are written: a crash in between leaves a listed row
            # whose model is built again on resume and listed twice (readers drop duplicates),
            # never a written model without its list
            _list_not_fitted(folder, not_fitted)
            _list_windows(folder, model, [str(name) for name in labels["window"].unique()])
            write_forecasts(folder, model, frame)
        memory = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        built.append(
            {
                "model": model,
                "rows": 0 if frame is None else len(frame),
                "seconds": time.monotonic() - started,
                "peak_gpu_gb": memory,
                "last_days": part.last_days or 0,
            }
        )
        logger.info("%s", built[-1])
    return pd.DataFrame(built, columns=["model", "rows", "seconds", "peak_gpu_gb", "last_days"])


def _part_labels(labels: pd.DataFrame, part: CachePart) -> pd.DataFrame:
    # the labels the models of this call forecast: every window, or only `part.windows`
    if part.windows is None:
        return labels
    unknown = sorted(set(part.windows) - set(labels["window"]))
    if unknown:
        raise ValueError(
            f"windows {unknown} are not windows of the run, got {sorted(set(labels['window']))}"
        )
    return labels[labels["window"].isin(part.windows)].reset_index(drop=True)


def _checked_seeds(part: CachePart, settings: CacheSettings) -> tuple[int, ...]:
    # the seeds this call samples; a smoke run is for development folds only
    if part.last_days is not None and settings.folds != "development":
        raise ValueError(f"last_days is for development folds only, got {settings.folds}")
    seeds = part.seeds or settings.seeds
    if not set(seeds) <= set(settings.seeds):
        raise ValueError(f"seeds {list(seeds)} are not a subset of {list(settings.seeds)}")
    return seeds


@dataclass(frozen=True, slots=True)
class _Panel:
    # the 1h bars (Kronos reads candles) and closes (every other model) of every asset
    bars: Bars
    closes: Closes


def _universe_markets(path: Path) -> dict[str, Market]:
    universe = pd.read_csv(path)
    chosen = universe[universe["in_universe"]]
    if chosen.empty:
        raise ValueError(f"{path} has no asset in the universe")
    return {
        str(symbol): Market(market)
        for symbol, market in zip(chosen["symbol"], chosen["market"], strict=True)
    }


def _write_tables(
    folder: Path,
    settings: CacheSettings,
    markets: dict[str, Market],
    bars: Bars,
    closes: Closes,
    *,
    last_days: int | None,
) -> None:
    # labels.parquet is written last: its presence means the tables of the run are complete.
    # A label must end inside its window, so no decision or label reaches into another window;
    # a smoke run (`last_days`) keeps only the decisions of the last days of each window
    _model_table(settings).to_csv(folder / "models.csv", index=False)
    _asset_table(markets, bars, settings).to_csv(folder / "assets.csv", index=False)
    parts: list[pd.DataFrame] = []
    for fold, step in cache_windows(settings):
        times = hour_grid(fold.evaluation_start, fold.evaluation_end, step_hours=step)
        labels = decision_labels(closes, times, settings, step_hours=step)
        inside = labels[labels["label_time"] <= fold.evaluation_end]
        if last_days is not None:
            first = fold.evaluation_end - pd.Timedelta(days=last_days)
            inside = inside[inside["time"] >= first]
        parts.append(inside.assign(window=fold.name))
    labels = pd.concat(parts, ignore_index=True)
    repeated = labels.duplicated(list(KEYS))
    if repeated.any():
        raise ValueError(f"{int(repeated.sum())} decisions are in two windows; the folds overlap")
    write_parquet_atomic(labels[list(LABEL_COLUMNS)], folder / "labels.parquet")


def _model_table(settings: CacheSettings) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in settings.models:
        is_baseline = model in BASELINES
        rows.append(
            {
                "model": model,
                "kind": "baseline" if is_baseline else "zero-shot",
                "release_date": "" if is_baseline else settings.release_dates[model],
                "pretraining_cutoff": settings.pretraining_cutoffs.get(model, ""),
                "every_hours": settings.kronos_every_hours
                if model_kind(model, settings) == "kronos"
                else 0,
            }
        )
    return pd.DataFrame(rows)


def median_spread_bps(bars: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Return the median bid-ask spread of 1h bars opening in ``[start, end)``, in bps of the mid.

    The window is the universe's selection window (``spread_window``), before every fold, so
    no cost is fitted on development or test data.

    Parameters
    ----------
    bars : pd.DataFrame
        1h bars indexed by UTC open times; a bid/ask source has ``spread`` and ``mid_close``.
    start, end : pd.Timestamp
        The window (UTC).

    Returns
    -------
    float
        The median of ``spread / mid_close * 1e4``; NaN without a ``spread`` column or bars in
        the window.
    """
    if "spread" not in bars.columns:
        return float("nan")
    inside = bars[(bars.index >= start) & (bars.index < end)]
    return float((inside["spread"] / inside["mid_close"]).median() * 1e4)


def _asset_table(markets: dict[str, Market], bars: Bars, settings: CacheSettings) -> pd.DataFrame:
    # crypto pays the Binance fee, every other asset half its median Dukascopy spread in the
    # spread window; US stocks from Alpaca have no bid and ask, so they get the assumed half
    # spread of study.toml
    start, end = settings.spread_window
    rows: list[dict[str, object]] = []
    for asset, market in markets.items():
        spread = median_spread_bps(bars[asset][Timeframe.H1], start=start, end=end)
        half_spread = spread / 2
        if is_us_stock(asset, market) and not np.isfinite(half_spread):
            half_spread = settings.us_stock_half_spread_bps
        cost = cost_per_side_bps(
            market, half_spread_bps=half_spread, crypto_fee=settings.crypto_fee
        )
        rows.append({"asset": asset, "market": market.value, "cost_per_side_bps": cost})
    return pd.DataFrame(rows)


def decision_labels(
    closes: Closes, times: pd.DatetimeIndex, settings: CacheSettings, *, step_hours: int
) -> pd.DataFrame:
    """Compute the vol-scaled label of every asset and horizon at the decision times.

    With ``end`` the asset's last close at or before the decision time ``t``, the label of
    horizon ``h`` is ``h`` of the asset's own bars ahead, the steps a model forecasts:
    ``label = log(P[end + h] / P[end]) / (sigma * sqrt(h))``, ``log_return`` the numerator,
    ``sigma`` the EWMA volatility of the own-bar log returns at ``end`` (known at ``t``;
    `src.panel.ewma_volatility`) and ``label_time`` the close time of bar ``end + h``. A
    decision is labelled only when its context is valid (`context_ends`: `lookback` closes),
    bar ``end + h`` exists, the label is finite and
    a new own bar closed since the previous decision time (``t - step_hours``): one decision
    per ``end``, the first after each own close. A market that trades around the clock keeps
    every decision; a market that closes has no decisions while it is closed, which would only
    repeat the label and the forecast of its last close.

    Parameters
    ----------
    closes : dict of str to pd.Series
        1h closes per asset, indexed by close times (UTC).
    times : pd.DatetimeIndex
        Decision times `step_hours` apart, shape ``(n_times,)``.
    settings : CacheSettings
        Horizons, the EWMA span and `lookback`.
    step_hours : int
        Hours between two decision times.

    Returns
    -------
    pd.DataFrame
        ``time``, ``asset``, ``horizon``, ``label_time``, ``label``, ``log_return``, ``sigma``;
        one row per labelled decision and horizon, no two of an asset sharing an ``end``.
    """
    parts: list[pd.DataFrame] = []
    for asset, series in closes.items():
        ends = context_ends(series, times, lookback=settings.lookback)
        previous = last_at_or_before(
            pd.DatetimeIndex(series.index), times - pd.Timedelta(hours=step_hours)
        )
        sigma = ewma_volatility(series, times, span_bars=settings.span_bars)
        prices = series.to_numpy(dtype=float)
        for horizon in settings.horizons:
            later = ends + horizon
            # one decision per own close: a new bar closed since the previous decision time
            # (an invalid context has end -1, never above `previous`)
            is_labelled = (ends > previous) & (later < len(series))
            end, label_end = ends[is_labelled], later[is_labelled]
            log_return = np.log(prices[label_end] / prices[end])
            columns = {
                "time": times[is_labelled],
                "asset": asset,
                "horizon": horizon,
                "label_time": series.index[label_end],
                "label": log_return / (sigma[is_labelled] * np.sqrt(horizon)),
                "log_return": log_return,
                "sigma": sigma[is_labelled],
            }
            parts.append(pd.DataFrame(columns))
    frame = pd.concat(parts, ignore_index=True)
    return frame[np.isfinite(frame["label"])].reset_index(drop=True)


def _forecast_rows(keys: pd.DataFrame, seed: int, quantiles: np.ndarray) -> pd.DataFrame:
    # one forecast row per key row: time, asset, horizon, seed and the nine quantiles
    table = pd.DataFrame(quantiles, columns=list(QUANTILE_COLUMNS))
    return pd.concat([keys[list(KEYS)].reset_index(drop=True).assign(seed=seed), table], axis=1)


def _baseline_forecasts(
    model: str,
    labels: pd.DataFrame,
    training: dict[str, pd.DataFrame],
    closes: Closes,
    settings: CacheSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # the baseline's forecasts of every window with labels, and the fits that did not converge
    # (GARCH): their asset gets no row in that window and is listed with arch's flag
    if model not in BASELINES:
        raise ValueError(f"{model} is not a baseline of {BASELINES}")
    parts: list[pd.DataFrame] = []
    failed: list[dict[str, object]] = []
    for fold, _ in cache_windows(settings):
        start = fold.evaluation_start
        test = labels[labels["window"] == fold.name].reset_index(drop=True)
        if test.empty:
            continue
        skipped: set[str] = set()
        train = training[fold.name]
        if model == "random-walk":
            quantiles = random_walk(train, test)
        elif model == "climatology":
            quantiles = climatology(train, test)
        elif model == "ewma-normal":
            quantiles = ewma_normal(test)
        elif model == "garch-normal":
            quantiles, not_fitted = garch_normal(
                test, closes, start=start, train_days=settings.train_days
            )
            skipped = set(not_fitted)
            for asset, flag in not_fitted.items():
                failed.append(
                    {"model": model, "window": fold.name, "asset": asset, "convergence_flag": flag}
                )
        elif model == "linear":
            quantiles = linear(
                train, test, closes, lags=settings.linear_lags, ridge=settings.linear_ridge
            )
        else:
            raise ValueError(f"{model} is not a baseline of {BASELINES}")
        # only a listed GARCH asset is left out; any other row is written as before (an invalid
        # forecast stays a row, which the validity check counts)
        is_fitted = ~test["asset"].astype(str).isin(skipped).to_numpy()
        parts.append(_forecast_rows(test[is_fitted], 0, quantiles[is_fitted]))
        logger.info("%s %s: %d rows", model, fold.name, int(is_fitted.sum()))
    columns = ["model", "window", "asset", "convergence_flag"]
    return pd.concat(parts, ignore_index=True), pd.DataFrame(failed, columns=columns)


def _list_windows(folder: Path, model: str, windows: list[str]) -> None:
    # the windows each model was built with (all of the run's, or those of --windows, in the
    # labels' order d, t, l), one row per model in windows.csv of the run: a record for people
    # (nothing reads it), since a resume never adds windows to a written model
    path = folder / "windows.csv"
    row = pd.DataFrame({"model": [model], "windows": [" ".join(windows)]})
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def _list_not_fitted(folder: Path, not_fitted: pd.DataFrame) -> None:
    # a baseline fit that did not converge leaves its asset's window without forecasts; it is
    # logged and appended to not_fitted.csv of the run, never filled (a rerun after a crash may
    # append it twice: readers drop duplicate rows)
    if not_fitted.empty:
        return
    logger.warning("fits that did not converge, left out:\n%s", not_fitted.to_string(index=False))
    path = folder / "not_fitted.csv"
    not_fitted.to_csv(path, mode="a", header=not path.exists(), index=False)


def training_labels(closes: Closes, start: pd.Timestamp, settings: CacheSettings) -> pd.DataFrame:
    """Compute the labels that fit the baselines of a window: known before the window starts.

    Decisions are hourly over the `train_days` before `start`; a label is kept only when its
    last bar closed by `start` (``label_time <= start``), so no fitted value sees the window.

    Parameters
    ----------
    closes : dict of str to pd.Series
        1h closes per asset, indexed by close times (UTC).
    start : pd.Timestamp
        The window start.
    settings : CacheSettings
        Horizons, EWMA span, context rules and `train_days`.

    Returns
    -------
    pd.DataFrame
        The columns of `decision_labels`.
    """
    times = hour_grid(start - pd.Timedelta(days=settings.train_days), start, step_hours=1)
    labels = decision_labels(closes, times, settings, step_hours=1)
    return labels[labels["label_time"] <= start].reset_index(drop=True)


def label_quantiles(
    prices: np.ndarray, last_close: np.ndarray, sigma: np.ndarray, horizons: tuple[int, ...]
) -> np.ndarray:
    """Turn price quantiles into quantiles of the vol-scaled label (the label's own transform).

    Each price quantile ``q`` of step ``h`` (``h`` own bars ahead, the label's horizon) becomes
    ``log(q / P[end]) / (sigma * sqrt(h))``. The transform is increasing, so the order of the
    quantiles is kept.

    Parameters
    ----------
    prices : np.ndarray
        Price quantiles, shape ``(n, max_horizon, k)`` (k levels, the cache's 9 or any other);
        step ``h`` is ``prices[:, h - 1]``.
    last_close : np.ndarray
        ``P[end]``, the last close of each context, shape ``(n,)``.
    sigma : np.ndarray
        The label's volatility at ``end``, shape ``(n,)``.
    horizons : tuple of int
        Horizons in own bars.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n, len(horizons), k)``; NaN or -inf where a price quantile
        is not positive (an invalid forecast).
    """
    steps = np.asarray(horizons)
    scale = sigma[:, None, None] * np.sqrt(steps)[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(prices[:, steps - 1] / last_close[:, None, None]) / scale


def _zero_shot_forecasts(
    model: str,
    labels: pd.DataFrame,
    panel: _Panel,
    settings: CacheSettings,
    *,
    samples: Path,
    seeds: tuple[int, ...],
) -> pd.DataFrame | None:
    # the model's labelled rows per window and their contexts; every model call forecasts
    # max(horizons) steps, and only the labelled horizons of a decision are written
    forecaster = _load_model(model, settings)
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold, _ in cache_windows(settings):
        rows = on_model_grid(labels[labels["window"] == fold.name], model, settings)
        if rows.empty:
            continue
        contexts = with_last_close(rows.drop_duplicates(["time", "asset"]), panel.closes)
        windows.append((rows, contexts))
        logger.info("%s %s: %d decisions", model, fold.name, len(contexts))
    if model_kind(model, settings) == "kronos":
        pooled = _pooled_kronos_quantiles(
            forecaster, windows, panel.bars, settings, samples=samples, seeds=seeds
        )
        if pooled is None:
            return None
        prices = pooled
    else:
        seed_everything(0)
        steps, chunk = max(settings.horizons), _chunk(model, settings)
        prices = [
            _price_quantiles(
                forecaster.predict,
                _close_windows(contexts, panel.closes, settings.lookback),
                steps,
                chunk,
            )
            for _, contexts in windows
        ]
    del forecaster
    torch.cuda.empty_cache()
    parts = [
        _labelled_rows(rows, contexts, window_prices, settings)
        for (rows, contexts), window_prices in zip(windows, prices, strict=True)
    ]
    return pd.concat(parts, ignore_index=True)


def _load_model(model: str, settings: CacheSettings) -> Forecaster:
    # the weights (and Kronos's tokenizer) must be at the commits pinned in the configuration
    kind, repository = settings.zero_shot[model]
    require_revision(repository, settings.revisions[model])
    extra: dict[str, object] = {}
    if kind == "kronos":
        require_revision(settings.kronos_tokenizer, settings.kronos_tokenizer_revision)
        extra["settings"] = KronosSettings(
            tokenizer=settings.kronos_tokenizer,
            n_samples=settings.kronos_paths,
            temperature=settings.kronos_temperature,
            top_p=settings.kronos_top_p,
            step=Timeframe.H1.timedelta,
        )
    return build_forecaster(kind, name=repository, device=settings.device, **extra)


def _chunk(model: str, settings: CacheSettings) -> int:
    # Moirai runs in a worker process, so it takes larger calls
    return settings.moirai_chunk if model_kind(model, settings) == "moirai2" else settings.chunk


def _pooled_kronos_quantiles(
    forecaster: Forecaster,
    windows: list[tuple[pd.DataFrame, pd.DataFrame]],
    bars: Bars,
    settings: CacheSettings,
    *,
    samples: Path,
    seeds: tuple[int, ...],
) -> list[np.ndarray] | None:
    # Kronos samples `kronos_paths` paths per seed: each seed of `seeds` is set once, every
    # window is sampled in order and the paths are saved to samples/seed<s>.npz, one array per
    # window named by the window, so seeds can run on separate machines with the same result.
    # Once every seed of the run is saved, the paths of all seeds are pooled (3 x 20 = 60)
    # before the quantiles are taken; until then there is no forecast (None)
    kronos = cast("KronosForecaster", forecaster)
    names = [str(rows["window"].iloc[0]) for rows, _ in windows]
    steps, chunk = max(settings.horizons), settings.chunk
    samples.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = samples / f"seed{seed}.npz"
        if path.exists():
            continue
        seed_everything(seed)
        paths: dict[str, np.ndarray] = {}
        for name, (_, contexts) in zip(names, windows, strict=True):
            candles = candle_windows(contexts, bars, settings.lookback)
            parts = [
                kronos.sample_closes(candles[first : first + chunk], steps)
                for first in range(0, len(candles), chunk)
            ]
            paths[name] = np.concatenate(parts)
        partial = samples / f"seed{seed}.partial.npz"
        # the window names are the keys (cast: numpy types every keyword as its own options)
        np.savez(partial, **cast("dict[str, Any]", paths))
        partial.rename(path)
    missing = [seed for seed in settings.seeds if not (samples / f"seed{seed}.npz").exists()]
    if missing:
        logger.info("%s: seeds %s are not sampled yet, no forecast written", samples, missing)
        return None
    pooled: list[list[np.ndarray]] = [[] for _ in windows]
    rows = [len(contexts) for _, contexts in windows]
    for seed in settings.seeds:
        saved = saved_paths(samples / f"seed{seed}.npz", names, rows)
        for index, paths_of in enumerate(saved):
            pooled[index].append(paths_of)
    levels = np.asarray(QUANTILE_LEVELS)
    return [
        np.moveaxis(np.quantile(np.concatenate(paths, axis=1), levels, axis=1), 0, -1)
        for paths in pooled
    ]


def saved_paths(path: Path, windows: list[str], rows: list[int]) -> list[np.ndarray]:
    """Return a seed's saved Kronos paths of `windows`, read by window name.

    ``samples/<model>/seed<s>.npz`` holds one array per window, named by the window. A file
    saved before the arrays were named (``arr_0``, ``arr_1``, ...) is read in order, only when
    it holds exactly as many arrays as `windows`; every array must have its window's number of
    contexts, which catches a legacy file read with other windows of other sizes (windows of
    the same size cannot be told apart in a legacy file, which is why the arrays are named).

    Parameters
    ----------
    path : Path
        The seed's ``.npz`` file.
    windows : list of str
        The window names, in the order of the arrays returned.
    rows : list of int
        The number of contexts of each window (its rows in the model's grid).

    Returns
    -------
    list of np.ndarray
        The paths of each window, shape ``(contexts, paths, steps)``.

    Raises
    ------
    ValueError
        If the file lacks a window or an array has another number of contexts than its window.
    """
    with np.load(path) as saved:
        if all(name in saved.files for name in windows):
            arrays = [saved[name] for name in windows]
        elif sorted(saved.files) == sorted(f"arr_{index}" for index in range(len(windows))):
            logger.warning("%s: unnamed windows, read in order %s", path, windows)
            arrays = [saved[f"arr_{index}"] for index in range(len(windows))]
        elif all(name.startswith("arr_") for name in saved.files):
            raise ValueError(
                f"{path} holds {len(saved.files)} unnamed arrays for {len(windows)} windows "
                f"{windows}; delete {path} and run the build again to sample it with these "
                "windows"
            )
        else:
            missing = sorted(set(windows) - set(saved.files))
            raise ValueError(
                f"{path} lacks windows {missing} (it holds {sorted(saved.files)}); delete {path} "
                "and run the build again to sample it with these windows"
            )
    for name, count, array in zip(windows, rows, arrays, strict=True):
        if len(array) != count:
            raise ValueError(
                f"{path}: window {name} holds {len(array)} contexts, its grid {count}; delete "
                f"{path} and run the build again to sample it with these windows"
            )
    return arrays


def _labelled_rows(
    rows: pd.DataFrame, contexts: pd.DataFrame, prices: np.ndarray, settings: CacheSettings
) -> pd.DataFrame:
    # the label quantiles of every horizon of each context, kept only where the decision has
    # a label at that horizon (a decision near the end of the data may lack the longer ones)
    quantiles = label_quantiles(
        prices, contexts["last_close"].to_numpy(), contexts["sigma"].to_numpy(), settings.horizons
    )
    keys = contexts.loc[contexts.index.repeat(len(settings.horizons))].reset_index(drop=True)
    keys["horizon"] = np.tile(settings.horizons, len(contexts))
    frame = _forecast_rows(keys, 0, quantiles.reshape(-1, len(QUANTILE_COLUMNS)))
    return frame.merge(rows[list(KEYS)], on=list(KEYS), validate="one_to_one")


def with_last_close(decisions: pd.DataFrame, closes: Closes) -> pd.DataFrame:
    """Add each decision's context end and last close; the order of the cache's contexts.

    Every labelled decision has a valid context (`decision_labels` checked it). The rows are
    grouped by asset in order of first appearance, the order in which the builder samples
    them (row i of a Kronos ``samples/<model>/seed<s>.npz`` window is row i here).

    Parameters
    ----------
    decisions : pd.DataFrame
        One row per decision, with ``time`` and ``asset``.
    closes : dict of str to pd.Series
        Each asset's 1h closes (`src.panel.hourly_closes`).

    Returns
    -------
    pd.DataFrame
        `decisions` plus ``end`` (the position of the context's last close) and ``last_close``.
    """
    parts: list[pd.DataFrame] = []
    for asset, group in decisions.groupby("asset", sort=False):
        series = closes[str(asset)]
        ends = last_at_or_before(pd.DatetimeIndex(series.index), pd.DatetimeIndex(group["time"]))
        parts.append(group.assign(end=ends, last_close=series.to_numpy(dtype=float)[ends]))
    return pd.concat(parts, ignore_index=True)


def _close_windows(contexts: pd.DataFrame, closes: Closes, lookback: int) -> np.ndarray:
    # the last `lookback` closes of each context, shape (n_contexts, lookback), oldest first
    windows = np.empty((len(contexts), lookback))
    steps = np.arange(1 - lookback, 1)
    ends = contexts["end"].to_numpy()
    for asset, rows in contexts.groupby("asset").indices.items():
        values = closes[str(asset)].to_numpy(dtype=float)
        windows[rows] = values[ends[rows, None] + steps]
    return windows


def candle_windows(contexts: pd.DataFrame, bars: Bars, lookback: int) -> list[pd.DataFrame]:
    """Return the last `lookback` candles of each context, the input Kronos reads.

    Kronos reads OHLC (mid prices for bid/ask sources), volume (Dukascopy: bid volume) and
    amount = volume * close, indexed by the bars' open times. Its future timestamps continue
    every hour (`KronosForecaster`, step = the 1h bar): the real times of the next own bars
    are unknown at the decision, and step h is read as the h-th next own bar, as for every model.

    Parameters
    ----------
    contexts : pd.DataFrame
        The contexts, with ``asset`` and ``end`` (`with_last_close`).
    bars : dict
        Each asset's bars per timeframe (`src.panel.load_panel_bars`).
    lookback : int
        Candles per context.

    Returns
    -------
    list of pd.DataFrame
        One frame of ``open``, ``high``, ``low``, ``close``, ``volume``, ``amount`` per context.
    """
    candles: dict[str, pd.DataFrame] = {}
    for asset in contexts["asset"].unique():
        frame = bars[str(asset)][Timeframe.H1]
        prefix = "mid_" if "mid_close" in frame.columns else ""
        table = pd.DataFrame(
            {field: frame[prefix + field] for field in ("open", "high", "low", "close")}
        )
        table["volume"] = frame["volume"] if "volume" in frame.columns else frame["bid_volume"]
        table["amount"] = table["volume"] * table["close"]
        candles[str(asset)] = table
    pairs = zip(contexts["asset"], contexts["end"], strict=True)
    return [candles[asset].iloc[end + 1 - lookback : end + 1] for asset, end in pairs]


def _price_quantiles(
    predict: Callable[..., Forecast],
    inputs: np.ndarray | list[pd.DataFrame],
    horizon: int,
    chunk: int,
) -> np.ndarray:
    parts = [np.empty((0, horizon, len(QUANTILE_COLUMNS)))]
    for first in range(0, len(inputs), chunk):
        part = inputs[first : first + chunk]
        parts.append(predict(part, horizon).checked_quantiles(len(part), horizon))
    return np.concatenate(parts)
