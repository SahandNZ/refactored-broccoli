"""Power of the rq1 decision rule: the smallest CRPS gain over the best baseline it detects.

A null result ("no model beats the best baseline") means something only if the rule could have
found a real but small edge. `power_table` plants an edge of known size into every rq1 cell and
counts how often the rq1 rule detects it in block-bootstrap resamples;
`minimum_detectable_effects` turns those detection rates into the minimum detectable effect
(MDE) of each cell, in % of the best baseline's CRPS. `power_tables` runs rq1 and both steps in
one call, the computation of 03-rq1-null and 09-mde-threshold; `threshold_shares` and
`powered_threshold` choose rq1's threshold (09).
"""

import numpy as np
import pandas as pd
from scipy.special import stdtr

from src.benchmark._cache import ForecastCache
from src.benchmark._decisions import CELL, cell_stream
from src.benchmark._scores import baseline_names, beats_baseline, null_tables
from src.benchmark._settings import NullRule, StudySettings
from src.stats import CONSTANT_TOLERANCE, circular_block_indices

POWER_CELL = ["model", *CELL]
"""The columns that name one cell of the power tables (the cells of rq1)."""


def _long_run_variances(drawn: np.ndarray, overlap: int) -> tuple[np.ndarray, np.ndarray]:
    # the Newey-West variance of every resample, with the lags `diebold_mariano` takes:
    # max(overlap, Andrews' lags of that resample); autocovariances of all rows at once by FFT
    n = drawn.shape[1]
    centred = drawn - drawn.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=1)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n, axis=1)[:, :n] / n
    rho = np.clip(autocovariance[:, 1] / autocovariance[:, 0], -0.99, 0.99)
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    andrews = np.floor(1.1447 * (alpha * n) ** (1 / 3)).astype(int)
    lags = np.minimum(np.maximum(overlap, andrews), n - 1)
    lag: np.ndarray = np.arange(n)[None, :]
    weights = np.where((lag >= 1) & (lag <= lags[:, None]), 1 - lag / (lags[:, None] + 1), 0.0)
    variances = autocovariance[:, 0] + 2 * (weights * autocovariance).sum(axis=1)
    return variances, lags


def _window_sums(
    centred: np.ndarray, windows: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # per resample and window: the sum of the drawn centred differences and their count, so
    # a window's win for any planted gain g is count > 0 and sum - g * count < 0
    drawn, drawn_windows = centred[indices], windows[indices]
    names = np.unique(windows)
    sums = np.stack([np.where(drawn_windows == name, drawn, 0.0).sum(axis=1) for name in names])
    counts = np.stack([(drawn_windows == name).sum(axis=1) for name in names])
    return sums, counts


def _detection_rates(
    cell: pd.DataFrame, indices: np.ndarray, *, n_cells: int, rule: NullRule, alpha: float
) -> list[dict[str, float]]:
    # the planted model: the baseline plus the real model's loss difference with its mean taken
    # out (same noise and autocorrelation, no edge), minus the planted edge; then the rq1 rule
    # on every resample. A gain shifts every difference by the same constant, so each resample's
    # long-run variance and lags are computed once and serve every gain
    windows = cell["window"].to_numpy()
    baseline = cell["crps_baseline"].to_numpy(dtype=float)
    difference = cell["crps_model"].to_numpy(dtype=float) - baseline
    baseline_crps = float(baseline.mean())
    centred = difference - difference.mean()
    overlap, n = int(cell["overlap"].iloc[0]), len(cell)
    drawn = centred[indices]
    variances, _ = _long_run_variances(drawn, overlap)
    # a resample without variance beyond rounding has no test, as in `diebold_mariano`
    scale = 2 * baseline_crps + float(np.mean(np.abs(difference)))
    has_variance = drawn.std(axis=1) > CONSTANT_TOLERANCE * scale
    drawn_means = drawn.mean(axis=1)
    sums, counts = _window_sums(centred, windows, indices)
    horizon = overlap + 1
    factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    rows: list[dict[str, float]] = []
    for improvement in rule.improvements_pct:
        shift = improvement / 100 * baseline_crps
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = (drawn_means - shift) / np.sqrt(variances / n) * np.sqrt(factor)
        statistics = np.where(has_variance & (variances > 0) & (factor > 0), statistics, np.nan)
        p_values = stdtr(n - 1, statistics)
        wins = ((counts > 0) & (sums - shift * counts < 0)).sum(axis=0)
        detected = beats_baseline(
            np.minimum(n_cells * p_values, 1.0),
            wins,
            statistics,
            alpha=alpha,
            min_winning_quarters=rule.min_winning_quarters,
        )
        rows.append(
            {
                "improvement_pct": improvement,
                "baseline_crps": baseline_crps,
                "planted_crps": float((baseline + centred).mean() - shift),
                "detection_rate": float(detected.mean()),
            }
        )
    return rows


def power_table(times: pd.DataFrame, settings: StudySettings, rule: NullRule) -> pd.DataFrame:
    """Return how often the rq1 rule detects a planted CRPS gain, per cell and gain.

    The planted model of a cell has, at each decision point, the best baseline's loss plus the
    real model's loss difference to it with its mean (over the cell's points) taken out,
    minus ``improvement / 100`` times the baseline's mean CRPS. So it has the real decision
    points, number of assets, noise and autocorrelation of the cell, and a true edge of
    exactly `improvement` % of the baseline's CRPS (0: no edge). The points of the cell are
    resampled with a circular-block bootstrap (`src.stats.circular_block_indices`, the cell's
    ``block``
    of `null_tables`, each cell from its own stream, `cell_stream`; a point
    keeps its window, so a block may cross a window's end), and each resample is judged by
    the rq1 rule (`beats_baseline`): at least `min_winning_quarters` windows won and the
    one-sided Diebold-Mariano test (the cell's ``overlap``, lags ``max(overlap, Andrews)`` of
    the resample, t with ``n - 1`` degrees of freedom, as `src.stats.diebold_mariano`)
    favouring the planted model with p * m < `alpha`. Here m is the number of scored cells
    of the cell's Holm family (``family``) and p * m is Holm's adjusted p of the most
    significant cell, the smallest adjustment Holm makes; using it for every cell is
    conservative (Bonferroni), so the real rule has at least this power. With no edge, a
    cell is detected in about ``alpha / m`` of the resamples (the one-sided level), reported
    as ``false_positive_level``.

    Parameters
    ----------
    times : pd.DataFrame
        The ``"times"`` table of `null_tables`: one row per scored model, market, horizon and
        decision point, in time order, with the cell's ``overlap``, ``block`` and ``family``.
    settings : StudySettings
        ``alpha`` and the bootstrap draws and seed (the same draws for every gain of a cell,
        so a larger gain is never detected less often).
    rule : NullRule
        The planted gains (increasing from 0) and `min_winning_quarters`.

    Returns
    -------
    pd.DataFrame
        One row per model, market, horizon and gain: ``family``, ``baseline``, ``times``,
        ``quarters`` (the windows with a decision point), ``block``, ``overlap``,
        ``improvement_pct``, ``baseline_crps``, ``planted_crps`` (the planted model's mean
        CRPS over the cell's points), ``detection_rate`` and
        ``false_positive_level``.

    Raises
    ------
    ValueError
        If the gains do not increase from 0, or a cell has fewer points than its block.
    """
    gains = list(rule.improvements_pct)
    if not gains or gains[0] != 0 or sorted(set(gains)) != gains:
        raise ValueError(f"improvements must increase from 0, got {gains}")
    family_sizes = times.drop_duplicates(POWER_CELL).groupby("family").size()
    rows: list[dict[str, object]] = []
    for (model, market, horizon), cell in times.groupby(POWER_CELL):
        block, family = int(cell["block"].iloc[0]), str(cell["family"].iloc[0])
        rng = cell_stream(settings.bootstrap.seed, "power", model, market, horizon)
        indices = circular_block_indices(
            len(cell), rng, n_resamples=settings.bootstrap.n_resamples, block=block
        )
        n_cells = int(family_sizes[family])
        rates = _detection_rates(cell, indices, n_cells=n_cells, rule=rule, alpha=settings.alpha)
        named = {
            "model": model,
            "market": market,
            "horizon": horizon,
            "family": family,
            "baseline": baseline_names(cell["baseline"]),
            "times": len(cell),
            "quarters": cell["window"].nunique(),
            "block": block,
            "overlap": int(cell["overlap"].iloc[0]),
        }
        rows.extend(
            {**named, **rate, "false_positive_level": settings.alpha / n_cells} for rate in rates
        )
    return pd.DataFrame(rows)


def power_tables(
    cache: ForecastCache, models: list[str], settings: StudySettings, rule: NullRule
) -> dict[str, pd.DataFrame]:
    """Return the tables of rq1 (`null_tables`) and its power: ``power`` and ``mde``.

    The one computation of 03-rq1-null and 09-mde-threshold: the rq1 cells, the detection
    rate of every planted gain in every scored cell (`power_table`) and the minimum
    detectable effect of every cell (`minimum_detectable_effects`).

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The zero-shot models and the baselines.
    settings : StudySettings
        The folds, ``alpha`` and the bootstrap.
    rule : NullRule
        The rule of rq1 and its power.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"times"``, ``"quarters"`` and ``"cells"`` of `null_tables`, ``"power"`` and ``"mde"``.
    """
    tables = null_tables(cache, models, settings, rule)
    power = power_table(tables["times"], settings, rule)
    mde = minimum_detectable_effects(power, target=rule.power_target)
    return {**tables, "power": power, "mde": mde}


def minimum_detectable_effects(power: pd.DataFrame, *, target: float) -> pd.DataFrame:
    """Return the minimum detectable effect (MDE) of every cell of `power_table`.

    The MDE is the smallest planted gain detected in at least `target` of the resamples, by
    linear interpolation between the two gains of the grid around the first one that reaches
    `target`; the first gain itself if it already reaches it. A cell that never reaches
    `target` has no MDE: it is above the largest gain of the grid.

    Parameters
    ----------
    power : pd.DataFrame
        The table of `power_table`.
    target : float
        The power to reach, e.g. 0.8.

    Returns
    -------
    pd.DataFrame
        One row per model, market and horizon: ``family``, ``baseline``, ``times``,
        ``quarters``, ``mde_pct`` (NaN when not reached), ``is_reached``, ``largest_pct`` (the
        largest gain tried), ``zero_rate`` (the detection rate at the first gain, which
        `power_table` makes 0: no edge) and ``false_positive_level``.
    """
    rows: list[dict[str, object]] = []
    for (model, market, horizon), group in power.groupby(POWER_CELL):
        cell = group.sort_values("improvement_pct")
        gains = cell["improvement_pct"].to_numpy(dtype=float)
        rates = cell["detection_rate"].to_numpy(dtype=float)
        reached = np.flatnonzero(rates >= target)
        if len(reached) == 0:
            mde = np.nan
        elif reached[0] == 0:
            mde = float(gains[0])
        else:
            low, high = reached[0] - 1, reached[0]
            share = (target - rates[low]) / (rates[high] - rates[low])
            mde = float(gains[low] + share * (gains[high] - gains[low]))
        rows.append(
            {
                "model": model,
                "market": market,
                "horizon": horizon,
                "family": cell["family"].iloc[0],
                "baseline": cell["baseline"].iloc[0],
                "times": cell["times"].iloc[0],
                "quarters": cell["quarters"].iloc[0],
                "mde_pct": mde,
                "is_reached": len(reached) > 0,
                "largest_pct": gains[-1],
                "zero_rate": rates[0],
                "false_positive_level": cell["false_positive_level"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def detectable_cells(mde: pd.DataFrame, *, min_winning_quarters: int) -> pd.DataFrame:
    """Return the cells of `minimum_detectable_effects` the rq1 rule can ever detect.

    The rule needs at least `min_winning_quarters` winning quarters, so a cell with fewer
    quarters (a model released late in the scored folds) is never detected, whatever its
    edge: its MDE is not a power problem but a rule that cannot apply. The power shares count
    only the cells that can meet the rule.

    Parameters
    ----------
    mde : pd.DataFrame
        The table of `minimum_detectable_effects`, with ``quarters``.
    min_winning_quarters : int
        The fewest winning quarters of the rule.

    Returns
    -------
    pd.DataFrame
        The rows of `mde` with at least `min_winning_quarters` quarters.
    """
    return mde[mde["quarters"] >= min_winning_quarters]


def threshold_shares(mde: pd.DataFrame, *, thresholds: list[float]) -> pd.DataFrame:
    """Return the share of cells whose MDE is at most each threshold, over all and per horizon.

    A cell counts as resolved at threshold X when its MDE was reached and is at most X; a cell
    whose MDE was never reached is never resolved. The shares never fall as X grows.

    Parameters
    ----------
    mde : pd.DataFrame
        The table of `minimum_detectable_effects`, one row per model, market and horizon.
    thresholds : list of float
        The thresholds X in % of the baseline's CRPS, increasing.

    Returns
    -------
    pd.DataFrame
        One row per threshold: ``threshold_pct``, ``cells`` (all cells), ``share`` (over all
        cells) and ``share_<h>h`` per horizon h (e.g. ``share_1h``).

    Raises
    ------
    ValueError
        If `thresholds` is empty or not increasing.
    """
    if not thresholds or sorted(set(thresholds)) != thresholds:
        raise ValueError(f"thresholds must be increasing, got {thresholds}")
    is_reached = mde["is_reached"].to_numpy(dtype=bool)
    values = mde["mde_pct"].to_numpy(dtype=float)
    horizons = mde["horizon"].to_numpy(dtype=int)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        is_resolved = is_reached & (values <= threshold)
        row = {"threshold_pct": threshold, "cells": len(mde), "share": float(is_resolved.mean())}
        for horizon in sorted(set(horizons.tolist())):
            row[f"share_{horizon}h"] = float(is_resolved[horizons == horizon].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def powered_threshold(shares: pd.DataFrame, *, share: float, horizon: int) -> float:
    """Return the smallest threshold of `threshold_shares` whose share at `horizon` reaches `share`.

    The threshold 09-mde-threshold chooses for rq1: the smallest X of the grid at which at
    least `share` of the cells of `horizon` (column ``share_<horizon>h``) have an MDE at or
    below X. Fixed before the run, so the threshold is not picked by eye.

    Parameters
    ----------
    shares : pd.DataFrame
        The table of `threshold_shares`.
    share : float
        The share of cells to reach, in (0, 1], e.g. 0.8.
    horizon : int
        The horizon whose cells count (the one 03-rq1-null's Q2 judges).

    Returns
    -------
    float
        The threshold in % of the baseline's CRPS, or NaN when no threshold of the grid
        reaches `share` or `shares` has no cell of `horizon`.

    Raises
    ------
    ValueError
        If `share` is not in (0, 1].
    """
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}")
    column = f"share_{horizon}h"
    if column not in shares.columns:
        return float("nan")
    ordered = shares.sort_values("threshold_pct")
    thresholds = ordered["threshold_pct"].to_numpy(dtype=float)
    reached = thresholds[ordered[column].to_numpy(dtype=float) >= share]
    return float(reached[0]) if len(reached) > 0 else float("nan")
