"""The test of the context's length after a closure, at the same encoding (Q1).

The benchmark input and trading hours only of 08-closures show a model the same kind of
input, real closes only and no closed hour; they differ only in how many bars reach the
model, so their difference is the effect of the context's length. The test is that of
08-closures Q1 (its `closure_tests`): closure-day series, a one-sided Diebold-Mariano test
and the skill's moving-block bootstrap interval, with 08's settings.
"""

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark import SCORED
from src.stats import (
    MIN_BOOTSTRAP_BLOCKS,
    block_indices,
    diebold_mariano,
    has_enough_blocks,
    holm,
    politis_white_block,
)

TRADING_HOURS = "trading hours only"
"""08-closures' reference: the own bars inside the last 512 calendar hours."""

BENCHMARK_INPUT = "benchmark input"
"""08-closures' benchmark input: the last 512 own bars."""

CLOSED = "closed markets"
"""08-closures' group of the markets that close."""

BASELINE = "ewma-normal"
"""08-closures' baseline, which reads no context and is not tested."""


@dataclass(frozen=True, slots=True)
class LengthSettings:
    """The settings of the test, read from the 08-closures run's ``config.toml`` options.

    Attributes
    ----------
    alpha : float
        One-sided level; the interval's level is ``1 - alpha``.
    min_closure_days : int
        A model with fewer closure days is not tested.
    bootstrap_resamples : int
        Bootstrap draws of the skill interval.
    bootstrap_seed : int
        The seed of the draws (with the model's checksum).
    """

    alpha: float
    min_closure_days: int
    bootstrap_resamples: int
    bootstrap_seed: int


def read_length_settings(options: dict[str, Any]) -> LengthSettings:
    """Read the test's settings from the ``[options]`` of an 08-closures run.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    return LengthSettings(
        alpha=float(options["alpha"]),
        min_closure_days=int(options["min_closure_days"]),
        bootstrap_resamples=int(options["bootstrap_resamples"]),
        bootstrap_seed=int(options["bootstrap_seed"]),
    )


def closure_day_series(frame: pd.DataFrame) -> pd.Series:
    """Return the mean CRPS per closure day: over the assets at each time, then over the times.

    As 08-closures' `closure_day_means`: each decision time weighs the same within its day and
    each day the same in a mean over the days.

    Parameters
    ----------
    frame : pd.DataFrame
        Scored losses of one model and input: ``closure_day``, ``time`` and ``crps``.

    Returns
    -------
    pd.Series
        Mean CRPS per closure day, in day order.
    """
    times = frame.groupby(["closure_day", "time"])["crps"].mean()
    return times.groupby(level="closure_day").mean().sort_index()


def _interval(
    alternative: np.ndarray,
    reference: np.ndarray,
    *,
    block: int,
    settings: LengthSettings,
    seed: list[int],
) -> tuple[float, float]:
    # the moving-block bootstrap interval of the skill 1 - mean(alt) / mean(ref), paired days
    rng = np.random.default_rng(seed)
    rows = block_indices(len(reference), rng, n_resamples=settings.bootstrap_resamples, block=block)
    draws = 1 - alternative[rows].mean(axis=1) / reference[rows].mean(axis=1)
    tail = settings.alpha / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def _test(
    alternative: np.ndarray, reference: np.ndarray, settings: LengthSettings, *, seed: list[int]
) -> dict[str, object]:
    # the benchmark input against trading hours only on the closure days (08's test)
    row: dict[str, object] = {
        "days": len(reference),
        "reference_crps": float(reference.mean()),
        "alternative_crps": float(alternative.mean()),
        "skill": float(1 - alternative.mean() / reference.mean()),
    }
    if len(reference) < settings.min_closure_days:
        return row | {"status": f"not tested: fewer than {settings.min_closure_days} closure days"}
    test = diebold_mariano(alternative, reference, overlap=0)
    block = min(max(1, politis_white_block(alternative - reference)), len(reference))
    if np.isnan(test["statistic"]) or not has_enough_blocks(len(reference), block):
        status = f"not tested: no variance or fewer than {MIN_BOOTSTRAP_BLOCKS} blocks"
        return row | {"block": block, "status": status}
    low, high = _interval(alternative, reference, block=block, settings=settings, seed=seed)
    return row | {
        "skill_low": low,
        "skill_high": high,
        "block": block,
        "dm_statistic": test["statistic"],
        "p_value": test["p_less"],
        "status": SCORED,
    }


TEST_COLUMNS = [
    "model", "family", "days", "reference_crps", "alternative_crps", "skill", "skill_low",
    "skill_high", "block", "dm_statistic", "p_value", "status", "adjusted_p",
]  # fmt: skip
"""The columns of `length_tests`."""


def length_tests(
    losses: pd.DataFrame, families: dict[str, str], settings: LengthSettings
) -> pd.DataFrame:
    """Test the benchmark input against trading hours only, per zero-shot model.

    On the closed markets, each input's scored losses become a series of closure days
    (`closure_day_series`) and the benchmark input is compared with trading hours only by a
    one-sided Diebold-Mariano test (H1: the benchmark input has the lower CRPS) and the
    skill's moving-block bootstrap interval with the Politis-White block (08-closures' test);
    Holm runs within each of 08's families over the tested models.

    Parameters
    ----------
    losses : pd.DataFrame
        ``metrics.parquet`` of an 08-closures run.
    families : dict of str to str
        Each model's Holm family in 08-closures (``on time`` or ``late``).
    settings : LengthSettings
        The settings of 08-closures.

    Returns
    -------
    pd.DataFrame
        One row per model with both inputs (`TEST_COLUMNS`).
    """
    is_closed = (losses["group"] == CLOSED) & (losses["model"] != BASELINE) & losses["is_scored"]
    closed = losses.loc[is_closed & losses["representation"].isin([TRADING_HOURS, BENCHMARK_INPUT])]
    rows: list[dict[str, object]] = []
    for model, frame in closed.groupby("model", sort=True):
        series = {
            str(arm): closure_day_series(part) for arm, part in frame.groupby("representation")
        }
        if set(series) != {TRADING_HOURS, BENCHMARK_INPUT}:
            continue
        reference, alternative = series[TRADING_HOURS], series[BENCHMARK_INPUT]
        if not reference.index.equals(alternative.index):
            raise ValueError(f"{model}: the two inputs are scored on other closure days")
        seed = [settings.bootstrap_seed, zlib.crc32(str(model).encode())]
        test = _test(alternative.to_numpy(), reference.to_numpy(), settings, seed=seed)
        rows.append({"model": model, "family": families[str(model)]} | test)
    table = pd.DataFrame(rows, columns=TEST_COLUMNS[:-1])
    adjusted = np.full(len(table), np.nan)
    for indices in table.groupby("family").indices.values():
        adjusted[indices] = holm(table["p_value"].to_numpy(dtype=float)[indices])
    return table.assign(adjusted_p=adjusted)


def length_by_market(losses: pd.DataFrame) -> pd.DataFrame:
    """Return, per model and market, the skill of the benchmark input over trading hours only.

    Descriptive: ``1 - CRPS_benchmark / CRPS_trading`` over the scored decisions of the
    closed markets (no test).

    Parameters
    ----------
    losses : pd.DataFrame
        ``metrics.parquet`` of an 08-closures run.

    Returns
    -------
    pd.DataFrame
        ``model``, ``market``, ``decisions`` (of trading hours only), ``decisions_benchmark``
        (each input drops its own invalid forecasts) and ``skill``.
    """
    is_closed = (losses["group"] == CLOSED) & (losses["model"] != BASELINE) & losses["is_scored"]
    closed = losses.loc[is_closed]
    rows: list[dict[str, object]] = []
    for (model, market), frame in closed.groupby(["model", "market"], sort=True):
        trading = frame.loc[frame["representation"] == TRADING_HOURS, "crps"]
        benchmark = frame.loc[frame["representation"] == BENCHMARK_INPUT, "crps"]
        if trading.empty or benchmark.empty:
            continue
        skill = 1 - float(benchmark.mean()) / float(trading.mean())
        rows.append(
            {
                "model": model,
                "market": market,
                "decisions": len(trading),
                "decisions_benchmark": len(benchmark),
                "skill": skill,
            }
        )
    columns = ["model", "market", "decisions", "decisions_benchmark", "skill"]
    return pd.DataFrame(rows, columns=columns)
