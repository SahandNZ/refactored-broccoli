"""Check the dispersion of Kronos's sampled intervals in a 02-forecasts run (a diagnostic).

The run is ``experiments/02-forecasts/runs/<run-id>/``.

For each Kronos model, the coverage and median width of the 10-90 % interval
(`src.benchmark.interval_dispersion`) from the pooled paths of every seed (the cache's
forecast) and from seed 0 alone, on all its decisions; with --top-p, also the nucleus-sampling
sensitivity: the first model resampled with seed 0 at top_p 0.9 (the reference setting) and
1.0 (no cut) on `--contexts` evenly spaced contexts of the last window (local GPU, a few
minutes). Writes ``tables/kronos_dispersion.csv`` of the run.

Usage: .venv/bin/python scripts/check_kronos_dispersion.py
--run-id ID [--top-p] [--contexts N]
"""

import argparse
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from src import LAB_DIR
from src.benchmark import (
    CacheSettings,
    ForecastCache,
    cache_settings,
    candle_windows,
    interval_dispersion,
    on_model_grid,
    read_cache,
    saved_paths,
    with_last_close,
)
from src.enums import Market, Timeframe
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS
from src.experiment import load_study, seed_everything, study_runs
from src.models import KronosSettings, build_forecaster, require_revision
from src.panel import hourly_closes, load_panel_bars

if TYPE_CHECKING:
    from src.models._kronos import KronosForecaster

logger = logging.getLogger("02-forecasts")

STUDY = LAB_DIR / "experiments" / "02-forecasts"
# contexts of the last window resampled by the top_p sensitivity (seed 0; minutes on a laptop)
TOP_P_CONTEXTS = 150


@dataclass(frozen=True, slots=True)
class Windows:
    """A Kronos model's windows and their contexts, in the builder's order."""

    names: list[str]
    contexts: list[pd.DataFrame]
    bars: dict[str, dict[Timeframe, pd.DataFrame]]


def kronos_windows(cache: ForecastCache, settings: CacheSettings, model: str) -> Windows:
    """Return the model's grid contexts per window, as the builder ordered them.

    The windows are those the model has forecasts for (a run built for some windows only, e.g.
    with ``--windows``, holds paths for those), in the builder's order: d1-d4, t1-t4, l1-l12.
    """
    markets = {
        str(a): Market(m)
        for a, m in zip(cache.assets["asset"], cache.assets["market"], strict=True)
    }
    bars = load_panel_bars(LAB_DIR / "data", markets, (Timeframe.H1,))
    closes = hourly_closes(bars)
    labels = cache.labels
    names: list[str] = []
    contexts: list[pd.DataFrame] = []
    built = set(cache.forecasts(model)["window"].astype(str))
    order = [fold.name for fold in (*DEVELOPMENT_FOLDS, *TEST_FOLDS, *LEAKAGE_FOLDS)]
    for window in [name for name in order if name in built]:
        rows = on_model_grid(labels[labels["window"] == window], model, settings)
        if not rows.empty:
            names.append(window)
            contexts.append(with_last_close(rows.drop_duplicates(["time", "asset"]), closes))
    return Windows(names, contexts, bars)


def top_p_rows(
    windows: Windows, cache: ForecastCache, settings: CacheSettings, model: str, contexts: int
) -> pd.DataFrame:
    """Resample `contexts` contexts of the last window with seed 0 at top_p 0.9 and 1.0."""
    last = windows.contexts[-1]
    chosen = last.iloc[np.linspace(0, len(last) - 1, contexts).round().astype(int)]
    candles = candle_windows(chosen, windows.bars, settings.lookback)
    _, repository = settings.zero_shot[model]
    require_revision(repository, settings.revisions[model])
    require_revision(settings.kronos_tokenizer, settings.kronos_tokenizer_revision)
    parts: list[pd.DataFrame] = []
    for top_p in (0.9, 1.0):
        sampling = KronosSettings(
            tokenizer=settings.kronos_tokenizer,
            n_samples=settings.kronos_paths,
            temperature=settings.kronos_temperature,
            top_p=top_p,
            step=Timeframe.H1.timedelta,
        )
        forecaster = cast(
            "KronosForecaster",
            build_forecaster("kronos", name=repository, device=settings.device, settings=sampling),
        )
        seed_everything(0)
        paths = forecaster.sample_closes(candles, max(settings.horizons))
        table = interval_dispersion([paths], [chosen], cache.labels, settings.horizons)
        parts.append(table.assign(sample=f"{contexts} contexts of the last window", top_p=top_p))
    return pd.concat(parts).assign(model=model, paths=f"seed 0, {settings.kronos_paths}")


def main() -> None:
    """Write the run's ``tables/kronos_dispersion.csv``."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--top-p", action="store_true")
    parser.add_argument("--contexts", type=int, default=TOP_P_CONTEXTS)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    folder = study_runs(STUDY) / arguments.run_id
    settings = cache_settings(load_study(folder / "config.toml"))
    cache = read_cache(folder)
    models = [name for name, (kind, _) in settings.zero_shot.items() if kind == "kronos"]
    tables: list[pd.DataFrame] = []
    for model in models:
        windows = kronos_windows(cache, settings, model)
        rows = [len(window) for window in windows.contexts]
        seeds = [
            saved_paths(folder / "samples" / model / f"seed{seed}.npz", windows.names, rows)
            for seed in settings.seeds
        ]
        pooled = [np.concatenate(window, axis=1) for window in zip(*seeds, strict=True)]
        top_p = settings.kronos_top_p
        for paths, label in ((pooled, f"{len(seeds)} seeds pooled"), (seeds[0], "seed 0")):
            table = interval_dispersion(paths, windows.contexts, cache.labels, settings.horizons)
            count = paths[0].shape[1]
            tables.append(
                table.assign(
                    model=model, sample="all decisions", top_p=top_p, paths=f"{label}, {count}"
                )
            )
        if arguments.top_p and model == models[0]:
            tables.append(top_p_rows(windows, cache, settings, model, arguments.contexts))
    columns = [
        "model",
        "sample",
        "top_p",
        "paths",
        "horizon",
        "decisions",
        "coverage",
        "median_width",
    ]
    out = folder / "tables" / "kronos_dispersion.csv"
    pd.concat(tables)[columns].to_csv(out, index=False, float_format="%.4f")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
