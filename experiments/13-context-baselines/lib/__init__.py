"""Baselines fitted on the zero-shot models' context (13-context-baselines).

`context_forecasts` refits the random walk, climatology, ridge regression and GARCH(1,1) of
02-forecasts on only the last `window` own bars before each decision (the 512 bars a
zero-shot model reads) instead of the 730 days before the quarter; `ContextCache` serves them
in place of the cache's baselines so rq1's test (`src.benchmark.null_tables`) runs unchanged.
Modules import each other relatively, so the package works under any module name.
"""

from ._context import (
    CONTEXT_BASELINES,
    ContextCache,
    ContextSettings,
    context_cache,
    context_forecasts,
    garch_spreads,
    rolling_climatology,
    rolling_random_walk,
    rolling_ridge,
)

__all__ = [
    "CONTEXT_BASELINES",
    "ContextCache",
    "ContextSettings",
    "context_cache",
    "context_forecasts",
    "garch_spreads",
    "rolling_climatology",
    "rolling_random_walk",
    "rolling_ridge",
]
