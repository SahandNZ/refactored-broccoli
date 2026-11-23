"""The shared benchmark code of the benchmark paper: the code several of its studies need.

It holds the shared forecast cache (`build_cache` writes one run of study 02-forecasts with the
vol-scaled labels and the label quantiles of every zero-shot model and baseline; `read_cache`
and `study_cache` read it), the settings of the cache and of the studies on it
(`cache_settings`, `study_settings`, `benchmark_settings`, `null_rule`), the decisions and
their timing (`scored_decisions`, `cell_timing`, `complete_cells`), the contexts of a
decision (`context_ends`, `calendar_windows`), the baselines (`ewma_normal`), the scoring
against the best baseline (`decision_losses`, `best_baseline`, `beats_baseline`) and the power
analysis of rq1 (`power_tables`, `threshold_shares`, `powered_threshold`). The code that only one
study needs lives in that study's own `lib/` package.
"""

from src.benchmark._baselines import ewma_normal
from src.benchmark._build import (
    CachePart,
    build_cache,
    candle_windows,
    label_quantiles,
    saved_paths,
    with_last_close,
)
from src.benchmark._cache import (
    QUANTILE_COLUMNS,
    ForecastCache,
    forecast_run,
    read_cache,
    study_cache,
)
from src.benchmark._contexts import HOUR, calendar_windows, context_ends, own_bars_only
from src.benchmark._decisions import (
    CELL,
    DECISION,
    RELEASED_AFTER,
    SCORED,
    cell_stream,
    cell_timing,
    cell_timings,
    complete_cells,
    decision_points,
    is_after_release,
    release_date,
    require_scored,
    scored_decisions,
    split_models,
    unscored_note,
    valid_decisions,
    valid_quantiles,
)
from src.benchmark._dispersion import interval_dispersion
from src.benchmark._names import (
    FINANCE_MODELS,
    MARKET_NAMES,
    MODEL_NAMES,
    REFERENCE_NAMES,
    dashed_rule,
    finance_last,
    paper_order,
)
from src.benchmark._power import (
    detectable_cells,
    power_tables,
    powered_threshold,
    threshold_shares,
)
from src.benchmark._scores import (
    baseline_losses,
    baseline_names,
    beats_baseline,
    best_baseline,
    decision_losses,
    full_quarters,
    null_tables,
)
from src.benchmark._settings import (
    Bootstrap,
    CacheSettings,
    NullRule,
    StudySettings,
    benchmark_settings,
    cache_settings,
    fold_names,
    null_rule,
    on_model_grid,
    read_bootstrap,
    scored_folds,
    study_settings,
)

__all__ = [
    "CELL",
    "DECISION",
    "FINANCE_MODELS",
    "HOUR",
    "MARKET_NAMES",
    "MODEL_NAMES",
    "QUANTILE_COLUMNS",
    "REFERENCE_NAMES",
    "RELEASED_AFTER",
    "SCORED",
    "Bootstrap",
    "CachePart",
    "CacheSettings",
    "ForecastCache",
    "NullRule",
    "StudySettings",
    "baseline_losses",
    "baseline_names",
    "beats_baseline",
    "benchmark_settings",
    "best_baseline",
    "build_cache",
    "cache_settings",
    "calendar_windows",
    "candle_windows",
    "cell_stream",
    "cell_timing",
    "cell_timings",
    "complete_cells",
    "context_ends",
    "dashed_rule",
    "decision_losses",
    "decision_points",
    "detectable_cells",
    "ewma_normal",
    "finance_last",
    "fold_names",
    "forecast_run",
    "full_quarters",
    "interval_dispersion",
    "is_after_release",
    "label_quantiles",
    "null_rule",
    "null_tables",
    "on_model_grid",
    "own_bars_only",
    "paper_order",
    "power_tables",
    "powered_threshold",
    "read_bootstrap",
    "read_cache",
    "release_date",
    "require_scored",
    "saved_paths",
    "scored_decisions",
    "scored_folds",
    "split_models",
    "study_cache",
    "study_settings",
    "threshold_shares",
    "unscored_note",
    "valid_decisions",
    "valid_quantiles",
    "with_last_close",
]
