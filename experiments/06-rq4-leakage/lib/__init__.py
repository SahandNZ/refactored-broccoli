"""Leakage of the zero-shot models of the benchmark (experiments/06-rq4-leakage).

Each model's skill over the best baseline before and after its release: the periods of its
decisions, the skill tables, the tests per cell (the difference in differences and the
checks) and the decision rule. Modules import each other relatively, so the package works
under any module name.
"""

from ._cache import JoinedCache, join_caches, leakage_cache, not_fitted, require_windows
from ._compare import TESTS, compare_periods, leakage_verdict
from ._figure import (
    FIGURE_COLUMNS,
    LIMIT_MARGIN,
    MARKET_LABELS,
    figure_families,
    figure_limit,
    figure_table,
)
from ._leakage import (
    LeakageOptions,
    leakage_grid,
    leakage_options,
    leakage_periods,
    leakage_tables,
)

__all__ = [
    "FIGURE_COLUMNS",
    "LIMIT_MARGIN",
    "MARKET_LABELS",
    "TESTS",
    "JoinedCache",
    "LeakageOptions",
    "compare_periods",
    "figure_families",
    "figure_limit",
    "figure_table",
    "join_caches",
    "leakage_cache",
    "leakage_grid",
    "leakage_options",
    "leakage_periods",
    "leakage_tables",
    "leakage_verdict",
    "not_fitted",
    "require_windows",
]
