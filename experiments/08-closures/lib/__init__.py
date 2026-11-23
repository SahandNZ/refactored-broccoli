"""The code of study 08-closures: forecasts after a reopening in every representation of closures.

`reopening_decisions`, `cut_weekends` and `closure_contexts` build the decisions after each
reopening and their contexts in the three representations and the benchmark input.
`closure_settings` and `benchmark_assets` read the study's options and assets;
`closure_losses` forecasts and scores every decision, `closure_tests` and `closure_verdict`
test forward-filled and missing against trading hours only for each model (Q1),
`best_representations` sets each model's best on the closed markets next to its best on
crypto with its weekends cut (Q2), and `rq5_table_body` writes the rows of the paper's table.
Modules import each other relatively, so the package works under any module name.
"""

from ._closure_contexts import (
    ARMS,
    BENCHMARK_INPUT,
    FORWARD_FILLED,
    MISSING,
    REPRESENTATIONS,
    TRADING_HOURS,
    ClosureContexts,
    closure_contexts,
    cut_weekends,
    reopening_decisions,
)
from ._closures import (
    ALL,
    BASELINE,
    CLOSED,
    CONTROL,
    ClosureSettings,
    analysis_settings,
    benchmark_assets,
    closure_day_means,
    closure_losses,
    closure_settings,
    closure_types,
)
from ._table import rq5_table_body
from ._verdict import (
    ALTERNATIVES,
    LATE,
    ON_TIME,
    REFERENCE,
    best_representations,
    block_length,
    closure_tests,
    closure_verdict,
    control_answer,
    family_answer,
    skill_interval,
    sorted_robustness,
)

__all__ = [
    "ALL",
    "ALTERNATIVES",
    "ARMS",
    "BASELINE",
    "BENCHMARK_INPUT",
    "CLOSED",
    "CONTROL",
    "FORWARD_FILLED",
    "LATE",
    "MISSING",
    "ON_TIME",
    "REFERENCE",
    "REPRESENTATIONS",
    "TRADING_HOURS",
    "ClosureContexts",
    "ClosureSettings",
    "analysis_settings",
    "benchmark_assets",
    "best_representations",
    "block_length",
    "closure_contexts",
    "closure_day_means",
    "closure_losses",
    "closure_settings",
    "closure_tests",
    "closure_types",
    "closure_verdict",
    "control_answer",
    "cut_weekends",
    "family_answer",
    "reopening_decisions",
    "rq5_table_body",
    "skill_interval",
    "sorted_robustness",
]
