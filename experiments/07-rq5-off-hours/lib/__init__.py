"""The code of study 07-rq5-off-hours: weekend crypto and the reopening of closed markets.

`off_hours_settings` reads the study's options and `universe_assets` its two asset sets;
`read_weekends` finds every weekend of every closed-market asset with the crypto basket over
the same closed hours. `reopening_forecasts` forecasts the first bar after each reopening in
five arms and `reopening_tables` scores and tests them (Q1), `covariate_use` measures how much
of the crypto move a model passes on; `gap_tables` regresses the reopening gap on Friday's
information and the weekend crypto move (Q2).
Modules import each other relatively, so the package works under any module name.
"""

from ._comparisons import (
    COMPARISONS,
    COVARIATE_TESTS,
    covariate_use,
    label_overlap,
    reopening_tables,
)
from ._gaps import gap_statistics, gap_tables
from ._reopening import (
    ARMS,
    MODEL_ARMS,
    WEEKEND_COLUMNS,
    ReopeningContexts,
    corrected_quantiles,
    fit_correction,
    model_free_quantiles,
    reopening_contexts,
    reopening_forecasts,
)
from ._settings import (
    AssetSets,
    OffHoursSettings,
    off_hours_settings,
    universe_assets,
)
from ._weekends import Weekends, basket_return, read_weekends, weekend_closures, weekend_rows

__all__ = [
    "ARMS",
    "COMPARISONS",
    "COVARIATE_TESTS",
    "MODEL_ARMS",
    "WEEKEND_COLUMNS",
    "AssetSets",
    "OffHoursSettings",
    "ReopeningContexts",
    "Weekends",
    "basket_return",
    "corrected_quantiles",
    "covariate_use",
    "fit_correction",
    "gap_statistics",
    "gap_tables",
    "label_overlap",
    "model_free_quantiles",
    "off_hours_settings",
    "read_weekends",
    "reopening_contexts",
    "reopening_forecasts",
    "reopening_tables",
    "universe_assets",
    "weekend_closures",
    "weekend_rows",
]
