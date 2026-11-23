"""Centre and scale of the cached forecasts (experiments/04-rq2-center-vs-scale).

The decision rule of rq2 (`centre_scale_rule`), the centre and scale scores of every model per
market, horizon and window (`centre_scale_tables`), the verdict (`centre_scale_verdict`) and
the table of the figure (`figure_table`; `src.figures.draw_centre_scale` draws it). The
coverage interval (`ratio_interval`) and the per-cell seed (`cell_seed`) are
exported for their tests.
Modules import each other relatively, so the package works under any module name.
"""

from ._bootstrap import ratio_interval
from ._centre_scale import cell_seed, centre_scale_tables, centre_scale_verdict
from ._figure import MARKET_LABELS, figure_table
from ._settings import CentreScaleRule, centre_scale_rule

__all__ = [
    "MARKET_LABELS",
    "CentreScaleRule",
    "cell_seed",
    "centre_scale_rule",
    "centre_scale_tables",
    "centre_scale_verdict",
    "figure_table",
    "ratio_interval",
]
