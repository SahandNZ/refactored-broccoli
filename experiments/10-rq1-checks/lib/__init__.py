"""Exploratory checks of rq1 (experiments/10-rq1-checks).

Q1 scores every model on the decision points of the grid model (Kronos), so the hourly models
and Kronos are compared on the same decisions; it also counts how many of those decisions
are a market's first bar after a closure, and the gain of each hourly model over Kronos on
the same decision points. Modules import each other relatively, so the
package works under any module name.
"""

from ._grid import GridCache, grid_cache, kronos_gaps, reopening_shares

__all__ = ["GridCache", "grid_cache", "kronos_gaps", "reopening_shares"]
