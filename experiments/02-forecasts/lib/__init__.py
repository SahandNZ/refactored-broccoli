"""The validity check of the forecast cache (experiments/02-forecasts, Q1).

Modules import each other relatively, so the package works under any module name.
"""

from ._validity import validity_table

__all__ = ["validity_table"]
