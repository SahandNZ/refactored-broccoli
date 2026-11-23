"""The bootstrap size check of 09-mde-threshold (experiments/09-mde-threshold).

Modules import each other relatively, so the package works under any module name.
"""

from ._size import bootstrap_size_table

__all__ = ["bootstrap_size_table"]
