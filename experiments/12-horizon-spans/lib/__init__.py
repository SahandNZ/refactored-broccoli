"""The calendar span of the benchmark's horizons (experiments/12-horizon-spans).

`horizon_spans` measures, per market and horizon, how many calendar hours lie between a
decision and the close of its label's last own bar. Modules import each other relatively, so
the package works under any module name.
"""

from ._spans import horizon_spans

__all__ = ["horizon_spans"]
