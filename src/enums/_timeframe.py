from enum import Enum

import pandas as pd


class Timeframe(Enum):
    """Bar length. 15m is the finest bar stored; every coarser one is built from it.

    The value is the short label used in file names and configurations: parse it with
    ``Timeframe("1h")`` and write it with ``Timeframe.H1.value``. A plain ``Enum``, so
    ``Timeframe.H1 == "1h"`` is False.
    """

    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"

    @property
    def pandas_freq(self) -> str:
        """The pandas frequency of one bar, e.g. ``"4h"``.

        ``1w`` is ``"7D"``; ``src.utils.bar_grid`` anchors it on a Monday.
        """
        return _PANDAS_FREQ[self]

    @property
    def timedelta(self) -> pd.Timedelta:
        """The length of one bar."""
        return pd.Timedelta(self.pandas_freq)

    @property
    def minutes(self) -> int:
        """The length of one bar in minutes."""
        return int(self.timedelta.total_seconds() // 60)


_PANDAS_FREQ = {
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.H12: "12h",
    Timeframe.D1: "1D",
    Timeframe.D3: "3D",
    Timeframe.W1: "7D",
}
