import numpy as np
import pandas as pd

from src.data import resample_bars
from src.enums import Timeframe


def _bars(seed: int) -> pd.DataFrame:
    index = pd.date_range("2026-11-09", periods=4 * 96, freq="15min", tz="UTC", name="open_time")
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, len(index)).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
            "n_bars": 1,
        },
        index=index,
    )


def test_changing_the_future_never_changes_past_bars():
    """Leakage test: bars that close by time t only depend on data before t."""
    original = _bars(0)
    cut = pd.Timestamp("2026-11-10 13:00", tz="UTC")
    changed = original.copy()
    changed.loc[changed.index >= cut, ["open", "high", "low", "close"]] *= 1.5
    for timeframe in Timeframe:
        before = resample_bars(original, timeframe)
        after = resample_bars(changed, timeframe)
        closed = before.index + timeframe.timedelta <= cut
        pd.testing.assert_frame_equal(before[closed], after[closed])
