from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.enums import Timeframe
from src.utils import to_nanoseconds

FEATURES = ("log_return", "log_range", "log_body", "log_volume", "log_age_hours")
"""Features of every bar, in order (see `bar_features`)."""


def bar_features(bars: pd.DataFrame) -> np.ndarray:
    """Turn one instrument's bars into model features.

    Prices are the mid prices for bid/ask sources and the trade prices otherwise; the first bar
    has a return of 0.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars of one timeframe with OHLC columns (``mid_*`` or plain) and optionally a volume
        column (``volume`` or ``bid_volume``).

    Returns
    -------
    np.ndarray
        Shape ``(n_bars, 4)``: log return of the close to the previous bar's close, log of
        high over low, log of close over open, log(1 + volume). The fifth feature (the bar's
        age at decision time) is added when windows are cut.
    """
    prefix = "mid_" if "mid_close" in bars.columns else ""
    close = bars[f"{prefix}close"].to_numpy(dtype=float)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    span = np.log(
        bars[f"{prefix}high"].to_numpy(dtype=float) / bars[f"{prefix}low"].to_numpy(dtype=float)
    )
    body = np.log(close / bars[f"{prefix}open"].to_numpy(dtype=float))
    volume_column = "volume" if "volume" in bars.columns else "bid_volume"
    volume = (
        bars[volume_column].to_numpy(dtype=float)
        if volume_column in bars.columns
        else np.zeros(len(bars))
    )
    return np.stack([returns, span, body, np.log1p(np.maximum(volume, 0))], axis=1)


@dataclass(frozen=True)
class PanelWindows:
    """Windows of the last bars of every (asset, timeframe) at any decision time.

    Each asset keeps its own bars (a closed market simply has none), so a window is always the
    asset's last `lookback` closed bars before the decision time, plus how old each bar is. A
    bar is used only after it has closed: its open time plus its length must be at or before
    the decision time (no look-ahead).

    Attributes
    ----------
    assets : tuple of str
        Asset order of the panel.
    timeframes : tuple of Timeframe
        Timeframe order.
    lookback : int
        Bars per window.
    max_age_hours : float
        An asset whose newest 1-hour-or-longer bar is older than this at a decision time is
        masked (e.g. a market closed for a long holiday).
    closes : dict
        ``(asset, timeframe) -> int64 nanosecond close times``, shape ``(n_bars,)``.
    features : dict
        ``(asset, timeframe) -> features``, shape ``(n_bars, 4)``.
    """

    assets: tuple[str, ...]
    timeframes: tuple[Timeframe, ...]
    lookback: int
    max_age_hours: float
    closes: dict[tuple[str, Timeframe], np.ndarray]
    features: dict[tuple[str, Timeframe], np.ndarray]

    @classmethod
    def build(
        cls,
        bars: dict[str, dict[Timeframe, pd.DataFrame]],
        *,
        lookback: int,
        max_age_hours: float,
    ) -> "PanelWindows":
        """Prepare the windows from each asset's bars per timeframe (UTC index of open times)."""
        assets = tuple(bars)
        timeframes = tuple(next(iter(bars.values())))
        closes, features = {}, {}
        for asset in assets:
            for timeframe in timeframes:
                frame = bars[asset][timeframe]
                close_times = to_nanoseconds(frame.index + timeframe.timedelta)
                closes[asset, timeframe] = close_times
                features[asset, timeframe] = bar_features(frame).astype(np.float32)
        return cls(assets, timeframes, lookback, max_age_hours, closes, features)

    def windows(self, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Cut the windows of every asset and timeframe at each decision time.

        Parameters
        ----------
        times : pd.DatetimeIndex
            Decision times (UTC), shape ``(batch,)``.

        Returns
        -------
        tuple of np.ndarray
            ``x`` of shape ``(batch, n_assets, n_timeframes, lookback, 5)`` (features of
            `FEATURES`, zeros where masked) and ``mask`` of shape ``(batch, n_assets)``, True
            where the asset has a full, fresh window.
        """
        now = to_nanoseconds(times)
        shape = (len(now), len(self.assets), len(self.timeframes), self.lookback, len(FEATURES))
        x = np.zeros(shape, dtype=np.float32)
        mask = np.ones((len(now), len(self.assets)), dtype=bool)
        steps = np.arange(-self.lookback + 1, 1)
        for a, asset in enumerate(self.assets):
            for t, timeframe in enumerate(self.timeframes):
                close_times = self.closes[asset, timeframe]
                if len(close_times) == 0:
                    mask[:, a] = False
                    continue
                last = np.searchsorted(close_times, now, side="right") - 1
                full = last >= self.lookback - 1
                rows = np.clip(last[:, None] + steps[None, :], 0, len(close_times) - 1)
                age_hours = (now[:, None] - close_times[rows]) / 3.6e12
                x[:, a, t, :, :4] = self.features[asset, timeframe][rows]
                x[:, a, t, :, 4] = np.log1p(np.maximum(age_hours, 0))
                fresh = age_hours[:, -1] <= max(
                    self.max_age_hours, timeframe.timedelta / pd.Timedelta(hours=1)
                )
                mask[:, a] &= full & fresh
        x[~mask] = 0.0
        return x, mask
