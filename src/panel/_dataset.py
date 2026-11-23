import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data._read import read_bars
from src.data._resample import resample_bars
from src.enums import Market, Timeframe
from src.panel._inputs import PanelWindows
from src.panel._labels import vol_scaled_returns
from src.utils import in_session, last_at_or_before, to_utc

logger = logging.getLogger(__name__)


def is_us_stock(symbol: str, market: Market) -> bool:
    """Whether `symbol` is a US stock (a stock whose Yahoo ticker has no exchange suffix)."""
    return market is Market.STOCK and "." not in symbol


def load_panel_bars(
    data_dir: Path, markets: dict[str, Market], timeframes: tuple[Timeframe, ...]
) -> dict[str, dict[Timeframe, pd.DataFrame]]:
    """Read every symbol's bars at every timeframe, US stocks cut to their regular session.

    US stocks (every source) keep only bars opening in 09:30-16:00 New York time; their
    coarser bars are rebuilt from those 15m bars, so pre- and after-market trades never enter.

    Parameters
    ----------
    data_dir : Path
        The lab's ``data/`` folder.
    markets : dict of str to Market
        The symbols to read and their markets (in panel order).
    timeframes : tuple of Timeframe
        The timeframes to read.

    Returns
    -------
    dict
        ``symbol -> timeframe -> bars`` (UTC index of open times).
    """
    panel: dict[str, dict[Timeframe, pd.DataFrame]] = {}
    for symbol, market in markets.items():
        if is_us_stock(symbol, market):
            base = read_bars(data_dir, market, symbol, Timeframe.M15)
            base = base[
                in_session(base.index, timezone="America/New_York", opens="09:30", closes="16:00")
            ]
            panel[symbol] = {
                tf: base if tf is Timeframe.M15 else resample_bars(base, tf) for tf in timeframes
            }
        else:
            panel[symbol] = {tf: read_bars(data_dir, market, symbol, tf) for tf in timeframes}
    return panel


def hourly_closes(bars: dict[str, dict[Timeframe, pd.DataFrame]]) -> dict[str, pd.Series]:
    """Return each symbol's 1h closes (mid for bid/ask sources) indexed by the bars' close times."""
    closes = {}
    for symbol, frames in bars.items():
        frame = frames[Timeframe.H1]
        price = frame["mid_close"] if "mid_close" in frame.columns else frame["close"]
        closes[symbol] = price.set_axis(frame.index + Timeframe.H1.timedelta)
    return closes


@dataclass(frozen=True, slots=True)
class PanelSettings:
    """How windows and labels are built.

    Attributes
    ----------
    lookback : int
        Bars per window.
    horizon_hours : int
        Forecast horizon of the labels.
    span_bars : int
        EWMA span (hourly bars) of the label's volatility.
    max_age_hours : float
        Assets whose newest bar is older are masked (see `PanelWindows`).
    """

    lookback: int
    horizon_hours: int
    span_bars: int
    max_age_hours: float


@dataclass(frozen=True)
class PanelData:
    """Inputs and labels of a panel, ready to be cut into batches.

    Attributes
    ----------
    windows : PanelWindows
        The windows of every (asset, timeframe).
    times : pd.DatetimeIndex
        Hourly decision times covered by `labels`.
    labels : np.ndarray
        Vol-scaled forward returns, shape ``(n_times, n_assets)``.
    returns : np.ndarray
        Raw forward log returns over the same horizon, shape ``(n_times, n_assets)`` (for
        trading results).
    """

    windows: PanelWindows
    times: pd.DatetimeIndex
    labels: np.ndarray
    returns: np.ndarray

    @classmethod
    def build(
        cls,
        bars: dict[str, dict[Timeframe, pd.DataFrame]],
        *,
        start: str,
        end: str,
        settings: PanelSettings,
    ) -> "PanelData":
        """Prepare windows and labels for every hour in ``[start, end)``."""
        windows = PanelWindows.build(
            bars, lookback=settings.lookback, max_age_hours=settings.max_age_hours
        )
        horizon_hours, span_bars = settings.horizon_hours, settings.span_bars
        times = pd.date_range(to_utc(start), to_utc(end), freq="1h", inclusive="left")
        closes = hourly_closes(bars)
        labels = np.stack(
            [
                vol_scaled_returns(
                    closes[s], times, horizon_hours=horizon_hours, span_bars=span_bars
                )
                for s in windows.assets
            ],
            axis=1,
        )
        later = times + pd.Timedelta(hours=horizon_hours)
        returns = np.stack(
            [_forward_log_return(closes[s], times, later) for s in windows.assets], axis=1
        )
        return cls(windows, times, labels.astype(np.float32), returns.astype(np.float32))

    def batches(
        self, selection: np.ndarray, *, batch_size: int, shuffle_seed: int | None
    ) -> list[tuple[torch.Tensor, ...]]:
        """Cut the chosen decision times into batches ``(x, mask, labels)``.

        Parameters
        ----------
        selection : np.ndarray
            Integer positions in `times`.
        batch_size : int
            Decisions per batch.
        shuffle_seed : int or None
            Shuffle the order with this seed (training); None keeps time order.
        """
        order = selection.copy()
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        result = []
        for first in range(0, len(order), batch_size):
            chosen = np.sort(order[first : first + batch_size])
            x, mask = self.windows.windows(self.times[chosen])
            labels = self.labels[chosen]
            mask = mask & np.isfinite(labels) if shuffle_seed is not None else mask
            result.append((torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels)))
        return result


def forward_returns(
    closes: dict[str, pd.Series], assets: tuple[str, ...], times: pd.DatetimeIndex, *, hours: int
) -> np.ndarray:
    """Log return of each asset from the last close by ``t`` to the last close by ``t + hours``.

    Returns
    -------
    np.ndarray
        Shape ``(n_times, n_assets)``; NaN when no new bar closes in between.
    """
    later = times + pd.Timedelta(hours=hours)
    return np.stack([_forward_log_return(closes[asset], times, later) for asset in assets], axis=1)


def _forward_log_return(
    closes: pd.Series, times: pd.DatetimeIndex, later: pd.DatetimeIndex
) -> np.ndarray:
    index = pd.DatetimeIndex(closes.index)
    now = last_at_or_before(index, times)
    after = last_at_or_before(index, later)
    values = np.log(closes.to_numpy(dtype=float))
    result = np.full(len(times), np.nan)
    valid = (now >= 0) & (after > now)
    result[valid] = values[after[valid]] - values[now[valid]]
    return result
