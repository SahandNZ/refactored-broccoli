import numpy as np
import pandas as pd

from src.enums import Market

BINANCE_USDM_BPS = {"maker": 2.0, "taker": 5.0}
"""Binance USD-M futures fees per side in bps; crypto trades pay one of them.

Source: https://www.binance.com/en/fee/futureFee, the regular (VIP 0) row of USD-M futures:
maker 0.0200 %, taker 0.0500 %, without the BNB discount (checked 2026-11-16)."""


def cost_per_side_bps(market: Market, *, half_spread_bps: float | None, crypto_fee: str) -> float:
    """Return the cost of one trade side in bps for an asset.

    Crypto pays the Binance USD-M futures fee (`crypto_fee` is ``"maker"`` or ``"taker"``);
    every other market crosses half its bid-ask spread, passed in by the caller: half the
    median Dukascopy spread of the asset, or for US stocks without bid and ask (Alpaca) an
    assumed half spread.

    Raises
    ------
    ValueError
        If a non-crypto asset has no spread or the fee name is unknown.
    """
    if market is Market.CRYPTO:
        if crypto_fee not in BINANCE_USDM_BPS:
            raise ValueError(f"crypto_fee must be maker or taker, got {crypto_fee!r}")
        return BINANCE_USDM_BPS[crypto_fee]
    if half_spread_bps is None or not np.isfinite(half_spread_bps):
        raise ValueError(f"a {market.value} asset needs its half spread in bps")
    return float(half_spread_bps)


def strategy_returns(
    signal: pd.DataFrame, realised: pd.DataFrame, costs_bps: pd.Series
) -> pd.DataFrame:
    """Gross and net returns of one long/short position per asset that follows the forecasts.

    The position is the sign of the signal (0 where it is missing). The return of a time is
    ``position * realised - |position change| * cost``: opening or closing a position pays
    one side, a flip from long to short two.

    Parameters
    ----------
    signal : pd.DataFrame
        Forecasts, shape ``(n_times, n_assets)``; NaN where an asset holds nothing.
    realised : pd.DataFrame
        The return each position earns at each time, same shape; NaN counts as 0.
    costs_bps : pd.Series
        Cost per side in bps for each asset (`cost_per_side_bps`).

    Returns
    -------
    pd.DataFrame
        Columns ``(asset, "gross")``, ``(asset, "net")``, ``(asset, "turnover")`` as a
        MultiIndex, shape ``(n_times, 3 * n_assets)``.
    """
    position = np.sign(signal.fillna(0.0).to_numpy())
    turnover = np.abs(np.diff(position, axis=0, prepend=0.0))
    gross = position * realised.fillna(0.0).to_numpy()
    net = gross - turnover * costs_bps[signal.columns].to_numpy() / 1e4
    pieces = {"gross": gross, "net": net, "turnover": turnover}
    frames = {
        name: pd.DataFrame(values, index=signal.index, columns=signal.columns)
        for name, values in pieces.items()
    }
    return pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)


def portfolio_returns(strategy: pd.DataFrame, *, kind: str) -> pd.Series:
    """Equal-weight average over assets of their ``"gross"``, ``"net"`` or ``"turnover"``."""
    returns = strategy.loc[:, strategy.columns.get_level_values(1) == kind]
    return returns.mean(axis=1)


def sharpe_ratio(returns: pd.Series, *, periods_per_year: float) -> float:
    """Annualised Sharpe ratio (mean over standard deviation, times sqrt of periods per year)."""
    deviation = float(returns.std(ddof=1))
    return float(returns.mean()) / deviation * np.sqrt(periods_per_year) if deviation > 0 else 0.0
