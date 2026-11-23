from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data import DataConfig, Instrument, check_quality, read_bars, resample_bars
from src.enums import Market, Timeframe
from src.utils import days_after, in_session, in_window, is_weekend

from ._settings import UniverseSettings


@dataclass(frozen=True, slots=True)
class Measurement:
    """What the selection needs of one candidate, measured on its own bars.

    Attributes
    ----------
    quality : dict
        `src.data.check_quality` on the bars of the selection window (empty without bars).
    first_bar : pd.Timestamp or None
        The first bar in the selection window; None without bars.
    dollar_volume : float
        Median daily traded value in the selection window (crypto, US stocks); NaN otherwise.
    spread_bps : float
        Median relative spread of the 1h bars in the selection window, in bps (bid/ask
        sources); NaN otherwise.
    closes : pd.Series
        Daily closes (mid for bid/ask sources) per UTC day, from the selection start to the
        held-out end; a day without a bar has no close.
    """

    quality: dict[str, object]
    first_bar: pd.Timestamp | None
    dollar_volume: float
    spread_bps: float
    closes: pd.Series


def selection_pool(symbol: str, market: Market) -> str:
    """Name the pool `symbol` is picked in: its market, or ``stock_us`` / ``stock_non_us``."""
    if market is not Market.STOCK:
        return market.value
    if "." in symbol:
        return "stock_non_us"
    return "stock_us"


def measure_candidate(
    data: DataConfig, instrument: Instrument, settings: UniverseSettings
) -> Measurement:
    """Read one candidate's base bars and measure its quality, liquidity and daily closes.

    Only bars in ``[settings.start, settings.holdout_end)`` are read; the quality and liquidity
    use only the selection window ``[start, end)``. A US stock keeps only its regular session,
    09:30-16:00 America/New_York on weekdays, as in the panel (Alpaca also has pre- and
    after-market and some Saturday bars). The quality check is the lab's own
    (`src.data.check_quality` with the limits of ``configs/data.toml``): the exchange sessions
    of stocks, indices and softs (calendars from ``configs/data.toml``), weekdays for forex,
    metals and energy, every day for crypto; coverage runs from the first bar to the window's
    end, so bars that stop early count as missing days. A
    stock with a Yahoo reference is compared with the reference closes of the same window.

    Parameters
    ----------
    data : DataConfig
        The data configuration (``data_dir`` and the quality limits).
    instrument : Instrument
        The candidate.
    settings : UniverseSettings
        The study's settings.

    Returns
    -------
    Measurement
        The candidate's measurements.

    Raises
    ------
    ValueError
        If the bars have neither a traded value (``amount``, or ``vwap`` and ``volume``) nor a
        bid/ask spread.
    """
    bars = read_bars(data.data_dir, instrument.market, instrument.name, instrument.timeframe)
    bars = bars[
        in_window(pd.DatetimeIndex(bars.index), start=settings.start, end=settings.holdout_end)
    ]
    if selection_pool(instrument.name, instrument.market) == "stock_us":
        times = pd.DatetimeIndex(bars.index)
        session = in_session(times, timezone="America/New_York", opens="09:30", closes="16:00")
        bars = bars[session & ~is_weekend(times)]
    window = bars[in_window(pd.DatetimeIndex(bars.index), start=settings.start, end=settings.end)]
    if window.empty:
        return Measurement({}, None, float("nan"), float("nan"), _daily_closes(bars))
    reference = _yahoo_closes(data.data_dir, instrument, start=settings.start, end=settings.end)
    quality = check_quality(
        window, instrument, limits=data.quality, reference=reference, end=settings.end
    )
    dollar_volume, spread_bps = _liquidity(window, instrument.name)
    return Measurement(quality, window.index[0], dollar_volume, spread_bps, _daily_closes(bars))


def _daily_closes(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    daily = resample_bars(bars, Timeframe.D1)
    return daily["mid_close"] if "mid_close" in daily.columns else daily["close"]


def _liquidity(window: pd.DataFrame, symbol: str) -> tuple[float, float]:
    """Return the median daily traded value and the median 1h spread in bps (one is NaN)."""
    if "bid_close" in window.columns and "ask_close" in window.columns:
        hourly = resample_bars(window, Timeframe.H1)
        return float("nan"), float((hourly["spread"] / hourly["mid_close"]).median() * 1e4)
    if "amount" in window.columns:
        traded = window["amount"]
    elif "vwap" in window.columns and "volume" in window.columns:
        traded = window["vwap"] * window["volume"]
    else:
        raise ValueError(f"{symbol} has no amount, vwap and volume, or bid and ask columns")
    daily = resample_bars(traded.to_frame("traded_value"), Timeframe.D1)
    return float(daily["traded_value"].median()), float("nan")


def _yahoo_closes(
    data_dir: Path, instrument: Instrument, *, start: str, end: str
) -> pd.Series | None:
    """Return a stock's Yahoo reference closes in ``[start, end)``; None without a reference.

    The reference lives at ``data/stock/<symbol>/reference/yahoo/1d.pq`` (written by
    `src.data.download_all`), indexed by the exchange's local date.
    """
    if instrument.market is not Market.STOCK or not instrument.has_reference:
        return None
    path = data_dir / instrument.market.value / instrument.name / "reference" / "yahoo" / "1d.pq"
    closes = pd.read_parquet(path)["close"]
    return closes[in_window(pd.DatetimeIndex(closes.index), start=start, end=end)]


def data_filter(
    markets: pd.Series, measured: dict[str, Measurement], settings: UniverseSettings
) -> tuple[list[str], dict[str, str]]:
    """Apply the data filter of 01-universe.

    A forex pair with a currency of ``settings.pegged_currencies`` is dropped first. Then a
    candidate stays when it has bars in the selection window, its first bar is on time and its
    quality check over the window passed. On time: by the end of the day
    ``settings.listed_by`` names for its market, else before ``settings.start`` plus
    ``settings.start_tolerance_days`` days (most markets first open on 2 January).

    Parameters
    ----------
    markets : pd.Series
        The `Market` of each candidate.
    measured : dict of str to Measurement
        The measurements of each candidate (`measure_candidate`).
    settings : UniverseSettings
        The study's settings.

    Returns
    -------
    tuple of (list of str, dict of str to str)
        The kept symbols and the reason each other symbol was dropped.
    """
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for symbol, measurement in measured.items():
        market: Market = markets[symbol]
        if market.value in settings.listed_by:
            latest = days_after(settings.listed_by[market.value], 1)
        else:
            latest = days_after(settings.start, settings.start_tolerance_days)
        codes = (symbol[:3], symbol[3:]) if market is Market.FX else ()
        pegs = [code for code in codes if code in settings.pegged_currencies]
        if pegs:
            dropped[symbol] = f"pegged currency {', '.join(pegs)}"
        elif measurement.first_bar is None:
            dropped[symbol] = "no bar in the selection window"
        elif measurement.first_bar >= latest:
            dropped[symbol] = f"starts {measurement.first_bar}"
        elif not measurement.quality["passed"]:
            dropped[symbol] = f"quality: {measurement.quality['reason']}"
        else:
            kept.append(symbol)
    return kept, dropped


def liquidity_table(measured: dict[str, Measurement]) -> pd.DataFrame:
    """Collect the liquidity of each candidate.

    ``liquidity`` ranks the candidates (higher is more liquid): the median daily traded value
    where there is one (crypto: quote volume; US stocks: ``vwap`` times ``volume`` of the
    regular session), else minus the median 1h spread in bps.

    Parameters
    ----------
    measured : dict of str to Measurement
        The measurements of the candidates.

    Returns
    -------
    pd.DataFrame
        Indexed by symbol: ``dollar_volume``, ``spread_bps`` and ``liquidity``.
    """
    table = pd.DataFrame(
        {
            "dollar_volume": {symbol: m.dollar_volume for symbol, m in measured.items()},
            "spread_bps": {symbol: m.spread_bps for symbol, m in measured.items()},
        },
        columns=["dollar_volume", "spread_bps"],
    )
    table["liquidity"] = table["dollar_volume"].fillna(-table["spread_bps"])
    return table


def liquidity_filter(
    table: pd.DataFrame, markets: pd.Series, settings: UniverseSettings
) -> tuple[list[str], dict[str, str]]:
    """Apply the liquidity filter of 01-universe.

    Crypto stays when its median daily amount is at least ``settings.min_daily_amount``; an
    asset with a spread stays when its spread is at most ``settings.max_spread_ratio`` times
    the median spread of its market's assets with a spread. US stocks (traded value, no
    spread) always stay.

    Parameters
    ----------
    table : pd.DataFrame
        The output of `liquidity_table`.
    markets : pd.Series
        The `Market` of each symbol.
    settings : UniverseSettings
        The study's settings.

    Returns
    -------
    tuple of (list of str, dict of str to str)
        The kept symbols and the reason each other symbol was dropped.
    """
    spreads = table["spread_bps"].dropna()
    market_spread = spreads.groupby(markets[spreads.index], sort=False).median().to_dict()
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for name, row in table.iterrows():
        symbol = str(name)
        market: Market = markets[symbol]
        if market is Market.CRYPTO and row["dollar_volume"] < settings.min_daily_amount:
            dropped[symbol] = f"liquidity: median daily amount {row['dollar_volume']:.0f}"
        elif symbol in spreads.index and (
            spreads[symbol] > settings.max_spread_ratio * market_spread[market]
        ):
            dropped[symbol] = f"liquidity: spread {spreads[symbol]:.1f} bps"
        else:
            kept.append(symbol)
    return kept, dropped
