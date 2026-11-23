import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.enums import Market, Source, Timeframe


@dataclass(frozen=True, slots=True)
class Instrument:
    """One instrument to download.

    Attributes
    ----------
    name : str
        Upper-case name used in paths, e.g. ``"EURUSD"`` or ``"BTCUSDT"``.
    source : Source
        Where its bars come from.
    market : Market
        The market it belongs to.
    code : str
        The source's own code, e.g. ``"eurusd"`` (Dukascopy) or ``"BTCUSDT"`` (Binance).
    timeframe : Timeframe
        The finest bar downloaded and stored; coarser ones are built from it (15m unless its
        group sets ``timeframe``).
    has_reference : bool
        Whether a stock has a Yahoo reference (daily closes and splits) for its quality check;
        False only for the stocks a group lists under ``no_reference``.
    calendar : str or None
        The ``exchange_calendars`` name of the exchange whose sessions the coverage is measured
        on: a stock's from its ticker suffix (``[stock_calendars]``), another instrument's from
        its group's ``calendars`` table; None for weekdays (every day for crypto).
    """

    name: str
    source: Source
    market: Market
    code: str
    timeframe: Timeframe = Timeframe.M15
    has_reference: bool = True
    calendar: str | None = None


@dataclass(frozen=True, slots=True)
class QualityLimits:
    """Thresholds an instrument must meet to stay a candidate (see ``check_quality``).

    Attributes
    ----------
    min_coverage : float
        Least share of expected trading days (exchange sessions for stocks) with a bar.
    max_gap_days : float
        Longest allowed time between two bars, in days.
    max_negative_spread_share : float
        Largest share of bars with ask below bid.
    outlier_sigma : float
        A 15m log return above this many robust sigmas counts as an outlier (reported).
    max_abs_return : float
        Largest allowed absolute 15m log return outside crypto (catches unadjusted splits).
    yahoo_return_median_bps, yahoo_return_p99_bps : float
        Largest median and 99th-percentile absolute difference, in bps, between our and
        Yahoo's daily close-to-close log returns (stocks).
    max_flat_share : float
        Largest share of bars whose high equals their low (catches frozen filler bars).
    """

    min_coverage: float
    max_gap_days: float
    max_negative_spread_share: float
    outlier_sigma: float
    max_abs_return: float
    yahoo_return_median_bps: float
    yahoo_return_p99_bps: float
    max_flat_share: float


@dataclass(frozen=True, slots=True)
class DataConfig:
    """What to download and how.

    Attributes
    ----------
    start, end : str
        UTC dates; bars opening in ``[start, end)`` are kept.
    workers : dict of str to int
        Download threads per source (``"binance"``, ``"dukascopy"``, ``"alpaca"``,
        ``"yahoo"``) and processes for building (``"build"``).
    quality : QualityLimits
        Pass thresholds of the quality checks.
    data_dir : Path
        The lab's ``data/`` folder.
    instruments : tuple of Instrument
        Every instrument to download.
    """

    start: str
    end: str
    workers: dict[str, int]
    quality: QualityLimits
    data_dir: Path
    instruments: tuple[Instrument, ...]


def load_data_config(path: Path, *, data_dir: Path, groups: list[str] | None = None) -> DataConfig:
    """Read a data configuration file.

    Parameters
    ----------
    path : Path
        TOML file with ``[period]`` (start, end), ``[download.workers]`` (per source),
        ``[quality]`` (the `QualityLimits`), ``[stock_calendars]`` (Yahoo ticker suffix =
        exchange calendar; the key ``""`` is a ticker without a suffix) and one
        ``[groups.<name>]`` table per group of instruments with ``market``, ``source`` and
        ``instruments`` (name = source code). Optional in a group: ``timeframe`` (without it
        15m), ``no_reference`` (stocks checked without a Yahoo reference; without it none) and,
        outside stock groups, ``calendars`` (instrument name = exchange calendar; an instrument
        without one is measured on weekdays, or every day for crypto).
    data_dir : Path
        The lab's ``data/`` folder.
    groups : list of str, optional
        Keep only these groups, in the file's order (all groups if None).

    Returns
    -------
    DataConfig
        The parsed configuration.

    Raises
    ------
    KeyError
        If a required value is missing, a group in `groups` is not in the file, a
        ``no_reference`` or ``calendars`` name is not an instrument of its group, a stock group
        has a ``calendars`` table, or a stock's ticker suffix is not in ``[stock_calendars]``.
    """
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if groups is not None:
        unknown = set(groups) - set(raw["groups"])
        if unknown:
            raise KeyError(f"unknown groups: {sorted(unknown)}")
    instruments = []
    for group, table in raw["groups"].items():
        if groups is not None and group not in groups:
            continue
        source, market = Source(table["source"]), Market(table["market"])
        timeframe = Timeframe(table.get("timeframe", "15m"))
        no_reference = set(table.get("no_reference", []))
        calendars: dict[str, str] = table.get("calendars", {})
        unknown_calendars = set(calendars) - set(table["instruments"])
        if unknown_calendars:
            raise KeyError(f"calendars names are not in group {group}: {sorted(unknown_calendars)}")
        unknown_names = no_reference - set(table["instruments"])
        if unknown_names:
            raise KeyError(f"no_reference names are not in group {group}: {sorted(unknown_names)}")
        if market is Market.STOCK:
            if calendars:
                raise KeyError(f"stock group {group} takes its calendars from [stock_calendars]")
            calendars = _stock_calendars(list(table["instruments"]), raw["stock_calendars"])
        for name, code in table["instruments"].items():
            has_reference = name not in no_reference
            instrument = Instrument(
                name, source, market, code, timeframe, has_reference, calendars.get(name)
            )
            instruments.append(instrument)
    return DataConfig(
        start=raw["period"]["start"],
        end=raw["period"]["end"],
        workers=dict(raw["download"]["workers"]),
        quality=QualityLimits(**raw["quality"]),
        data_dir=data_dir,
        instruments=tuple(instruments),
    )


def _stock_calendars(names: list[str], by_suffix: dict[str, str]) -> dict[str, str]:
    """Return the exchange calendar of each stock from its Yahoo ticker suffix.

    ``"SAP.DE"`` has the suffix ``"DE"``; a ticker without a dot (``"AAPL"``) has ``""``.

    Raises
    ------
    KeyError
        If a suffix is not in `by_suffix`.
    """
    calendars: dict[str, str] = {}
    for name in names:
        suffix = name.rpartition(".")[2] if "." in name else ""
        if suffix not in by_suffix:
            raise KeyError(f"stock {name}: suffix {suffix!r} is not in [stock_calendars]")
        calendars[name] = by_suffix[suffix]
    return calendars
