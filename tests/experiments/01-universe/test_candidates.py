import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.data import DataConfig, Instrument, load_data_config
from src.enums import Market, Source, Timeframe

type WriteBars = Callable[[Path, Instrument, pd.Series, float], None]

LIMITS = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=Path()).quality
STOCK_CALENDARS = tomllib.loads((LAB_DIR / "configs" / "data.toml").read_text())["stock_calendars"]


def _instrument(name: str, market: Market, *, has_reference: bool = False) -> Instrument:
    source = {Market.CRYPTO: Source.BINANCE, Market.STOCK: Source.ALPACA}.get(
        market, Source.DUKASCOPY
    )
    calendar = None
    if market is Market.STOCK:
        calendar = STOCK_CALENDARS[name.rpartition(".")[2] if "." in name else ""]
    return Instrument(name, source, market, name.lower(), Timeframe.H1, has_reference, calendar)


def _hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")


def _prices(times: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    returns = np.random.default_rng(seed).normal(0, 0.002, len(times))
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=times)


def _measure(
    universe: ModuleType,
    tmp_path: Path,
    write_bars: WriteBars,
    instrument: Instrument,
    times: pd.DatetimeIndex,
    options: dict[str, Any],
) -> Any:
    write_bars(tmp_path, instrument, _prices(times), 1.0)
    data = DataConfig("2020-01-01", "2026-11-01", {}, LIMITS, tmp_path, (instrument,))
    return universe.measure_candidate(data, instrument, universe.read_settings(options))


def _weekday_hours(times: pd.DatetimeIndex, *, first_hour: int, last_hour: int) -> pd.DatetimeIndex:
    return times[(times.day_of_week < 5) & (times.hour >= first_hour) & (times.hour <= last_hour)]


# the quality check of the selection window, with the exchange calendars of src.data


def test_a_market_with_sunday_evening_sessions_covers_every_weekday(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # Brent trades from Sunday 23:00 to Friday 22:00 UTC: Sundays are not open days
    times = _hours("2024-01-01", "2024-04-01")
    friday_close = (times.day_of_week == 4) & (times.hour >= 22)
    saturday = times.day_of_week == 5
    sunday_early = (times.day_of_week == 6) & (times.hour < 23)
    times = times[~(friday_close | saturday | sunday_early)]
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("BRENT", Market.COMMODITY), times, options
    )
    assert measured.quality["coverage"] == pytest.approx(1.0)
    assert measured.quality["passed"]


def test_a_regional_holiday_is_no_missing_day(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # Tokyo, April-May 2024: Golden Week (29 April, 3 and 6 May) is closed on the exchange
    # calendar; a stock without bars on those days covers every session
    times = _weekday_hours(_hours("2024-04-01", "2024-06-01"), first_hour=0, last_hour=5)
    holidays = pd.DatetimeIndex(["2024-04-29", "2024-05-03", "2024-05-06"], tz="UTC")
    times = times[~times.normalize().isin(holidays)]
    window = {**options, "start": "2024-04-01", "end": "2024-06-01", "holdout_end": "2024-06-02"}
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("7203.T", Market.STOCK), times, window
    )
    assert measured.quality["coverage"] == pytest.approx(1.0)
    assert measured.quality["passed"]


def test_a_single_stock_of_its_exchange_missing_three_weeks_fails(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # the only Helsinki candidate: its own bars cannot define the exchange's open days
    times = _weekday_hours(_hours("2024-01-02", "2024-04-01"), first_hour=8, last_hour=14)
    missing = (times >= pd.Timestamp("2024-02-05", tz="UTC")) & (
        times < pd.Timestamp("2024-02-26", tz="UTC")
    )
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("NOKIA.HE", Market.STOCK), times[~missing],
        options,
    )  # fmt: skip
    assert not measured.quality["passed"]
    assert "gap" in measured.quality["reason"]
    assert "coverage" in measured.quality["reason"]


@pytest.mark.parametrize(("gap_start", "passes"), [("2024-03-05", True), ("2024-02-05", False)])
def test_a_gap_after_the_selection_window_does_not_matter(
    universe: ModuleType,
    tmp_path: Path,
    write_bars: WriteBars,
    options: dict[str, Any],
    gap_start: str,
    passes: bool,
):
    times = _hours("2024-01-01", "2024-04-01")
    gap = (times >= pd.Timestamp(gap_start, tz="UTC")) & (
        times < pd.Timestamp(gap_start, tz="UTC") + pd.Timedelta(days=20)
    )
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("BTCUSDT", Market.CRYPTO), times[~gap], options
    )
    assert measured.quality["passed"] is passes


def test_bars_that_stop_inside_the_selection_window_count_as_missing_to_its_end(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # a delisted coin: its bars stop on 2024-02-20, the selection window ends on 2024-03-01
    times = _hours("2024-01-01", "2024-02-20")
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("GONEUSDT", Market.CRYPTO), times, options
    )
    assert measured.quality["coverage"] == pytest.approx(50 / 60, abs=1e-4)
    assert measured.quality["max_gap_days"] == pytest.approx(10.04, abs=0.01)
    assert not measured.quality["passed"]
    assert "coverage" in measured.quality["reason"]


def test_the_yahoo_check_uses_only_the_selection_window(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    instrument = _instrument("AAPL", Market.STOCK, has_reference=True)
    times = _weekday_hours(_hours("2024-01-01", "2024-04-01"), first_hour=14, last_hour=20)
    # the reference agrees with the last regular-session close of each day (20:00 UTC is
    # 15:00 in New York), except after the selection window, where it is far off
    last = _prices(times)[times.hour == 20]
    dates = pd.DatetimeIndex(last.index).date
    reference = pd.Series(last.to_numpy(), index=pd.Index(dates, name="date"))
    reference[reference.index >= pd.Timestamp("2024-03-01").date()] *= 2.0
    folder = tmp_path / "stock" / "AAPL" / "reference" / "yahoo"
    folder.mkdir(parents=True)
    reference.to_frame("close").to_parquet(folder / "1d.pq")
    measured = _measure(universe, tmp_path, write_bars, instrument, times, options)
    assert measured.quality["yahoo_return_median_bps"] == pytest.approx(0.0, abs=1e-6)
    assert measured.quality["passed"]
    later = {**options, "end": "2024-04-01", "holdout_end": "2024-04-02"}
    assert not _measure(universe, tmp_path, write_bars, instrument, times, later).quality["passed"]


def test_a_us_stock_keeps_only_its_regular_session(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # every hour of every day, Saturdays included; New York moves to summer time on 10 March
    instrument = _instrument("AAPL", Market.STOCK)
    times = _hours("2024-01-01", "2024-04-01")
    prices = _prices(times)
    write_bars(tmp_path, instrument, prices, 1.0)
    data = DataConfig("2020-01-01", "2026-11-01", {}, LIMITS, tmp_path, (instrument,))
    measured = universe.measure_candidate(data, instrument, universe.read_settings(options))
    local = times.tz_convert("America/New_York")
    session = (local.hour >= 10) & (local.hour <= 15) & (local.day_of_week < 5)
    daily_close = prices[session].groupby(times[session].normalize()).last()
    pd.testing.assert_series_equal(
        measured.closes, daily_close, check_names=False, check_freq=False
    )
    traded = (prices[session] * 1e4).groupby(times[session].normalize()).sum()
    traded = traded[traded.index < pd.Timestamp("2024-03-01", tz="UTC")]
    assert measured.dollar_volume == pytest.approx(traded.median())


def test_a_candidate_without_bars_in_the_window_is_measured_as_empty(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("LATE", Market.CRYPTO),
        _hours("2024-03-05", "2024-04-01"), options,
    )  # fmt: skip
    assert measured.first_bar is None
    assert measured.quality == {}
    assert len(measured.closes) > 0


# the data filter


def _measurement(universe: ModuleType, first_bar: str | None, *, passed: bool = True) -> Any:
    quality: dict[str, object] = {"passed": passed, "reason": "coverage 0.5 < 0.95"}
    start = None if first_bar is None else pd.Timestamp(first_bar, tz="UTC")
    return universe.Measurement(quality, start, np.nan, 1.0, pd.Series(dtype=float))


def test_the_start_rule_allows_a_few_days_but_a_listing_date_none(
    universe: ModuleType, options: dict[str, Any]
):
    measured = {
        "ON_TIME": _measurement(universe, "2024-01-02T07:00"),
        "LATE": _measurement(universe, "2024-01-08T00:00"),
        "LISTED": _measurement(universe, "2024-01-10T23:00"),
        "LISTED_LATE": _measurement(universe, "2024-01-11T00:00"),
    }
    markets = pd.Series(
        {"ON_TIME": Market.INDEX, "LATE": Market.INDEX, "LISTED": Market.CRYPTO,
         "LISTED_LATE": Market.CRYPTO}
    )  # fmt: skip
    kept, dropped = universe.data_filter(markets, measured, universe.read_settings(options))
    assert kept == ["ON_TIME", "LISTED"]
    assert dropped["LATE"].startswith("starts 2024-01-08")
    assert dropped["LISTED_LATE"].startswith("starts 2024-01-11")


def test_pegged_pairs_no_bars_and_failed_quality_are_dropped_with_their_reason(
    universe: ModuleType, options: dict[str, Any]
):
    names = ("USDHKD", "EURDKK", "USDCNH", "EMPTY", "BAD")
    measured = {name: _measurement(universe, "2024-01-01") for name in names}
    measured["EMPTY"] = _measurement(universe, None)
    measured["BAD"] = _measurement(universe, "2024-01-01", passed=False)
    markets = pd.Series(dict.fromkeys(names, Market.FX))
    kept, dropped = universe.data_filter(markets, measured, universe.read_settings(options))
    assert kept == ["USDCNH"]
    assert dropped == {
        "USDHKD": "pegged currency HKD",
        "EURDKK": "pegged currency DKK",
        "EMPTY": "no bar in the selection window",
        "BAD": "quality: coverage 0.5 < 0.95",
    }


# liquidity


def test_liquidity_is_traded_value_for_crypto_and_us_stocks_and_minus_the_spread_otherwise(
    universe: ModuleType,
):
    empty = pd.Series(dtype=float)
    table = universe.liquidity_table(
        {
            "BTCUSDT": universe.Measurement({}, None, 3.0, np.nan, empty),
            "EURUSD": universe.Measurement({}, None, np.nan, 1.0, empty),
        }
    )
    assert table["liquidity"].tolist() == pytest.approx([3.0, -1.0])


def test_the_spread_is_the_median_of_the_1h_spreads(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    times = _hours("2024-01-01", "2024-04-01")
    measured = _measure(
        universe, tmp_path, write_bars, _instrument("EURUSD", Market.FX), times, options
    )
    # the fake bars have a spread of 1e-4 of the price
    assert measured.spread_bps == pytest.approx(1.0)
    assert np.isnan(measured.dollar_volume)


def test_bars_without_a_liquidity_measure_raise(
    universe: ModuleType, tmp_path: Path, options: dict[str, Any]
):
    instrument = _instrument("ODD", Market.INDEX)
    times = _hours("2024-01-01", "2024-02-01")
    path = tmp_path / "index" / "ODD" / "ohlcv" / "1h.pq"
    path.parent.mkdir(parents=True)
    prices = _prices(times)
    pd.DataFrame(
        {"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices}
    ).rename_axis("open_time").to_parquet(path)
    data = DataConfig("2020-01-01", "2026-11-01", {}, LIMITS, tmp_path, (instrument,))
    with pytest.raises(ValueError, match="ODD"):
        universe.measure_candidate(data, instrument, universe.read_settings(options))


def test_us_stocks_pass_the_spread_filter_and_do_not_set_the_market_median(
    universe: ModuleType, options: dict[str, Any]
):
    table = pd.DataFrame(
        {"dollar_volume": [5.0, np.nan, np.nan, np.nan], "spread_bps": [np.nan, 1.0, 2.0, 9.0]},
        index=["AAPL", "SAP.DE", "BMW.DE", "WIDE.PA"],
    )
    markets = pd.Series(dict.fromkeys(table.index, Market.STOCK))
    kept, dropped = universe.liquidity_filter(table, markets, universe.read_settings(options))
    # median spread of SAP.DE, BMW.DE and WIDE.PA is 2 bps; 9 bps is above 3 x 2
    assert kept == ["AAPL", "SAP.DE", "BMW.DE"]
    assert dropped == {"WIDE.PA": "liquidity: spread 9.0 bps"}


def test_thin_crypto_is_dropped(universe: ModuleType, options: dict[str, Any]):
    table = pd.DataFrame(
        {"dollar_volume": [5.0, 0.5], "spread_bps": [np.nan, np.nan]}, index=["BIG", "THIN"]
    )
    markets = pd.Series(dict.fromkeys(table.index, Market.CRYPTO))
    kept, dropped = universe.liquidity_filter(table, markets, universe.read_settings(options))
    assert kept == ["BIG"]
    assert dropped == {"THIN": "liquidity: median daily amount 0"}


def test_selection_pool_splits_stocks_into_us_and_non_us(universe: ModuleType):
    assert universe.selection_pool("AAPL", Market.STOCK) == "stock_us"
    assert universe.selection_pool("7203.T", Market.STOCK) == "stock_non_us"
    assert universe.selection_pool("EURUSD", Market.FX) == "forex"
