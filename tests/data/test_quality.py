import datetime

import numpy as np
import pandas as pd
import pytest

from src.data._config import Instrument, QualityLimits
from src.data._quality import (
    adjust_for_splits,
    check_quality,
    session_closes,
    stock_calendar,
    yahoo_return_difference,
)
from src.enums import Market, Source

LIMITS = QualityLimits(
    min_coverage=0.95, max_gap_days=10, max_negative_spread_share=0.001, outlier_sigma=10,
    max_abs_return=0.4, yahoo_return_median_bps=30, yahoo_return_p99_bps=300, max_flat_share=0.25,
)  # fmt: skip
NVDA = Instrument("NVDA", Source.DUKASCOPY, Market.STOCK, "nvdaususd", calendar="XNYS")


def _session_bars(prices: dict[str, float]) -> pd.DataFrame:
    rows = []
    for day, price in prices.items():
        index = pd.date_range(f"{day} 14:30", f"{day} 20:45", freq="15min", tz="UTC")
        columns = {"close": price, "high": price + 1, "low": price - 1, "volume": 10.0}
        rows.append(pd.DataFrame(columns, index=index))
    return pd.concat(rows)


def test_split_adjustment_removes_the_jump():
    bars = _session_bars({"2024-06-07": 1200.0, "2024-06-10": 121.0})
    splits = pd.Series([10.0], index=[datetime.date(2024, 6, 10)])
    adjusted = adjust_for_splits(bars, splits, timezone="America/New_York")
    assert adjusted.loc["2024-06-07", "close"].iloc[0] == pytest.approx(120.0)
    assert adjusted.loc["2024-06-07", "volume"].iloc[0] == pytest.approx(100.0)
    assert adjusted.loc["2024-06-10", "close"].iloc[0] == pytest.approx(121.0)
    assert bars.loc["2024-06-07", "close"].iloc[0] == pytest.approx(1200.0)


def test_session_closes_take_the_last_bar_inside_the_session():
    bars = _session_bars({"2024-06-07": 100.0, "2024-06-10": 101.0})
    after_hours = pd.DataFrame(
        {"close": [999.0], "high": [999.0], "low": [999.0], "volume": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-10 22:00", tz="UTC")]),
    )
    closes = session_closes(pd.concat([bars, after_hours]), stock_calendar(NVDA))
    assert closes.to_dict() == {datetime.date(2024, 6, 7): 100.0, datetime.date(2024, 6, 10): 101.0}


def test_a_stock_without_a_calendar_raises():
    with pytest.raises(ValueError, match=r"SAP\.DE has no exchange calendar"):
        stock_calendar(Instrument("SAP.DE", Source.DUKASCOPY, Market.STOCK, "x"))


def _year_of_sessions(seed: int) -> dict[str, float]:
    """Random-walk closes on every New York session of 2024 (252 sessions)."""
    sessions = stock_calendar(NVDA).sessions_in_range("2024-01-02", "2024-12-31")
    walk = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(sessions))))
    return {str(day.date()): float(price) for day, price in zip(sessions, walk, strict=True)}


def _reference(prices: dict[str, float]) -> pd.Series:
    return pd.Series(list(prices.values()), index=[pd.Timestamp(d).date() for d in prices])


def test_clean_stock_passes_and_a_noisy_reference_fails():
    prices = _year_of_sessions(0)
    bars, reference = _session_bars(prices), _reference(prices)
    good = check_quality(bars, NVDA, limits=LIMITS, reference=reference, end="2025-01-01")
    assert good["passed"], good["reason"]
    assert good["yahoo_return_median_bps"] == pytest.approx(0.0)
    assert good["yahoo_return_p99_bps"] == pytest.approx(0.0)
    # closes that disagree by about 1 % every day: returns differ by about 100 bps
    noise = np.exp(np.random.default_rng(1).normal(0, 0.01, len(reference)))
    bad = check_quality(bars, NVDA, limits=LIMITS, reference=reference * noise, end="2025-01-01")
    assert not bad["passed"]
    assert "Yahoo median return difference" in bad["reason"]


def test_a_level_offset_to_yahoo_does_not_count():
    # Yahoo's closes 2 % higher on every day (another adjustment of past prices): same returns
    prices = _year_of_sessions(2)
    row = check_quality(
        _session_bars(prices), NVDA, limits=LIMITS, reference=_reference(prices) * 1.02,
        end="2025-01-01",
    )  # fmt: skip
    assert row["yahoo_return_median_bps"] == pytest.approx(0.0, abs=1e-6)
    assert row["passed"], row["reason"]


@pytest.mark.parametrize(("bad_days", "passes"), [(0, True), (1, True), (2, False)])
def test_a_few_corporate_action_days_do_not_decide_the_check(bad_days: int, passes: bool):
    # Yahoo's close is 20 % off on `bad_days` single days (a corporate action handled
    # differently): each moves the two returns around it; the 99th percentile of 251 returns
    # leaves out the largest 2.5, so one such day a year passes and two fail (the selection
    # window's 4 years of about 1000 returns absorb about 5 such days)
    prices = _year_of_sessions(3)
    reference = _reference(prices)
    for position in range(bad_days):
        reference.iloc[50 + 40 * position] *= 1.2
    row = check_quality(
        _session_bars(prices), NVDA, limits=LIMITS, reference=reference, end="2025-01-01"
    )
    assert row["yahoo_return_median_bps"] == pytest.approx(0.0, abs=1e-6)
    assert row["passed"] is passes
    if not passes:
        assert "Yahoo 99th percentile return difference" in row["reason"]


def test_the_return_difference_runs_between_common_days_only():
    # Yahoo has no close on day 3: our return from day 2 to day 4 is compared with Yahoo's
    ours = pd.Series([100.0, 110.0, 121.0, 133.1], index=[1, 2, 3, 4])
    yahoo = pd.Series([50.0, 55.0, 66.55], index=[1, 2, 4])
    difference = yahoo_return_difference(ours, yahoo)
    assert difference.index.tolist() == [2, 4]
    np.testing.assert_allclose(difference.to_numpy(), [0.0, 0.0], atol=1e-9)
    shifted = yahoo_return_difference(ours, yahoo.mul([1.0, 1.0, 1.01]))
    assert shifted.iloc[1] == pytest.approx(abs(np.log(1.01)) * 1e4)


def test_negative_spread_and_jump_fail_outside_crypto():
    index = pd.date_range("2024-06-03", periods=5 * 96, freq="15min", tz="UTC")
    close = np.full(len(index), 1.10)
    close[200:] = 2.0
    bars = pd.DataFrame({"mid_close": close, "mid_high": close + 0.001, "mid_low": close - 0.001,
                         "spread": 0.0001}, index=index)  # fmt: skip
    bars.iloc[10, bars.columns.get_loc("spread")] = -0.001
    row = check_quality(
        bars,
        Instrument("EURUSD", Source.DUKASCOPY, Market.FX, "eurusd"),
        limits=LIMITS,
        reference=None,
        end="2024-06-08",
    )
    assert not row["passed"]
    assert "jump" in row["reason"]
    assert row["negative_spread_share"] > 0


def test_split_adjustment_keeps_integer_volumes_valid():
    index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    bars = pd.DataFrame({"close": [10.0, 10.0, 5.0, 5.0], "volume": [3, 3, 6, 6]}, index=index)
    splits = pd.Series([1.5], index=[pd.Timestamp("2024-01-03")])
    adjusted = adjust_for_splits(bars, splits, timezone="UTC")
    assert adjusted["volume"].tolist() == [4.5, 4.5, 6.0, 6.0]


def test_an_index_with_an_exchange_calendar_does_not_count_its_holidays_as_missing():
    # Hong Kong sessions of 2024 H1: every weekday bar except the exchange holidays
    days = pd.bdate_range("2024-01-02", "2024-06-28", tz="UTC") + pd.Timedelta(hours=2)
    hkg = Instrument("HKG", Source.DUKASCOPY, Market.INDEX, "hkgidxhkd", calendar="XHKG")
    sessions = stock_calendar(
        Instrument("0700.HK", Source.DUKASCOPY, Market.STOCK, "x", calendar="XHKG")
    )
    open_days = sessions.sessions_in_range("2024-01-02", "2024-06-28")
    index = days[days.normalize().tz_localize(None).isin(open_days)]
    close = np.linspace(100, 101, len(index))
    bars = pd.DataFrame(
        {"close": close, "open": close, "high": close + 0.1, "low": close - 0.1}, index=index
    )
    on_exchange = check_quality(bars, hkg, limits=LIMITS, reference=None, end="2024-06-29")
    on_weekdays = check_quality(
        bars,
        Instrument("HKG", Source.DUKASCOPY, Market.INDEX, "x"),
        limits=LIMITS,
        reference=None,
        end="2024-06-29",
    )
    assert on_exchange["coverage"] == pytest.approx(1.0)
    assert on_weekdays["coverage"] < 0.95


def _hourly_crypto(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, len(index))))
    return pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999}, index=index)


BTC = Instrument("BTCUSDT", Source.BINANCE, Market.CRYPTO, "BTCUSDT")


def test_bars_that_stop_early_are_missing_days_up_to_the_end():
    # bars stop on 2024-03-01, but the period runs to 2024-04-01
    bars = _hourly_crypto("2024-01-01", "2024-03-01")
    full = check_quality(bars, BTC, limits=LIMITS, reference=None, end="2024-03-01")
    assert full["coverage"] == pytest.approx(1.0)
    assert full["passed"], full["reason"]
    early = check_quality(bars, BTC, limits=LIMITS, reference=None, end="2024-04-01")
    assert early["coverage"] == pytest.approx(60 / 91, abs=1e-4)
    assert early["max_gap_days"] == pytest.approx(31.04, abs=0.01)
    assert not early["passed"]
    assert "coverage" in early["reason"]
    assert "gap" in early["reason"]


def test_frozen_filler_bars_fail_the_flat_share():
    bars = _hourly_crypto("2024-01-01", "2024-03-01")
    frozen = bars.index >= pd.Timestamp("2024-02-10", tz="UTC")
    bars.loc[frozen, ["close", "high", "low"]] = 100.0
    row = check_quality(bars, BTC, limits=LIMITS, reference=None, end="2024-03-01")
    assert row["flat_share"] == pytest.approx(frozen.mean(), abs=1e-4)
    assert not row["passed"]
    assert "high equals low" in row["reason"]
    few = _hourly_crypto("2024-01-01", "2024-03-01")
    few.iloc[:10, [1, 2]] = few.iloc[:10, [0, 0]].to_numpy()
    assert check_quality(few, BTC, limits=LIMITS, reference=None, end="2024-03-01")["passed"]
