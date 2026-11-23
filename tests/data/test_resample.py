import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data import add_mid_and_spread, resample_bars
from src.enums import Timeframe


def _bid_ask_bars(index: pd.DatetimeIndex, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 1, len(index)).cumsum()
    frame = pd.DataFrame(index=index)
    for side, shift in (("bid", 0.0), ("ask", 0.05)):
        frame[f"{side}_open"] = close + shift + rng.normal(0, 0.1, len(index))
        frame[f"{side}_close"] = close + shift
        frame[f"{side}_high"] = frame[[f"{side}_open", f"{side}_close"]].max(axis=1) + 0.2
        frame[f"{side}_low"] = frame[[f"{side}_open", f"{side}_close"]].min(axis=1) - 0.2
        frame[f"{side}_volume"] = rng.uniform(1, 2, len(index))
    frame["n_bars"] = 1
    frame.index.name = "open_time"
    return add_mid_and_spread(frame)


def _week_of_15m(start: str = "2026-11-09") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=7 * 96, freq="15min", tz="UTC")


@pytest.mark.parametrize(
    ("timeframe", "first_open", "count"),
    [
        (Timeframe.M30, "2026-11-09 00:00", 7 * 48),
        (Timeframe.H1, "2026-11-09 00:00", 7 * 24),
        (Timeframe.H4, "2026-11-09 00:00", 7 * 6),
        (Timeframe.H12, "2026-11-09 00:00", 7 * 2),
        (Timeframe.D1, "2026-11-09 00:00", 7),
        (Timeframe.W1, "2026-11-09 00:00", 1),
    ],
)
def test_bar_boundaries(timeframe, first_open, count):
    bars = resample_bars(_bid_ask_bars(_week_of_15m()), timeframe)
    assert bars.index[0] == pd.Timestamp(first_open, tz="UTC")
    assert len(bars) == count
    assert (bars["n_bars"] == timeframe.minutes // 15).all()


def test_4h_and_12h_open_on_the_utc_grid():
    index = pd.date_range("2026-11-09 03:00", periods=40, freq="15min", tz="UTC")
    four_hour = resample_bars(_bid_ask_bars(index), Timeframe.H4)
    assert set(four_hour.index.hour) <= {0, 4, 8, 12, 16, 20}
    assert set(resample_bars(_bid_ask_bars(index), Timeframe.H12).index.hour) <= {0, 12}


def test_3d_is_anchored_on_2020_01_01():
    index = pd.date_range("2020-01-01", periods=10 * 96, freq="15min", tz="UTC")
    opens = resample_bars(_bid_ask_bars(index), Timeframe.D3).index
    assert list(opens) == list(pd.date_range("2020-01-01", periods=4, freq="3D", tz="UTC"))
    late = pd.date_range("2026-11-09", periods=96, freq="15min", tz="UTC")
    late_open = resample_bars(_bid_ask_bars(late), Timeframe.D3).index[0]
    assert (late_open - pd.Timestamp("2020-01-01", tz="UTC")).days % 3 == 0


def test_weeks_open_on_monday():
    index = pd.date_range("2026-11-12", periods=20 * 96, freq="15min", tz="UTC")
    opens = resample_bars(_bid_ask_bars(index), Timeframe.W1).index
    assert set(opens.day_name()) == {"Monday"}


def test_closed_market_bars_are_absent_not_filled():
    week = _week_of_15m()
    trading = week[week.day_of_week < 5]
    bars = resample_bars(_bid_ask_bars(trading), Timeframe.D1)
    assert list(bars.index.day_name()) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    assert not bars.isna().any().any()


def test_ohlc_aggregation_and_volumes():
    source = _bid_ask_bars(_week_of_15m())
    hour = resample_bars(source, Timeframe.H1).iloc[0]
    first = source.iloc[:4]
    assert hour["bid_open"] == pytest.approx(first["bid_open"].iloc[0])
    assert hour["bid_high"] == pytest.approx(first["bid_high"].max())
    assert hour["bid_low"] == pytest.approx(first["bid_low"].min())
    assert hour["bid_close"] == pytest.approx(first["bid_close"].iloc[-1])
    assert hour["ask_volume"] == pytest.approx(first["ask_volume"].sum())


def test_mid_and_spread_are_recomputed_from_bid_and_ask():
    hour = resample_bars(_bid_ask_bars(_week_of_15m()), Timeframe.H1)
    assert np.allclose(hour["mid_close"], (hour["bid_close"] + hour["ask_close"]) / 2)
    assert np.allclose(hour["mid_high"], (hour["bid_high"] + hour["ask_high"]) / 2)
    assert np.allclose(hour["spread"], hour["ask_close"] - hour["bid_close"])


def test_rejects_naive_index_and_string_timeframe():
    bars = _bid_ask_bars(_week_of_15m())
    with pytest.raises(ValueError, match="UTC"):
        resample_bars(bars.tz_convert(None), Timeframe.H1)
    with pytest.raises(TypeError, match="Timeframe"):
        resample_bars(bars, "1h")  # type: ignore[arg-type]


@settings(max_examples=50, deadline=None)
@given(
    keep=st.lists(st.booleans(), min_size=96, max_size=96),
    timeframe=st.sampled_from([Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.H12]),
)
def test_every_15m_bar_lands_in_exactly_one_coarser_bar(keep, timeframe):
    index = pd.date_range("2026-11-09", periods=96, freq="15min", tz="UTC")[np.array(keep)]
    if len(index) == 0:
        return
    bars = resample_bars(_bid_ask_bars(index), timeframe)
    assert bars["n_bars"].sum() == len(index)
    assert (bars["bid_high"] >= bars["bid_low"]).all()


def test_vwap_is_the_volume_weighted_mean_of_the_finer_vwaps():
    index = pd.date_range("2024-01-02 14:00", periods=4, freq="15min", tz="UTC", name="open_time")
    bars = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": [1.0, 3.0, 0.0, 4.0],
            "vwap": [10.0, 20.0, 99.0, 30.0],
        },
        index=index,
    )
    hourly = resample_bars(bars, Timeframe.H1)
    assert hourly["vwap"].iloc[0] == pytest.approx((10 * 1 + 20 * 3 + 30 * 4) / 8)
    assert hourly["volume"].iloc[0] == pytest.approx(8.0)
