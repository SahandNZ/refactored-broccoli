"""rq4: the periods of a model's decisions and the grid of the leakage quarters."""

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluate import LEAKAGE_FOLDS, TEST_FOLDS

RELEASE = pd.Timestamp("2025-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2024-06-30", tz="UTC")
NO_CUTOFF = pd.NaT


def _losses(
    times: Sequence[str] | pd.DatetimeIndex, horizons: Sequence[int], windows: Sequence[str]
) -> pd.DataFrame:
    # a bar every hour: the label of horizon h ends h hours after its decision
    time = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601")
    label_time = time + pd.to_timedelta(list(horizons), unit="h")
    return pd.DataFrame({
        "time": time, "asset": "BTCUSDT", "horizon": list(horizons), "window": list(windows),
        "crps": 1.0, "market": "crypto", "label_time": label_time,
    })  # fmt: skip


def _periods(
    leakage: ModuleType, losses: pd.DataFrame, cutoff: pd.Timestamp | None
) -> pd.DataFrame:
    # None: the model states no cutoff (NaT in the cache)
    return leakage.leakage_periods(
        losses,
        release=RELEASE,
        cutoff=NO_CUTOFF if cutoff is None else cutoff,
        before_windows=["l1"],
        between_windows=["l1"],
        after_windows=["t1"],
    )


# --- leakage_periods: no look-ahead in the split ------------------------------------------


def test_a_label_at_the_cutoff_or_the_release_is_left_out(leakage: ModuleType) -> None:
    times = ["2024-06-29 22:00", "2024-06-29 23:00", "2024-12-31 22:00", "2024-12-31 23:00"]
    periods = _periods(leakage, _losses(times, [1, 1, 1, 1], ["l1"] * 4), CUTOFF)
    assert periods["period"].tolist() == ["before", "between"]
    kept = pd.to_datetime([times[0], times[2]], utc=True)
    assert periods["time"].tolist() == kept.tolist()


def test_a_decision_at_the_cutoff_is_between_and_at_the_release_is_not_after(
    leakage: ModuleType,
) -> None:
    times = ["2024-06-30 00:00", "2025-01-01 00:00", "2025-01-01 01:00"]
    periods = _periods(leakage, _losses(times, [1, 1, 1], ["l1", "t1", "t1"]), CUTOFF)
    assert periods["period"].tolist() == ["between", "after"]
    assert periods["time"].iloc[1] == pd.Timestamp("2025-01-01 01:00", tz="UTC")


def test_without_a_cutoff_everything_before_the_release_is_before(leakage: ModuleType) -> None:
    times = ["2023-01-01", "2024-12-31 22:00", "2025-03-01"]
    periods = _periods(leakage, _losses(times, [1, 1, 24], ["l1"] * 3), None)
    assert periods["period"].tolist() == ["before", "before"]


def test_a_cutoff_after_the_release_counts_only_until_the_release(leakage: ModuleType) -> None:
    losses = _losses(["2024-12-31 22:00", "2025-02-01"], [1, 1], ["l1", "t1"])
    periods = _periods(leakage, losses, pd.Timestamp("2025-06-30", tz="UTC"))
    assert periods["period"].tolist() == ["before", "after"]


def test_decisions_outside_the_windows_are_left_out(leakage: ModuleType) -> None:
    losses = _losses(["2023-01-01", "2025-02-01", "2025-02-01"], [1, 1, 1], ["d1", "d2", "t1"])
    assert _periods(leakage, losses, CUTOFF)["window"].tolist() == ["t1"]


def test_between_also_takes_the_development_folds_and_before_does_not(
    leakage: ModuleType,
) -> None:
    # a model released in 2025: its months between the cutoff and the release include the
    # development folds; a development decision is never in before
    times = ["2024-05-01", "2024-11-02", "2024-12-31 22:00"]
    losses = _losses(times, [1, 1, 1], ["l1", "d1", "d1"])
    periods = leakage.leakage_periods(
        losses,
        release=RELEASE,
        cutoff=CUTOFF,
        before_windows=["l1"],
        between_windows=["l1", "d1"],
        after_windows=["t1"],
    )
    assert periods["period"].tolist() == ["before", "between", "between"]
    without_cutoff = leakage.leakage_periods(
        losses,
        release=RELEASE,
        cutoff=NO_CUTOFF,
        before_windows=["l1"],
        between_windows=["l1", "d1"],
        after_windows=["t1"],
    )
    assert without_cutoff["period"].tolist() == ["before"]


def test_a_model_without_a_release_date_raises(leakage: ModuleType) -> None:
    losses = _losses(["2023-01-01"], [1], ["l1"])
    with pytest.raises(ValueError, match="needs a release date"):
        leakage.leakage_periods(
            losses,
            release=pd.NaT,
            cutoff=CUTOFF,
            before_windows=["l1"],
            between_windows=["l1"],
            after_windows=["t1"],
        )


def test_a_label_time_after_the_cutoff_is_not_before_even_when_the_decision_is(
    leakage: ModuleType,
) -> None:
    # a market that closes: the 1-bar label of a decision before the cutoff ends after it
    losses = _losses(["2024-06-29 20:00"], [1], ["l1"]).assign(
        label_time=pd.Timestamp("2024-07-01 14:00", tz="UTC")
    )
    assert _periods(leakage, losses, CUTOFF).empty


hours = st.integers(0, 4 * 365 * 24)
start = pd.Timestamp("2022-01-01", tz="UTC")


@given(
    offsets=st.lists(hours, min_size=1, max_size=40),
    horizons=st.lists(st.sampled_from([1, 4, 24]), min_size=40, max_size=40),
    release_hour=hours,
    cutoff_hour=st.none() | hours,
)
@settings(max_examples=200, deadline=None)
def test_every_period_uses_only_data_on_its_side_of_the_dates(
    leakage: ModuleType,
    offsets: list[int],
    horizons: list[int],
    release_hour: int,
    cutoff_hour: int | None,
) -> None:
    times = start + pd.to_timedelta(offsets, unit="h")
    windows = ["l1" if index % 2 else "t1" for index in range(len(offsets))]
    losses = _losses(times, horizons[: len(offsets)], windows)
    release = start + pd.Timedelta(hours=release_hour)
    cutoff = NO_CUTOFF if cutoff_hour is None else start + pd.Timedelta(hours=cutoff_hour)
    periods = leakage.leakage_periods(
        losses,
        release=release,
        cutoff=cutoff,
        before_windows=["l1"],
        between_windows=["l1"],
        after_windows=["t1"],
    )
    label_time = periods["label_time"]
    seen_until = release if cutoff_hour is None else min(cutoff, release)
    before = periods["period"] == "before"
    between = periods["period"] == "between"
    after = periods["period"] == "after"
    assert (label_time[before] < seen_until).all()
    assert ((periods["time"][between] >= cutoff) & (label_time[between] < release)).all()
    assert (periods["time"][after] > release).all()
    assert (periods.loc[before | between, "window"] == "l1").all()
    assert (periods.loc[after, "window"] == "t1").all()
    expected_before = losses["window"].eq("l1") & (losses["label_time"] < seen_until)
    assert before.sum() == expected_before.sum()
    assert after.sum() == (losses["window"].eq("t1") & (losses["time"] > release)).sum()
    if cutoff_hour is None:
        assert not between.any()


# --- leakage_grid: the after period on the grid of the leakage quarters --------------------


def _labels(times: Sequence[str], assets: Sequence[str], window: str = "t1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(list(times), utc=True),
            "asset": list(assets),
            "window": window,
            "horizon": 1,
        }
    )


def _grid(leakage: ModuleType, labels: pd.DataFrame) -> list[tuple[str, str, str]]:
    kept = leakage.leakage_grid(labels, hours=4, folds=TEST_FOLDS)
    return [
        (asset, str(time.strftime("%d %H:%M")), str(point.strftime("%d %H:%M")))
        for time, asset, point in zip(kept["time"], kept["asset"], kept["point"], strict=True)
    ]


def test_the_grid_keeps_the_last_decision_since_the_previous_point(leakage: ModuleType) -> None:
    # t1 starts 2025-11-01 00:00, grid points 00:00, 04:00, ...; the leakage quarters decide at
    # a point with the last close since the previous point: a trading-hours market whose first
    # bar of the day closes at 14:00 decides at 16:00 (last close 16:00), not at 14:00
    day = "2025-11-01 "
    crypto = [day + f"{hour:02d}:00" for hour in range(9)]
    stock = [day + hour for hour in ("14:00", "15:00", "16:00", "17:00", "18:00", "21:00")]
    labels = _labels(crypto + stock, ["BTC"] * 9 + ["AAPL"] * 6)
    assert _grid(leakage, labels) == [
        ("BTC", "01 00:00", "01 00:00"),
        ("BTC", "01 04:00", "01 04:00"),
        ("BTC", "01 08:00", "01 08:00"),
        ("AAPL", "01 16:00", "01 16:00"),
        ("AAPL", "01 18:00", "01 20:00"),
        ("AAPL", "01 21:00", "02 00:00"),
    ]


def test_the_grid_matches_the_decisions_of_a_4_hour_window(leakage: ModuleType) -> None:
    # random sessions: a point has a decision exactly when a bar closed since the previous
    # point, and it is the last such bar, as the leakage quarters of 02-forecasts decide
    rng = np.random.default_rng(5)
    hours = np.sort(rng.choice(24 * 20, size=200, replace=False))
    start = TEST_FOLDS[0].evaluation_start
    times = start + pd.to_timedelta(hours, unit="h")
    kept = leakage.leakage_grid(
        _labels(list(times.astype(str)), ["AAPL"] * 200), hours=4, folds=TEST_FOLDS
    )
    for point in pd.date_range(start, periods=24 * 20 // 4, freq="4h"):
        since = times[(times > point - pd.Timedelta(hours=4)) & (times <= point)]
        row = kept[kept["point"] == point]
        assert len(row) == min(len(since), 1)
        if len(since) > 0:
            assert row["time"].iloc[0] == since.max()


def test_a_window_already_on_the_grid_keeps_every_decision(leakage: ModuleType) -> None:
    # a leakage quarter decides only at 4-hour points (a closed market skips some): each
    # decision is the last of its point and keeps its own time
    times = ["2022-01-01 00:00", "2022-01-01 04:00", "2022-01-01 16:00", "2022-01-02 08:00"]
    labels = _labels(times, ["AAPL"] * 4, window="l1")
    kept = leakage.leakage_grid(labels, hours=4, folds=LEAKAGE_FOLDS)
    assert (kept["time"] == kept["point"]).all()
    assert len(kept) == 4


def test_the_grid_keeps_every_horizon_once_and_drops_the_window_end(leakage: ModuleType) -> None:
    labels = pd.concat(
        [_labels(["2025-11-01 01:00", "2026-01-31 23:00"], ["BTC", "BTC"]).assign(horizon=h)
         for h in (1, 4, 24)]
    )  # fmt: skip
    assert _grid(leakage, labels) == [("BTC", "01 01:00", "01 04:00")]
