import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.enums import Timeframe
from src.utils import (
    bar_grid,
    closures,
    days_after,
    days_of_month,
    expected_days,
    half_years,
    hour_grid,
    in_window,
    is_utc_index,
    iso_to_utc,
    last_at_or_before,
    months_between,
    ms_to_utc,
    spacing_hours,
    to_utc,
    week_starts,
    years_between,
)


def test_to_utc_reads_naive_as_utc_and_converts_aware():
    assert to_utc("2026-01-01") == pd.Timestamp("2026-01-01", tz="UTC")
    oslo = pd.Timestamp("2026-01-01 01:00", tz="Europe/Oslo")
    assert to_utc(oslo) == pd.Timestamp("2026-01-01 00:00", tz="UTC")


def test_ms_to_utc():
    index = ms_to_utc(pd.Series([0, 900_000]))
    assert is_utc_index(index)
    assert index.name == "open_time"
    assert index[1] == pd.Timestamp("1970-01-01 00:15", tz="UTC")


def test_is_utc_index_rejects_naive_and_other_zones():
    assert not is_utc_index(pd.date_range("2026-01-01", periods=2, freq="h"))
    assert not is_utc_index(pd.date_range("2026-01-01", periods=2, freq="h", tz="Europe/Oslo"))


def test_bar_grid_anchors():
    assert bar_grid(Timeframe.H4) == ("4h", pd.Timestamp("2020-01-01", tz="UTC"))
    frequency, origin = bar_grid(Timeframe.W1)
    assert frequency == "7D"
    assert origin.day_name() == "Monday"


def test_bar_grid_rejects_strings():
    with pytest.raises(TypeError, match="Timeframe"):
        bar_grid("1h")  # type: ignore[arg-type]


def test_years_between_cuts_both_ends():
    assert years_between("2025-06-01", "2026-02-01") == [
        (2025, "2025-06-01", "2026-01-01"),
        (2026, "2026-01-01", "2026-02-01"),
    ]
    assert years_between("2026-01-01", "2026-01-01") == []


def test_months_between_excludes_end():
    assert months_between("2025-11-15", "2026-02-01") == ["2025-11", "2025-12", "2026-01"]


def test_days_of_month_handles_leap_year():
    days = days_of_month("2024-02")
    assert len(days) == 29
    assert days[0] == pd.Timestamp("2024-02-01", tz="UTC")


def test_expected_days_weekdays_only():
    start, end = to_utc("2026-11-09"), to_utc("2026-11-15 23:00")
    assert len(expected_days(start, end, weekdays_only=False)) == 7
    assert len(expected_days(start, end, weekdays_only=True)) == 5


def test_week_starts_are_the_monday_midnight_of_the_iso_week():
    times = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-07 23:59", "2024-01-08 00:00", "2020-01-01 12:00"], tz="UTC"
    )
    expected = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-08", "2019-12-30"], tz="UTC")
    pd.testing.assert_index_equal(week_starts(times), expected)
    # a time in another zone is placed in its UTC week (Monday 00:30 in Oslo is Sunday UTC)
    oslo = pd.DatetimeIndex(["2024-01-08 00:30"], tz="Europe/Oslo")
    assert week_starts(oslo)[0] == to_utc("2024-01-01")


def test_iso_to_utc():
    index = iso_to_utc(pd.Series(["2020-01-02T09:00:00Z", "2020-01-02T10:00:00+01:00"]))
    assert is_utc_index(index)
    assert list(index) == [pd.Timestamp("2020-01-02 09:00", tz="UTC")] * 2


def test_hour_grid_excludes_the_end_and_rejects_a_zero_step():
    start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-02", tz="UTC")
    grid = hour_grid(start, end, step_hours=4)
    assert grid.tolist() == [start + pd.Timedelta(hours=4 * k) for k in range(6)]
    assert str(grid.tz) == "UTC"
    assert len(hour_grid(start, end, step_hours=1)) == 24
    with pytest.raises(ValueError, match="step_hours must be >= 1, got 0"):
        hour_grid(start, end, step_hours=0)


def test_last_at_or_before_finds_the_last_time_not_after_each_time():
    index = pd.DatetimeIndex(["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 05:00"], tz="UTC")
    times = pd.DatetimeIndex(
        ["2025-01-01 00:59", "2025-01-01 01:00", "2025-01-01 04:59", "2025-01-02"], tz="UTC"
    )
    assert last_at_or_before(index, times).tolist() == [-1, 0, 1, 2]
    microseconds = pd.DatetimeIndex(index.as_unit("us"))
    assert last_at_or_before(microseconds, times).tolist() == [-1, 0, 1, 2]


def test_spacing_hours_is_the_median_gap_and_needs_two_times():
    times = pd.DatetimeIndex(
        ["2025-01-01 00:00", "2025-01-01 04:00", "2025-01-01 08:00", "2025-01-03 00:00"], tz="UTC"
    )
    assert spacing_hours(times) == pytest.approx(4.0)
    assert spacing_hours(pd.date_range("2025-01-01", periods=5, freq="167h", tz="UTC")) == 167.0
    with pytest.raises(ValueError, match="at least 2 times, got 1"):
        spacing_hours(times[:1])


# --- closures and half-years ---------------------------------------------------------------


def _close_times(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(list(values), tz="UTC")


def test_a_closure_runs_from_the_close_of_a_bar_to_the_open_of_the_next():
    # the last bar closes Friday 21:00, the next opens Sunday 21:00 (closes 22:00): 48 hours
    times = _close_times("2024-03-08 20:00", "2024-03-08 21:00", "2024-03-10 22:00")
    found = closures(times, bar=pd.Timedelta(hours=1), min_hours=48)
    assert found["position"].tolist() == [2]
    assert found["close_time"].tolist() == [times[1]]
    assert found["reopen_time"].tolist() == [pd.Timestamp("2024-03-10 21:00", tz="UTC")]
    assert found["closed_hours"].tolist() == [48.0]
    assert closures(times, bar=pd.Timedelta(hours=1), min_hours=48.5).empty


@given(st.integers(1, 400), st.integers(1, 60), st.integers(1, 30))
@settings(max_examples=50, deadline=None)
def test_every_closure_is_a_gap_of_at_least_the_minimum(start, closed, minimum):
    hours = pd.date_range("2024-01-01 01:00", periods=500, freq="1h", tz="UTC")
    times = hours[:start].append(hours[start + closed :])
    found = closures(times, bar=pd.Timedelta(hours=1), min_hours=minimum)
    assert found["position"].tolist() == ([start] if closed >= minimum else [])


def test_half_years_split_at_july():
    times = _close_times("2024-01-01", "2024-06-30 23:00", "2024-07-01", "2025-12-31")
    assert half_years(times).tolist() == ["2024H1", "2024H1", "2024H2", "2025H2"]


def test_days_after_shifts_whole_utc_days_both_ways():
    assert days_after("2024-01-01", -1) == pd.Timestamp("2023-12-31", tz="UTC")
    assert days_after("2020-01-01", 7) == pd.Timestamp("2020-01-08", tz="UTC")
    oslo = pd.Timestamp("2024-03-01 01:00", tz="Europe/Oslo")
    assert days_after(oslo, 1) == pd.Timestamp("2024-03-02 00:00", tz="UTC")


def test_in_window_keeps_start_drops_end_and_reads_naive_dates_as_utc():
    aware = pd.date_range("2023-12-31", periods=4, freq="D", tz="UTC")
    assert in_window(aware, start="2024-01-01", end="2024-01-03").tolist() == [
        False, True, True, False,
    ]  # fmt: skip
    naive = pd.DatetimeIndex(["2023-12-31", "2024-01-01", "2024-01-03"])
    assert in_window(naive, start="2024-01-01", end="2024-01-03").tolist() == [False, True, False]
