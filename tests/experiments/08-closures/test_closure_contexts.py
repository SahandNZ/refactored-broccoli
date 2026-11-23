"""The decisions after a reopening and their contexts (08-closures), on synthetic closes."""

from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.panel import ewma_volatility

HOUR = pd.Timedelta(hours=1)
LOOKBACK, SPAN = 72, 24


def _closes(open_times: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    # one 1h close per open bar, indexed by its close time
    steps = np.random.default_rng(seed).normal(0, 0.003, len(open_times))
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=open_times + HOUR)


def _forex(weeks: int = 6, seed: int = 0) -> pd.Series:
    # closed from Friday 21:00 to Sunday 21:00 UTC: 48 hours without a bar every weekend
    hours = pd.date_range("2025-01-06", periods=weeks * 168, freq="1h", tz="UTC")
    weekday, hour = hours.day_of_week, hours.hour
    is_open = ~((weekday == 4) & (hour >= 21) | (weekday == 5) | (weekday == 6) & (hour < 21))
    return _closes(hours[is_open], seed)


def _crypto(weeks: int = 6) -> pd.Series:
    return _closes(pd.date_range("2025-01-06", periods=weeks * 168, freq="1h", tz="UTC"), 3)


def _contexts(
    closures: ModuleType, closes: pd.Series, *, lookback: int = LOOKBACK, min_own_closes: int = 1
):
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    return closures.closure_contexts(
        {"X": closes},
        decisions.assign(asset="X"),
        lookback=lookback,
        span_bars=SPAN,
        min_own_closes=min_own_closes,
    )


# Decisions after a reopening


def test_targets_are_the_first_three_own_bars_after_each_weekend(closures: ModuleType):
    closes = _forex()
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    sundays = pd.date_range("2025-01-12 21:00", periods=6, freq="7D", tz="UTC")
    expected = sorted(day + offset * HOUR for day in sundays for offset in range(3))
    assert list(decisions["time"]) == expected
    assert list(decisions["bar"]) == [0, 1, 2] * 6
    np.testing.assert_allclose(decisions["closed_hours"], 48.0)
    # each decision is one hour before its target bar closes: the target is the next own bar
    assert set(decisions["time"] + HOUR) <= set(closes.index)


def test_the_first_target_is_the_reopening_bar_whose_decision_sees_the_whole_closure(
    closures: ModuleType,
):
    closes = _forex()
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    first = decisions[decisions["bar"] == 0]["time"]
    for time in first:
        before = closes.index[closes.index <= time]
        assert (time - before[-1]) / HOUR == 48
        assert time + HOUR in closes.index


@pytest.mark.parametrize(("closed", "is_event"), [(11, False), (12, True), (13, True)])
def test_a_closure_needs_at_least_min_closure_hours(
    closures: ModuleType, closed: int, is_event: bool
):
    hours = pd.date_range("2025-01-06", periods=200, freq="1h", tz="UTC")
    closes = _closes(hours[(np.arange(200) < 100) | (np.arange(200) >= 100 + closed)])
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    assert len(decisions) == (3 if is_event else 0)


def test_a_new_reopening_restarts_the_count_of_target_bars(closures: ModuleType):
    hours = pd.date_range("2025-01-06", periods=200, freq="1h", tz="UTC")
    position = np.arange(200)
    is_open = (position < 50) | ((position >= 70) & (position < 71)) | (position >= 90)
    closes = _closes(hours[is_open])
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    assert list(decisions["bar"]) == [0, 0, 1, 2]
    np.testing.assert_allclose(decisions["closed_hours"], [20.0, 19.0, 19.0, 19.0])


def test_a_series_without_closures_has_no_decision(closures: ModuleType):
    assert closures.reopening_decisions(_crypto(), min_closure_hours=12, reopen_bars=3).empty


def test_artificial_closures_cut_exactly_the_weekend_hours(closures: ModuleType):
    closes = _crypto()
    cut = closures.cut_weekends(closes)
    open_days = (pd.DatetimeIndex(cut.index) - HOUR).day_of_week
    removed = pd.DatetimeIndex(closes.index.difference(cut.index))
    assert (open_days < 5).all()
    assert ((removed - HOUR).day_of_week >= 5).all()
    assert len(removed) == 48 * 6
    np.testing.assert_allclose(cut.to_numpy(), closes[cut.index].to_numpy())
    decisions = closures.reopening_decisions(cut, min_closure_hours=12, reopen_bars=3)
    assert set(decisions.loc[decisions["bar"] == 0, "time"].dt.day_of_week) == {0}
    assert set(decisions.loc[decisions["bar"] == 0, "time"].dt.hour) == {0}
    np.testing.assert_allclose(decisions["closed_hours"], 48.0)


# The three contexts


def test_every_representation_has_the_same_decisions_and_labels(closures: ModuleType):
    contexts = _contexts(closures, _forex())
    assert set(contexts.windows) == set(closures.ARMS)
    for window in contexts.windows.values():
        assert window.shape == (len(contexts.decisions), LOOKBACK)
    assert len(contexts.decisions) == 18


def test_the_calendar_representations_differ_only_at_closed_hours(closures: ModuleType):
    contexts = _contexts(closures, _forex())
    filled, missing = contexts.windows[closures.FORWARD_FILLED], contexts.windows[closures.MISSING]
    is_closed = contexts.is_closed
    assert is_closed.any()
    np.testing.assert_array_equal(filled[~is_closed], missing[~is_closed])
    assert np.isnan(missing[is_closed]).all()
    assert np.isfinite(missing[~is_closed]).all()
    assert np.isfinite(filled).all()


def test_the_nan_arm_is_nan_exactly_at_the_hours_without_a_bar(closures: ModuleType):
    closes = _forex()
    contexts = _contexts(closures, closes)
    close_times = set(closes.index)
    for row, time in enumerate(contexts.decisions["time"]):
        hours = pd.date_range(end=time, periods=LOOKBACK, freq="1h")
        expected = np.array([hour not in close_times for hour in hours])
        np.testing.assert_array_equal(contexts.is_closed[row], expected)
        np.testing.assert_array_equal(np.isnan(contexts.windows[closures.MISSING][row]), expected)


def test_every_representation_sees_the_same_calendar_hours_and_trading_hours_keep_only_own_bars(
    closures: ModuleType,
):
    # D2: the trading-hours arm is the own closes inside the same LOOKBACK calendar hours as
    # the other two arms, right-aligned and NaN-padded on the left (a shorter context), not
    # LOOKBACK own bars reaching further back
    closes = _forex()
    contexts = _contexts(closures, closes)
    for row, time in enumerate(contexts.decisions["time"]):
        hours = pd.date_range(end=time, periods=LOOKBACK, freq="1h")
        own = closes[(closes.index >= hours[0]) & (closes.index <= time)].to_numpy()
        trading = contexts.windows[closures.TRADING_HOURS][row]
        assert contexts.own_bars[row] == len(own) == (~contexts.is_closed[row]).sum()
        assert np.isnan(trading[: LOOKBACK - len(own)]).all()
        np.testing.assert_array_equal(trading[LOOKBACK - len(own) :], own)
        # the values it keeps are the forward-filled window's values at the open hours
        is_open = ~contexts.is_closed[row]
        np.testing.assert_array_equal(
            contexts.windows[closures.FORWARD_FILLED][row][is_open], trading[LOOKBACK - len(own) :]
        )
    # a reopening's first decision lost the weekend's 48 hours
    first = contexts.decisions["bar"].to_numpy() == 0
    assert (contexts.own_bars[first] == LOOKBACK - 48).all()


def test_forward_fill_repeats_the_last_close_and_flags_it(closures: ModuleType):
    closes = _forex()
    contexts = _contexts(closures, closes)
    filled = contexts.windows[closures.FORWARD_FILLED]
    for row, time in enumerate(contexts.decisions["time"]):
        hours = pd.date_range(end=time, periods=LOOKBACK, freq="1h")
        last = closes[closes.index <= hours[0]].iloc[-1]
        expected = closes.reindex(hours).ffill().fillna(last).to_numpy()
        np.testing.assert_array_equal(filled[row], expected)
        closed = contexts.is_closed[row]
        np.testing.assert_array_equal(filled[row][1:][closed[1:]], filled[row][:-1][closed[1:]])
    # the decision at a reopening sees the whole weekend repeated at Friday's close
    first = contexts.decisions["bar"].to_numpy() == 0
    assert contexts.is_closed[first][:, -48:].all()
    np.testing.assert_array_equal(
        filled[first][:, -48:],
        np.repeat(contexts.decisions["last_close"][first].to_numpy()[:, None], 48, axis=1),
    )


def test_the_label_is_the_vol_scaled_return_of_the_next_own_bar(closures: ModuleType):
    closes = _forex()
    contexts = _contexts(closures, closes)
    decisions = contexts.decisions
    times = pd.DatetimeIndex(decisions["time"])
    after = closes[pd.DatetimeIndex(decisions["label_time"])].to_numpy()
    before = np.array([closes[closes.index <= time].iloc[-1] for time in times])
    sigma = ewma_volatility(closes, times, span_bars=SPAN)
    np.testing.assert_allclose(decisions["label"], np.log(after / before) / sigma)
    np.testing.assert_allclose(decisions["last_close"], before)
    assert (decisions["label_time"] == times + HOUR).all()


def test_decisions_without_enough_history_are_dropped_and_long_closures_kept(closures: ModuleType):
    closes = _forex()
    long = _contexts(closures, closes, lookback=24 * 9)
    assert len(long.decisions) == 18 - 3
    # every reopening bar comes after a 48-hour closure and is still a decision
    assert (_contexts(closures, closes).decisions.groupby("bar").size() == 6).all()


def test_a_decision_that_is_not_one_hour_before_an_own_close_raises(closures: ModuleType):
    closes = _forex()
    decisions = pd.DataFrame({"asset": ["X"], "time": [pd.Timestamp("2025-01-24 21:00", tz="UTC")]})
    with pytest.raises(ValueError, match="one hour before an own close"):
        closures.closure_contexts(
            {"X": closes}, decisions, lookback=LOOKBACK, span_bars=SPAN, min_own_closes=1
        )
    with pytest.raises(ValueError, match="at least one decision"):
        closures.closure_contexts(
            {"X": closes},
            decisions.iloc[:0],
            lookback=LOOKBACK,
            span_bars=SPAN,
            min_own_closes=1,
        )


def test_no_context_or_label_uses_data_after_its_time(closures: ModuleType):
    closes = _forex()
    cut = pd.Timestamp("2025-01-26 22:00", tz="UTC")
    changed = closes.copy()
    later = changed.index > cut
    changed[later] *= np.exp(np.random.default_rng(9).normal(0, 0.05, later.sum()))
    original, moved = _contexts(closures, closes), _contexts(closures, changed)
    pd.testing.assert_series_equal(original.decisions["time"], moved.decisions["time"])
    is_known = (original.decisions["time"] <= cut).to_numpy()
    for name in closures.ARMS:
        np.testing.assert_array_equal(
            original.windows[name][is_known], moved.windows[name][is_known]
        )
    has_label = (original.decisions["label_time"] <= cut).to_numpy()
    np.testing.assert_array_equal(
        original.decisions["label"][has_label], moved.decisions["label"][has_label]
    )
    assert is_known.sum() > has_label.sum() > 0
    assert not np.allclose(
        original.decisions["label"][~has_label], moved.decisions["label"][~has_label]
    )


def test_select_keeps_the_marked_decisions_and_their_contexts(closures: ModuleType):
    contexts = _contexts(closures, _forex())
    rows = contexts.decisions["bar"].to_numpy() == 1
    chosen = contexts.select(rows)
    assert (chosen.decisions["bar"] == 1).all()
    for name in closures.REPRESENTATIONS:
        np.testing.assert_array_equal(chosen.windows[name], contexts.windows[name][rows])
    np.testing.assert_array_equal(chosen.is_closed, contexts.is_closed[rows])


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=30), min_size=3, max_size=8), st.integers(0, 9))
def test_representation_invariants_hold_for_any_pattern_of_closures(
    closures: ModuleType, gaps: list[int], seed: int
):
    # alternate 30 open hours with a closure of each length in `gaps`
    hours = pd.date_range("2025-01-06", periods=40 * 30 + sum(gaps), freq="1h", tz="UTC")
    keep = np.ones(len(hours), dtype=bool)
    position = 0
    for gap in gaps:
        position += 30
        keep[position : position + gap] = False
        position += gap
    closes = _closes(hours[keep], seed)
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    assert len(decisions) == 3 * sum(gap >= 12 for gap in gaps)
    if decisions.empty:
        return
    contexts = closures.closure_contexts(
        {"X": closes}, decisions.assign(asset="X"), lookback=20, span_bars=5, min_own_closes=1
    )
    filled, missing, is_closed = (
        contexts.windows[closures.FORWARD_FILLED],
        contexts.windows[closures.MISSING],
        contexts.is_closed,
    )
    np.testing.assert_array_equal(np.isnan(missing), is_closed)
    np.testing.assert_array_equal(filled[~is_closed], missing[~is_closed])
    for row in range(len(contexts.decisions)):
        is_open = ~is_closed[row]
        np.testing.assert_array_equal(
            filled[row][is_open],
            contexts.windows[closures.TRADING_HOURS][row][len(is_open) - is_open.sum() :],
        )


# The benchmark input and the closure day


def test_the_benchmark_input_is_the_last_lookback_own_closes(closures: ModuleType):
    # D2: the descriptive arm is the benchmark's input, reaching further back than the
    # calendar hours the three representations share
    closes = _forex()
    contexts = _contexts(closures, closes)
    benchmark = contexts.windows[closures.BENCHMARK_INPUT]
    for row, time in enumerate(contexts.decisions["time"]):
        own = closes[closes.index <= time].to_numpy()[-LOOKBACK:]
        np.testing.assert_array_equal(benchmark[row], own)
    assert np.isfinite(benchmark).all()
    # it holds more own bars than trading hours only, which shares the calendar hours
    assert (contexts.own_bars < LOOKBACK).all()
    np.testing.assert_array_equal(benchmark[:, -1], contexts.windows[closures.TRADING_HOURS][:, -1])


def test_a_decision_needs_lookback_own_closes_as_well_as_lookback_calendar_hours(
    closures: ModuleType,
):
    # 72 calendar hours exist before the first Monday's reopening of a stock-like series
    # open 8 hours a day, but not 72 own closes: no decision until it has both
    hours = pd.date_range("2025-01-06", periods=24 * 28, freq="1h", tz="UTC")
    is_open = (hours.hour >= 8) & (hours.hour < 16)
    closes = _closes(hours[is_open])
    contexts = _contexts(closures, closes)
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    for time in contexts.decisions["time"]:
        assert (closes.index <= time).sum() >= LOOKBACK
    dropped = decisions[~decisions["time"].isin(contexts.decisions["time"])]
    assert len(dropped) > 0
    for time in dropped["time"]:
        assert (closes.index <= time).sum() < LOOKBACK


def test_the_closure_day_is_the_utc_day_the_closure_ends_for_every_target_bar(
    closures: ModuleType,
):
    # a closure that ends at 23:00 UTC: its later target bars fall on the next day but
    # count on the day the closure ends
    hours = pd.date_range("2025-01-06", periods=200, freq="1h", tz="UTC")
    position = np.arange(200)
    closes = _closes(hours[(position < 50) | (position >= 71)])
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    reopening = hours[71]
    assert reopening.hour == 23
    assert list(decisions["time"]) == [reopening, reopening + HOUR, reopening + 2 * HOUR]
    assert (decisions["closure_day"] == reopening.normalize()).all()
    assert decisions["time"].iloc[1].day != reopening.day


def test_a_decision_whose_calendar_hours_hold_no_own_bar_is_dropped(closures: ModuleType):
    # a data gap longer than the window after the asset already had LOOKBACK own closes:
    # the reopening's first decision would have an empty trading-hours context
    hours = pd.date_range("2025-01-06", periods=24 * 20, freq="1h", tz="UTC")
    position = np.arange(len(hours))
    closes = _closes(hours[(position < 200) | (position >= 200 + LOOKBACK + 10)])
    decisions = closures.reopening_decisions(closes, min_closure_hours=12, reopen_bars=3)
    assert len(decisions) == 3
    contexts = _contexts(closures, closes)
    assert list(contexts.decisions["bar"]) == [1, 2]
    assert (contexts.own_bars >= 1).all()
    assert contexts.short_contexts == 1


@pytest.mark.parametrize(("min_own_closes", "bars"), [(1, [1, 2]), (2, [2]), (3, [])])
def test_a_decision_needs_min_own_closes_in_its_trading_hours_context(
    closures: ModuleType, min_own_closes: int, bars: list[int]
):
    # O1: after a gap the reopening's decisions hold 0, 1 and 2 own closes in their window;
    # those under min_own_closes are dropped (from every arm) and counted
    hours = pd.date_range("2025-01-06", periods=24 * 20, freq="1h", tz="UTC")
    position = np.arange(len(hours))
    closes = _closes(hours[(position < 200) | (position >= 200 + LOOKBACK + 10)])
    contexts = _contexts(closures, closes, min_own_closes=min_own_closes)
    assert list(contexts.decisions["bar"]) == bars
    assert contexts.short_contexts == 3 - len(bars)
    assert (contexts.own_bars >= min_own_closes).all()
    for window in contexts.windows.values():
        assert len(window) == len(bars)
    # select keeps the group's count
    chosen = contexts.select(np.zeros(len(bars), dtype=bool))
    assert chosen.short_contexts == contexts.short_contexts


@pytest.mark.parametrize("min_own_closes", [0, LOOKBACK + 1])
def test_min_own_closes_must_be_between_one_and_the_lookback(
    closures: ModuleType, min_own_closes: int
):
    with pytest.raises(ValueError, match=f"between 1 and lookback {LOOKBACK}, got"):
        _contexts(closures, _forex(), min_own_closes=min_own_closes)


def test_min_own_closes_is_a_boundary_on_the_shortest_trading_hours_context(
    closures: ModuleType,
):
    # the shortest context keeps its decision at a threshold of its own length, not above
    shortest = int(_contexts(closures, _forex()).own_bars.min())
    at = _contexts(closures, _forex(), min_own_closes=shortest)
    above = _contexts(closures, _forex(), min_own_closes=shortest + 1)
    assert at.short_contexts == 0
    assert len(at.decisions) == 18
    assert above.short_contexts == 18 - len(above.decisions) > 0


def test_only_decisions_kept_otherwise_count_as_short(closures: ModuleType):
    # with 216 calendar hours the first weekend's 3 decisions lack history: they are dropped
    # for that and never counted as short, even when no context is long enough
    closes = _forex()
    assert len(_contexts(closures, closes, lookback=24 * 9).decisions) == 15
    none = _contexts(closures, closes, lookback=24 * 9, min_own_closes=24 * 9)
    assert none.decisions.empty
    assert none.short_contexts == 15
