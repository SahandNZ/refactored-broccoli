"""The sign book and its net result of 05-rq3-net (its ``lib``): `book_returns`, `net_tables`."""

import dataclasses
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.benchmark import (
    RELEASED_AFTER,
    SCORED,
    Bootstrap,
    ForecastCache,
    StudySettings,
    read_cache,
    scored_decisions,
)
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS

MODELS = ["random-walk", "climatology", "skilled", "sampled"]
FOLDS = list(TEST_FOLDS)
BOOTSTRAP = Bootstrap(level=0.95, n_resamples=200, seed=0, block_hours=24, min_block=4)
SETTINGS = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, BOOTSTRAP)
HOURS = pd.date_range("2025-11-01", periods=30, freq="1h", tz="UTC")
TIMES = HOURS[:5]

type Row = tuple[pd.Timestamp, str, float, float]


def _medians(rows: list[Row], *, held_hours: int = 1) -> pd.DataFrame:
    # decisions whose label ends `held_hours` later (1: a bar every hour, the next grid time)
    frame = pd.DataFrame(rows, columns=["time", "asset", "median", "log_return"])
    time = pd.to_datetime(frame["time"], utc=True)
    return frame.assign(time=time, label_time=time + pd.Timedelta(hours=held_hours))


def _costs(**costs: float) -> pd.Series:
    return pd.Series(costs, dtype=float)


# --- book_returns: one book against a hand computation -------------------------------------


def test_the_book_follows_the_sign_and_pays_each_side(rq3: ModuleType) -> None:
    returns = [0.01, -0.02, 0.03, 0.04, -0.01]
    rows: list[Row] = [
        (TIMES[0], "A", 0.5, returns[0]),
        (TIMES[1], "A", 2.0, returns[1]),
        (TIMES[2], "A", -0.1, returns[2]),
        (TIMES[4], "A", 0.3, returns[4]),
    ]
    book = rq3.book_returns(_medians(rows), _costs(A=10.0), times=TIMES)
    position = np.array([1, 1, -1, 0, 1])
    turnover = np.array([1, 0, 2, 1, 1])
    gross = position * np.expm1(np.array(returns))
    np.testing.assert_allclose(book["gross"], gross)
    np.testing.assert_allclose(book["turnover"], turnover)
    np.testing.assert_allclose(book["net"], gross - turnover * 10.0 / 1e4)
    assert book.index.equals(TIMES)


def test_a_zero_median_holds_nothing(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 0.0, 0.05), (TIMES[1], "A", 0.0, -0.05)]
    book = rq3.book_returns(_medians(rows), _costs(A=5.0), times=TIMES[:2])
    np.testing.assert_allclose(book.to_numpy(), 0.0)


def test_every_asset_of_the_market_has_equal_weight_also_without_forecasts(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 1.0, 0.02), (TIMES[1], "A", 1.0, 0.04)]
    book = rq3.book_returns(_medians(rows), _costs(A=0.0, B=0.0), times=TIMES[:2])
    np.testing.assert_allclose(book["gross"], np.expm1([0.02, 0.04]) / 2)
    np.testing.assert_allclose(book["turnover"], [0.5, 0.0])


def test_each_asset_pays_its_own_cost(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 1.0, 0.0), (TIMES[0], "B", -1.0, 0.0)]
    book = rq3.book_returns(_medians(rows), _costs(A=2.0, B=6.0), times=TIMES[:1])
    assert book["net"].iloc[0] == pytest.approx(-(2.0 + 6.0) / 2 / 1e4)


def test_medians_off_the_decision_grid_are_not_used(rq3: ModuleType) -> None:
    rows: list[Row] = [
        (TIMES[0], "A", 1.0, 0.01),
        (TIMES[1], "A", -1.0, 0.5),
        (TIMES[2], "A", 1.0, 0.02),
    ]
    book = rq3.book_returns(_medians(rows), _costs(A=0.0), times=TIMES[[0, 2]])
    np.testing.assert_allclose(book["gross"], np.expm1([0.01, 0.02]))
    np.testing.assert_allclose(book["turnover"], [1.0, 0.0])


def test_a_position_is_held_for_its_own_bars_without_overlap(rq3: ModuleType) -> None:
    # a market that closes: each label ends 3 grid hours later, so a position opened at
    # hour 0 is held through hours 1 and 2 (no new return, no trade; the decisions there are
    # dropped) and the next opens at 3
    returns = [0.01, 0.02, 0.03, -0.04, 0.05, 0.06]
    signs = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    rows: list[Row] = [(HOURS[k], "A", signs[k], returns[k]) for k in range(6)]
    book = rq3.book_returns(_medians(rows, held_hours=3), _costs(A=10.0), times=HOURS[:6])
    np.testing.assert_allclose(book["gross"], [np.expm1(0.01), 0, 0, -np.expm1(-0.04), 0, 0])
    np.testing.assert_allclose(book["turnover"], [1, 0, 0, 2, 0, 0])
    np.testing.assert_allclose(book["net"], book["gross"] - book["turnover"] * 10.0 / 1e4)


def test_an_ended_position_without_a_new_forecast_is_closed_and_pays(rq3: ModuleType) -> None:
    # the position of hour 4 ends at hour 6, after the last decision: its close is charged
    rows: list[Row] = [(HOURS[0], "A", 1.0, 0.01), (HOURS[4], "A", 1.0, 0.02)]
    book = rq3.book_returns(_medians(rows, held_hours=2), _costs(A=3.0), times=HOURS[:7])
    np.testing.assert_allclose(book["turnover"], [1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(book["gross"], np.expm1([0.01, 0, 0, 0, 0.02, 0, 0]))
    assert book["net"].iloc[-1] == pytest.approx(-3.0 / 1e4)


def test_a_label_not_after_its_decision_raises(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 1.0, 0.01)]
    medians = _medians(rows).assign(label_time=TIMES[0])
    with pytest.raises(ValueError, match="label_time must be after its time"):
        rq3.book_returns(medians, _costs(A=1.0), times=TIMES[:1])


def test_a_missing_realised_return_raises(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 1.0, float("nan"))]
    with pytest.raises(ValueError, match="finite log_return"):
        rq3.book_returns(_medians(rows), _costs(A=1.0), times=TIMES[:2])


def test_the_ends_of_dropped_decisions_are_no_book_times(rq3: ModuleType) -> None:
    # h = 3 with decisions at 0, 1, 2 and 6: 1 and 2 are dropped, so 4 and 5 are no times
    rows: list[Row] = [(HOURS[k], "A", 1.0, 0.01) for k in (0, 1, 2, 6)]
    decisions = _medians(rows, held_hours=3).assign(window="t1")
    times = rq3._net._book_times(decisions, rq3._net._openings(decisions))["time"].tolist()
    assert times == list(HOURS[[0, 1, 2, 3, 6, 9]])


def test_a_book_time_takes_the_fold_of_its_decision_or_of_the_position_it_ends(
    rq3: ModuleType,
) -> None:
    # A decides at 0 (t1, ends at 3) and at 3 (t2, ends at 6): hour 3 is a decision of t2,
    # so the close of the t1 position there counts in t2; the last end, 6, counts in t2
    rows: list[Row] = [(HOURS[0], "A", 1.0, 0.01), (HOURS[3], "A", 1.0, 0.02)]
    decisions = _medians(rows, held_hours=3).assign(window=["t1", "t2"])
    times = rq3._net._book_times(decisions, rq3._net._openings(decisions))
    assert times["time"].tolist() == list(HOURS[[0, 3, 6]])
    assert times["window"].tolist() == ["t1", "t2", "t2"]


def test_an_asset_without_a_cost_raises(rq3: ModuleType) -> None:
    rows: list[Row] = [(TIMES[0], "A", 1.0, 0.01), (TIMES[0], "C", 1.0, 0.01)]
    with pytest.raises(ValueError, match=r"assets \['C'\] have no cost per side"):
        rq3.book_returns(_medians(rows), _costs(A=1.0), times=TIMES[:1])


# --- book_returns and the break-even cost: properties --------------------------------------

returns = st.floats(-0.2, 0.2, allow_nan=False)
medians = st.floats(-1.0, 1.0, allow_nan=False)

type Book = tuple[pd.DataFrame, list[str], pd.DatetimeIndex]


@st.composite
def books(draw: st.DrawFn) -> Book:
    n_times = draw(st.integers(1, 30))
    n_assets = draw(st.integers(1, 3))
    rows: list[Row] = []
    for step in range(n_times):
        for asset in range(n_assets):
            if draw(st.booleans()):
                rows.append((HOURS[step], f"A{asset}", draw(medians), draw(returns)))
    times = HOURS[:n_times]
    return _medians(rows), [f"A{asset}" for asset in range(n_assets)], times


@settings(max_examples=60, deadline=None)
@given(book=books(), cost=st.floats(0.0, 50.0))
def test_net_is_gross_minus_turnover_times_one_cost(
    rq3: ModuleType, book: Book, cost: float
) -> None:
    frame, assets, times = book
    result = rq3.book_returns(frame, pd.Series(cost, index=assets), times=times)
    np.testing.assert_allclose(
        result["net"], result["gross"] - result["turnover"] * cost / 1e4, atol=1e-15
    )
    assert ((result["turnover"] >= 0) & (result["turnover"] <= 2 + 1e-12)).all()


@settings(max_examples=60, deadline=None)
@given(book=books())
def test_zero_cost_leaves_net_equal_to_gross(rq3: ModuleType, book: Book) -> None:
    frame, assets, times = book
    result = rq3.book_returns(frame, pd.Series(0.0, index=assets), times=times)
    np.testing.assert_allclose(result["net"], result["gross"])


@settings(max_examples=60, deadline=None)
@given(book=books(), cost=st.floats(0.1, 50.0), factor=st.floats(0.0, 10.0))
def test_costs_scale_linearly_and_leave_gross_and_turnover_alone(
    rq3: ModuleType, book: Book, cost: float, factor: float
) -> None:
    frame, assets, times = book
    costs = pd.Series(np.linspace(cost, 2 * cost, len(assets)), index=assets)
    base = rq3.book_returns(frame, costs, times=times)
    scaled = rq3.book_returns(frame, costs * factor, times=times)
    np.testing.assert_allclose(scaled[["gross", "turnover"]], base[["gross", "turnover"]])
    np.testing.assert_allclose(
        scaled["gross"] - scaled["net"], factor * (base["gross"] - base["net"]), atol=1e-15
    )


@settings(max_examples=60, deadline=None)
@given(book=books())
def test_trading_at_the_break_even_cost_leaves_a_zero_mean_net_return(
    rq3: ModuleType, book: Book
) -> None:
    frame, assets, times = book
    free = rq3.book_returns(frame, pd.Series(0.0, index=assets), times=times)
    summary = rq3._net._summary(free.reset_index(names="time"))
    if free["turnover"].sum() == 0:
        assert np.isnan(summary["break_even_bps"])
        return
    at_break_even = rq3.book_returns(
        frame, pd.Series(summary["break_even_bps"], index=assets), times=times
    )
    assert at_break_even["net"].mean() == pytest.approx(0.0, abs=1e-12)


@settings(max_examples=40, deadline=None)
@given(book=books(), cut=st.integers(0, 29), shift=st.floats(-1.0, 1.0))
def test_changing_the_future_leaves_the_past_book_unchanged(
    rq3: ModuleType, book: Book, cut: int, shift: float
) -> None:
    frame, assets, times = book
    costs = pd.Series(3.0, index=assets)
    is_future = frame["time"] > HOURS[cut]
    changed = frame.copy()
    changed.loc[is_future, "median"] = -changed.loc[is_future, "median"] + shift
    changed.loc[is_future, "log_return"] = changed.loc[is_future, "log_return"] * 3
    before = rq3.book_returns(frame, costs, times=times)
    after = rq3.book_returns(changed, costs, times=times)
    past = before.index <= HOURS[cut]
    pd.testing.assert_frame_equal(before[past], after[past])


# --- _summary and the one-bar delay ----------------------------------------------------------


def _summary_of(
    rq3: ModuleType, gross: list[float], net: list[float], turnover: list[float], hours: int = 1
) -> dict[str, float]:
    times = pd.date_range("2025-11-01", periods=len(gross), freq=f"{hours}h", tz="UTC")
    book = pd.DataFrame({"time": times, "gross": gross, "net": net, "turnover": turnover})
    return rq3._net._summary(book)


def test_the_summary_gives_cost_break_even_and_annualised_sharpe(rq3: ModuleType) -> None:
    summary = _summary_of(rq3, [0.002, 0.0, 0.001], [0.0015, 0.0, 0.0005], [1.0, 0.0, 1.0], 4)
    assert summary["times"] == 3
    assert summary["turnover"] == pytest.approx(2 / 3)
    assert summary["break_even_bps"] == pytest.approx(0.001 / (2 / 3) * 1e4)
    assert summary["cost_bps"] == pytest.approx(0.001 / 3 / (2 / 3) * 1e4)
    net = pd.Series([0.0015, 0.0, 0.0005])
    expected = net.mean() / net.std(ddof=1) * np.sqrt(365.25 * 24 / 4)
    assert summary["net_sharpe"] == pytest.approx(expected)


def test_net_mean_is_turnover_times_break_even_minus_cost(rq3: ModuleType) -> None:
    summary = _summary_of(rq3, [0.003, -0.001, 0.002], [0.002, -0.0015, 0.0], [1.0, 0.5, 2.0])
    gap = summary["turnover"] * (summary["break_even_bps"] - summary["cost_bps"]) / 1e4
    assert summary["net_mean"] == pytest.approx(gap)


def test_a_book_that_never_trades_has_no_break_even(rq3: ModuleType) -> None:
    summary = _summary_of(rq3, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    assert np.isnan(summary["break_even_bps"])
    assert np.isnan(summary["cost_bps"])
    assert summary["net_sharpe"] == 0.0


def test_the_delay_moves_each_median_to_the_next_own_bar(rq3: ModuleType) -> None:
    # A decides at hours 0, 1 and 3 (closed at 2, so the next bar after 1 is 3); the median
    # of hour 0 trades at 1 with the label of 1, the one of 1 at 3, the one of 3 is dropped
    medians = _medians(
        [(HOURS[0], "A", 1.0, 0.01), (HOURS[1], "A", -1.0, 0.02), (HOURS[3], "A", 1.0, 0.03)]
    ).assign(horizon=1)
    next_bars = pd.DataFrame(
        {"time": HOURS[[0, 1, 3]], "asset": "A", "next_time": HOURS[[1, 3, 4]]}
    )
    delayed = rq3._net._delayed(medians, next_bars).sort_values("time", ignore_index=True)
    assert delayed["time"].tolist() == list(HOURS[[1, 3]])
    assert delayed["median"].tolist() == pytest.approx([1.0, -1.0])
    assert delayed["log_return"].tolist() == pytest.approx([0.02, 0.03])


# --- net_tables on the synthetic cache -----------------------------------------------------


@pytest.fixture(name="cache", scope="module")
def _cache(quarters_cache_folder: Path) -> ForecastCache:
    return read_cache(quarters_cache_folder)


@pytest.fixture(name="tables", scope="module")
def _tables(rq3: ModuleType, cache: ForecastCache) -> dict[str, pd.DataFrame]:
    return rq3.net_tables(cache, MODELS, SETTINGS)


def _cell(tables: dict[str, pd.DataFrame], model: str, market: str, horizon: int) -> pd.Series:
    cells = tables["cells"]
    chosen = cells[
        (cells["model"] == model) & (cells["market"] == market) & (cells["horizon"] == horizon)
    ]
    assert len(chosen) == 1
    return chosen.iloc[0]


def _with_costs(cache: ForecastCache, factor: float) -> ForecastCache:
    assets = cache.assets.assign(cost_per_side_bps=cache.assets["cost_per_side_bps"] * factor)
    return dataclasses.replace(cache, assets=assets)


def _released(cache: ForecastCache, model: str, date: str) -> ForecastCache:
    models = cache.models.copy()
    models.loc[models["model"] == model, "release_date"] = pd.Timestamp(date, tz="UTC")
    return dataclasses.replace(cache, models=models)


def _models_only(table: pd.DataFrame) -> pd.DataFrame:
    # the rows of the models, without the buy-and-hold reference
    return table[table["kind"] != "reference"]


def test_tables_have_one_row_per_cell_and_fold(tables: dict[str, pd.DataFrame]) -> None:
    # 4 models x 3 markets x 3 horizons, plus buy and hold at horizon 1 in every market
    assert len(tables["cells"]) == 4 * 3 * 3 + 3
    assert len(tables["quarters"]) == (4 * 3 * 3 + 3) * 4
    assert set(tables["quarters"]["window"]) == {"t1", "t2", "t3", "t4"}
    assert (tables["cells"]["quarters"] == 4).all()
    assert (tables["cells"]["post_release_quarters"] == 4).all()
    kinds = tables["cells"].groupby("model")["kind"].first()
    assert kinds.to_dict() == {
        "climatology": "baseline",
        "random-walk": "baseline",
        "sampled": "zero-shot",
        "skilled": "zero-shot",
        "buy-and-hold": "reference",
    }


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_the_break_even_matches_a_direct_computation(
    cache: ForecastCache, tables: dict[str, pd.DataFrame], horizon: int
) -> None:
    labels = cache.labels[
        (cache.labels["asset"] == "BTCUSDT") & (cache.labels["horizon"] == horizon)
    ]
    labels = labels[labels["window"].isin([fold.name for fold in FOLDS])]
    sign = dict(zip(labels["time"], np.sign(labels["label"]), strict=True))
    realised = dict(zip(labels["time"], np.expm1(labels["log_return"]), strict=True))
    ends = dict(zip(labels["time"], labels["label_time"], strict=True))
    # the book's times: every decision time and the end of every position; a position opens
    # at a decision time once the last one has ended, and is flat after its end (the last
    # close is charged too)
    opening_ends: set[pd.Timestamp] = set()
    held_end: pd.Timestamp | None = None
    for time in sorted(sign):
        if held_end is None or time >= held_end:
            opening_ends.add(ends[time])
            held_end = ends[time]
    book_times = sorted(set(labels["time"]) | opening_ends)
    gross, turnover, position, held_until, effective = 0.0, 0.0, 0.0, None, 0
    for time in book_times:
        previous, step = position, 0.0
        if held_until is not None and time >= held_until:
            position, held_until = 0.0, None
        if held_until is None and time in sign:
            position, held_until = sign[time], ends[time]
            step = position * realised[time]
        gross += step
        turnover += abs(position - previous)
        effective += int(step != 0 or position != previous)
    times = len(book_times)
    cell = _cell(tables, "skilled", "crypto", horizon)
    assert cell["times"] == times
    assert cell["effective_times"] == (times if horizon == 1 else effective)
    if horizon > 1:
        assert effective < times
    assert cell["gross_mean"] == pytest.approx(gross / times)
    assert cell["turnover"] == pytest.approx(turnover / times)
    assert cell["break_even_bps"] == pytest.approx(gross / turnover * 1e4)
    assert cell["cost_bps"] == pytest.approx(5.0)
    assert cell["net_mean"] == pytest.approx((gross - turnover * 5.0 / 1e4) / times)


def test_the_cost_paid_is_each_markets_cost(tables: dict[str, pd.DataFrame]) -> None:
    skilled = tables["cells"][tables["cells"]["model"] == "skilled"]
    expected = {"crypto": 5.0, "forex": 0.4, "stock": 1.0}
    for market, cost in expected.items():
        np.testing.assert_allclose(skilled.loc[skilled["market"] == market, "cost_bps"], cost)
    table = tables["costs"].set_index("market")
    assert table["cost_per_side_bps"].to_dict() == pytest.approx(expected)
    assert table["assets"].to_dict() == {"crypto": 1, "forex": 1, "stock": 1}


def test_every_scored_cell_has_intervals_around_its_estimates(
    tables: dict[str, pd.DataFrame],
) -> None:
    cells = _models_only(tables["cells"])
    scored = cells[cells["status"] == SCORED]
    # the sampled stock book at h = 24 has 40 effective times, fewer than 5 blocks of its 14
    assert len(scored) == 17
    short = cells[cells["status"] != SCORED].set_index(["model", "market", "horizon"])
    assert short.loc[("sampled", "stock", 24), "status"] == (
        "not scored: 40 effective book times, fewer than 5 blocks of 14"
    )
    assert (scored["net_low"] <= scored["net_mean"]).all()
    assert (scored["net_mean"] <= scored["net_high"]).all()
    # the sampled model holds its short for every 4-hour label at h = 4, so its book trades
    # only at the ends of the windows: some draws have no trade and no break-even
    has_interval = scored["break_even_low"].notna()
    assert (scored.loc[~has_interval, "model"] == "sampled").all()
    assert has_interval[scored["model"] == "skilled"].all()
    within = scored[has_interval]
    assert (within["break_even_low"] <= within["break_even_high"]).all()
    skilled = within[within["model"] == "skilled"]
    assert (skilled["break_even_low"] <= skilled["break_even_bps"]).all()
    assert (skilled["break_even_bps"] <= skilled["break_even_high"]).all()
    assert (scored["block"] >= 1).all()
    np.testing.assert_allclose(scored["blocks"], scored["effective_times"] / scored["block"])
    assert (scored["effective_times"] <= scored["times"]).all()
    assert scored["p_value"].between(0, 1).all()


def test_zero_costs_leave_net_equal_to_gross(rq3: ModuleType, cache: ForecastCache) -> None:
    free = rq3.net_tables(_with_costs(cache, 0.0), ["skilled"], SETTINGS)["cells"]
    free = free[free["status"] == SCORED]
    np.testing.assert_allclose(free["net_mean"], free["gross_mean"])
    np.testing.assert_allclose(free["net_sharpe"], free["gross_sharpe"])
    np.testing.assert_allclose(free["cost_bps"], 0.0, atol=1e-9)


def test_doubled_costs_double_the_cost_and_keep_the_break_even(
    rq3: ModuleType, cache: ForecastCache, tables: dict[str, pd.DataFrame]
) -> None:
    doubled = _models_only(rq3.net_tables(_with_costs(cache, 2.0), ["skilled"], SETTINGS)["cells"])
    base = tables["cells"][tables["cells"]["model"] == "skilled"].reset_index(drop=True)
    np.testing.assert_allclose(doubled["break_even_bps"], base["break_even_bps"])
    np.testing.assert_allclose(doubled["turnover"], base["turnover"])
    np.testing.assert_allclose(
        doubled["gross_mean"] - doubled["net_mean"], 2 * (base["gross_mean"] - base["net_mean"])
    )


def test_a_model_with_a_zero_median_never_trades(tables: dict[str, pd.DataFrame]) -> None:
    # the synthetic climatology is a zero-centred normal (the real one is not)
    cells = tables["cells"][tables["cells"]["model"].isin(["random-walk", "climatology"])]
    np.testing.assert_allclose(cells[["gross_mean", "net_mean", "turnover"]], 0.0)
    assert cells["break_even_bps"].isna().all()
    assert (cells["status"] == "not scored: never trades").all()
    assert cells[["p_value", "block", "break_even_low"]].isna().all().all()


def test_the_seeds_of_a_sampling_model_are_averaged_into_one_short_position(
    cache: ForecastCache, tables: dict[str, pd.DataFrame]
) -> None:
    decisions, _ = scored_decisions(cache, "sampled", windows=["t1", "t2", "t3", "t4"])
    np.testing.assert_allclose(decisions["q50"], (0.1 - 0.3) / 2, atol=1e-6)
    assert decisions.duplicated(["time", "asset", "horizon"]).sum() == 0
    cell = _cell(tables, "sampled", "crypto", 1)
    labels = cache.labels[(cache.labels["asset"] == "BTCUSDT") & (cache.labels["horizon"] == 1)]
    is_fourth = pd.DatetimeIndex(labels["time"]).hour % 4 == 0
    labels = labels[labels["window"].isin(["t1", "t2", "t3", "t4"]) & is_fourth]
    assert cell["gross_mean"] * cell["times"] == pytest.approx(
        -np.expm1(labels["log_return"]).sum()
    )
    assert cell["turnover"] * cell["times"] == pytest.approx(2 * len(labels))


def test_buy_and_hold_is_long_every_asset_from_its_first_bar_to_its_last(
    cache: ForecastCache, tables: dict[str, pd.DataFrame]
) -> None:
    # one asset per market: long at every 1-bar label of the scored folds; the synthetic
    # windows are apart, so each opens once and closes once (8 sides over the 4 folds)
    hold = tables["cells"][tables["cells"]["model"] == "buy-and-hold"]
    assert set(hold["horizon"]) == {1}
    assert (hold["status"] == SCORED).all()
    labels = cache.labels[(cache.labels["asset"] == "BTCUSDT") & (cache.labels["horizon"] == 1)]
    labels = labels[labels["window"].isin(["t1", "t2", "t3", "t4"])]
    crypto = hold[hold["market"] == "crypto"].iloc[0]
    assert crypto["gross_mean"] * crypto["times"] == pytest.approx(
        np.expm1(labels["log_return"]).sum()
    )
    assert crypto["turnover"] * crypto["times"] == pytest.approx(8.0)
    assert crypto["post_release_quarters"] == 4


def test_buy_and_hold_over_contiguous_folds_enters_and_closes_once(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    # two assets on contiguous folds: A has every bar, B starts later and misses bars (a
    # closed market: its 1-bar label ends at its next bar); each enters once and closes once
    times = pd.date_range("2025-11-01", periods=8, freq="1h", tz="UTC")
    a_rows = pd.DataFrame({"time": times[:-1], "label_time": times[1:], "asset": "A"})
    b_times = times[[2, 3, 6]]
    b_rows = pd.DataFrame({"time": b_times[:-1], "label_time": b_times[1:], "asset": "B"})
    labels = pd.concat([a_rows, b_rows], ignore_index=True).assign(
        horizon=1,
        log_return=0.01,
        label=0.0,
        window=lambda frame: np.where(frame["time"] < times[4], "t1", "t2"),
    )
    assets = pd.DataFrame({"asset": ["A", "B"], "market": "crypto", "cost_per_side_bps": 0.0})
    small = dataclasses.replace(cache, labels=labels, assets=assets)
    hold = rq3._net._buy_and_hold(small, ["t1", "t2"])
    book = rq3._net._books(hold, small)
    assert book["turnover"].sum() == pytest.approx(2.0)
    assert book["gross"].sum() == pytest.approx((7 + 2) * np.expm1(0.01) / 2)


def test_the_delay_check_loses_the_skill_and_skips_a_sparse_model(
    tables: dict[str, pd.DataFrame],
) -> None:
    # the skilled median carries half of its own label, not of the next bar's; the sampled
    # model decides every 4 hours, so its next own bar has no decision (as Kronos)
    delayed = tables["delayed"]
    assert set(delayed["model"]) == {"random-walk", "climatology", "skilled"}
    skilled = delayed[delayed["model"] == "skilled"].set_index(["market", "horizon"])
    on_time = tables["cells"][tables["cells"]["model"] == "skilled"].set_index(
        ["market", "horizon"]
    )
    assert (skilled["gross_mean"] < on_time.loc[skilled.index, "gross_mean"]).all()


def test_the_delay_check_needs_labels_at_horizon_one(rq3: ModuleType, cache: ForecastCache) -> None:
    labels = cache.labels[cache.labels["horizon"] != 1]
    with pytest.raises(ValueError, match=r"needs labels at horizon 1, got \[4, 24\]"):
        rq3.net_tables(dataclasses.replace(cache, labels=labels), ["skilled"], SETTINGS)


def test_a_zero_shot_model_trades_only_after_its_release(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    later = rq3.net_tables(_released(cache, "skilled", "2026-02-01"), ["skilled"], SETTINGS)
    assert set(_models_only(later["quarters"])["window"]) == {"t2", "t3", "t4"}
    cell = later["cells"].iloc[0]
    assert cell["quarters"] == 3
    assert cell["post_release_quarters"] == 3


def test_a_partial_quarter_is_not_a_post_release_quarter(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    later = rq3.net_tables(_released(cache, "skilled", "2026-02-01 12:00"), ["skilled"], SETTINGS)
    cells = _models_only(later["cells"])
    assert (cells["quarters"] == 3).all()
    assert (cells["post_release_quarters"] == 2).all()


def test_a_baseline_has_no_release_filter(tables: dict[str, pd.DataFrame]) -> None:
    climatology = tables["quarters"][tables["quarters"]["model"] == "climatology"]
    assert set(climatology["window"]) == {"t1", "t2", "t3", "t4"}


def test_errors_name_what_is_missing(rq3: ModuleType, cache: ForecastCache) -> None:
    with pytest.raises(ValueError, match=r"models \['nope'\], windows \[\]"):
        rq3.net_tables(cache, ["nope"], SETTINGS)


def test_a_model_released_after_the_scored_folds_is_not_scored(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    late = _released(cache, "skilled", "2027-01-01")
    cells = rq3.net_tables(late, ["skilled", "sampled"], SETTINGS)["cells"]
    skilled = cells[cells["model"] == "skilled"]
    assert len(skilled) == 9
    assert (skilled["status"] == RELEASED_AFTER).all()
    assert skilled["p_value"].isna().all()
    assert (skilled["post_release_quarters"] == 0).all()


def test_no_scored_cell_at_all_raises_with_the_reasons(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    late = _released(cache, "skilled", "2027-01-01")
    message = (
        r"no zero-shot model of rq3 could be scored on \['t1', 't2', 't3', 't4'\]: 9 cells not "
        r"scored: released after the scored folds"
    )
    with pytest.raises(ValueError, match=message):
        rq3.net_tables(late, ["climatology", "skilled"], SETTINGS)


def test_a_missing_fold_raises(rq3: ModuleType, cache: ForecastCache) -> None:
    settings = dataclasses.replace(SETTINGS, scored_folds=(LEAKAGE_FOLDS[0],))
    with pytest.raises(ValueError, match=r"windows \['l1'\]"):
        rq3.net_tables(cache, ["skilled"], settings)


def test_a_rerun_gives_the_same_tables(
    rq3: ModuleType, cache: ForecastCache, tables: dict[str, pd.DataFrame]
) -> None:
    again = rq3.net_tables(cache, MODELS, SETTINGS)
    for name, table in tables.items():
        pd.testing.assert_frame_equal(again[name], table)


def test_a_book_too_short_for_the_automatic_block_is_not_scored(rq3: ModuleType) -> None:
    # for h > 1 only the times with a return or a trade count: an h = 3 book that opens a
    # position every 3 times (one return and one trade at that time) has 1 effective time
    # per position; a time with only a closing trade counts too; at h = 1 all count
    def book(gross: list[float], turnover: list[float]) -> pd.DataFrame:
        times = pd.date_range("2025-11-01", periods=len(gross), freq="1h", tz="UTC")
        frame = pd.DataFrame({"time": times, "gross": gross, "turnover": turnover})
        return frame.assign(net=frame["gross"] - frame["turnover"] * 1e-4, overlap=0)

    held = book([0.01, 0.0, 0.0] * 10, [1.0, 0.0, 0.0] * 10)
    assert rq3.is_effective(held, horizon=3).sum() == 10
    assert rq3.is_effective(held, horizon=1).sum() == 30
    assert rq3._net._status(held, horizon=3, effective=10)[0] == (
        "not scored: 10 effective book times, fewer than 11 for the block"
    )
    closed = book([0.01, 0.0, 0.0] * 10 + [0.0], [1.0, 0.0, 0.0] * 10 + [1.0])
    assert rq3.is_effective(closed, horizon=3).sum() == 11
    status, block = rq3._net._status(closed, horizon=3, effective=11)
    # the block floor is 2, and 11 times hold 5 blocks of 2
    assert block == 2
    assert status == SCORED
    never = rq3._net._status(book([0.0] * 20, [0.0] * 20), horizon=1, effective=20)
    assert never == ("not scored: never trades", None)
    # a net return without spread has no bootstrap error to test with
    flat = book([0.01] * 20, [0.0] * 19 + [1.0]).assign(net=0.01)
    assert rq3._net._status(flat, horizon=1, effective=20) == (
        "not scored: no spread in the net return",
        2,
    )


def test_the_overlap_counts_the_later_effective_times_a_position_spans(rq3: ModuleType) -> None:
    # two assets on different clocks at h = 2: A opens at hours 0 and 4, B at 1 and 5; each
    # position spans one later effective time, the other asset's opening; its own close, at
    # its label end, does not count
    rows: list[Row] = [
        (HOURS[0], "A", 1.0, 0.01),
        (HOURS[4], "A", -1.0, 0.02),
        (HOURS[1], "B", 1.0, 0.03),
        (HOURS[5], "B", -1.0, 0.01),
    ]
    decisions = _medians(rows, held_hours=2).assign(window="t1")
    openings = pd.concat([rq3._net._openings(group) for _, group in decisions.groupby("asset")])
    times = rq3._net._book_times(decisions, openings)
    book = rq3.book_returns(decisions, _costs(A=1.0, B=1.0), times=pd.DatetimeIndex(times["time"]))
    overlap = rq3._net._overlap(book.reset_index(names="time"), openings, horizon=2)
    assert overlap == 1
    alone = decisions[decisions["asset"] == "A"]
    one_clock = rq3.book_returns(alone, _costs(A=1.0), times=HOURS[[0, 2, 4, 6]])
    assert rq3._net._overlap(one_clock.reset_index(names="time"), alone, horizon=2) == 0


def test_a_release_on_the_first_day_of_a_fold_counts_that_fold(
    rq3: ModuleType, cache: ForecastCache
) -> None:
    released = _released(cache, "skilled", "2026-02-01")
    assert rq3._net._post_release_quarters(released, "skilled", TEST_FOLDS) == 3
    later = _released(cache, "skilled", "2026-02-02")
    assert rq3._net._post_release_quarters(later, "skilled", TEST_FOLDS) == 2
