from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.benchmark import (
    Bootstrap,
    NullRule,
    StudySettings,
    detectable_cells,
    power_tables,
    powered_threshold,
    read_cache,
    threshold_shares,
)
from src.benchmark._power import _long_run_variances, minimum_detectable_effects, power_table
from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS
from src.stats import andrews_lags, block_indices, diebold_mariano, newey_west_variance

GAINS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]


def _cell(seed, *, times=800, noise=0.3, model="m", market="crypto", horizon=1):
    # one rq1 cell of the "times" table: a baseline CRPS near 1 and a model whose loss differs
    # from it by noise of standard deviation `noise` (plus a real edge of 10 %)
    rng = np.random.default_rng(seed)
    baseline = 1.0 + 0.1 * rng.standard_normal(times)
    return pd.DataFrame({
        "model": model, "market": market, "horizon": horizon, "baseline": "climatology",
        "window": np.repeat(["t1", "t2", "t3", "t4"], times // 4),
        "time": pd.date_range("2025-11-01", periods=times, freq="1h", tz="UTC"),
        "decisions": rng.integers(1, 4, times), "crps_baseline": baseline,
        "crps_model": 0.9 * baseline + noise * rng.standard_normal(times),
    })  # fmt: skip


def _rule(gains=GAINS, *, min_winning_quarters=3) -> NullRule:
    return NullRule(
        min_winning_quarters=min_winning_quarters, main_family_min_quarters=3,
        skill_resamples=50, improvements_pct=tuple(gains), power_target=0.8, mde_horizon=1,
        threshold_grid_pct=(0.0, 10.0), power_share=0.8, power_share_sensitivity=(0.5, 0.9),
        mde_threshold_pct=2.0,
    )  # fmt: skip


def _power(times, gains=GAINS, *, n_resamples=200, block=24, seed=0, min_winning_quarters=3):
    # the cells' timing as `null_tables` gives it: `block` decision times, the overlap of an
    # h-bar label is h - 1 times, one family unless the table names its own
    bootstrap = Bootstrap(0.95, n_resamples, seed, block_hours=24, min_block=4)
    settings = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, bootstrap)
    rule = _rule(gains, min_winning_quarters=min_winning_quarters)
    if "family" not in times.columns:
        times = times.assign(family="main")
    return power_table(times.assign(block=block, overlap=times["horizon"] - 1), settings, rule)


# --- power_table --------------------------------------------------------------------------


def test_without_an_edge_the_detection_rate_is_the_one_sided_level() -> None:
    power = _power(_cell(0), [0.0], n_resamples=4000, block=1, min_winning_quarters=0)
    assert power["false_positive_level"].tolist() == pytest.approx([0.05])
    assert power["detection_rate"].iloc[0] == pytest.approx(0.05, abs=0.015)
    with_quarters = _power(_cell(0), [0.0], n_resamples=4000, block=1)
    assert with_quarters["detection_rate"].iloc[0] <= power["detection_rate"].iloc[0]


def test_the_real_models_own_edge_is_taken_out() -> None:
    power = _power(_cell(1, noise=0.05))
    assert power["detection_rate"].iloc[0] < 0.05
    assert power["detection_rate"].iloc[-1] == 1.0


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    noise=st.floats(0.05, 2.0),
    horizon=st.sampled_from([1, 4, 24]),
    block=st.sampled_from([1, 5, 24]),
)
def test_the_detection_rate_never_falls_as_the_gain_grows(seed, noise, horizon: int, block) -> None:
    power = _power(_cell(seed, noise=noise, horizon=horizon), n_resamples=60, block=block)
    rates = power["detection_rate"].to_numpy()
    assert (np.diff(rates) >= 0).all()
    assert ((rates >= 0) & (rates <= 1)).all()


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_a_large_gain_is_always_detected(horizon: int) -> None:
    power = _power(_cell(2, horizon=horizon), [0.0, 50.0])
    assert power["detection_rate"].tolist()[-1] == 1.0


def test_the_planted_model_has_exactly_the_planted_gain() -> None:
    power = _power(_cell(3))
    improvement = 100 * (1 - power["planted_crps"] / power["baseline_crps"])
    np.testing.assert_allclose(improvement, GAINS, atol=1e-9)
    np.testing.assert_allclose(power["baseline_crps"], _cell(3)["crps_baseline"].mean())


def test_the_power_uses_the_cells_block_and_overlap() -> None:
    power = _power(_cell(3, horizon=24), [0.0], n_resamples=10, block=7)
    assert power[["block", "overlap"]].iloc[0].tolist() == [7, 23]


@pytest.mark.parametrize("overlap", [0, 3, 23])
def test_each_resamples_variance_and_lags_are_those_of_the_dm_test(overlap: int) -> None:
    # the vectorised long-run variance of the power analysis is the one diebold_mariano uses
    rng = np.random.default_rng(overlap)
    noise = rng.standard_normal((5, 300))
    drawn = noise + 0.6 * np.roll(noise, 1, axis=1)
    variances, lags = _long_run_variances(drawn, overlap)
    for row, variance, lag in zip(drawn, variances, lags, strict=True):
        assert lag == min(max(overlap, andrews_lags(row)), len(row) - 1)
        centred = (row - row.mean())[None, :]
        assert variance == pytest.approx(newey_west_variance(centred, lags=lag)[0], rel=1e-9)
        test = diebold_mariano(row, np.zeros_like(row), overlap=overlap)
        assert test["lags"] == lag


def test_without_an_edge_the_planted_resamples_follow_the_dm_test() -> None:
    # gain 0, block 1, no quarter condition: the detection rate is the share of resamples
    # whose one-sided DM p-value (from diebold_mariano) is below alpha
    times = _cell(11, times=200).assign(family="main")
    power = _power(times, [0.0], n_resamples=50, block=1, min_winning_quarters=0)

    indices = block_indices(200, np.random.default_rng([0, 0]), n_resamples=50, block=1)
    baseline = times["crps_baseline"].to_numpy()
    difference = times["crps_model"].to_numpy() - baseline
    centred = difference - difference.mean()
    detected = [
        diebold_mariano(baseline[rows] + centred[rows], baseline[rows], overlap=0)["p_less"] < 0.05
        for rows in indices
    ]
    assert power["detection_rate"].iloc[0] == pytest.approx(np.mean(detected))


def test_the_draws_are_reproducible_with_the_seed() -> None:
    times = _cell(4, noise=1.0)
    first, again = _power(times, seed=7), _power(times, seed=7)
    pd.testing.assert_frame_equal(first, again)
    other = _power(times, seed=8)
    assert not np.allclose(first["detection_rate"], other["detection_rate"])


def test_the_holm_threshold_uses_the_number_of_cells_of_the_family() -> None:
    alone = _power(_cell(5, noise=1.0))
    pair = _power(pd.concat([_cell(5, noise=1.0), _cell(5, noise=1.0, market="forex")]))
    assert pair["false_positive_level"].iloc[0] == pytest.approx(0.05 / 2)
    crypto = pair[pair["market"] == "crypto"]
    assert (crypto["detection_rate"].to_numpy() <= alone["detection_rate"].to_numpy() + 0.05).all()
    assert crypto["detection_rate"].sum() < alone["detection_rate"].sum()
    assert len(pair) == 2 * len(GAINS)
    assert set(pair.columns) == {
        "model", "market", "horizon", "family", "baseline", "times", "quarters", "block", "overlap",
        "improvement_pct", "baseline_crps", "planted_crps", "detection_rate",
        "false_positive_level",
    }  # fmt: skip
    # a cell of the late family counts only the late cells: alone in its family, m = 1
    families = pd.concat(
        [_cell(5, noise=1.0).assign(family="main"),
         _cell(5, noise=1.0, market="forex").assign(family="late")]
    )  # fmt: skip
    split = _power(families).drop_duplicates("market").set_index("market")
    assert split["false_positive_level"].to_dict() == pytest.approx({"crypto": 0.05, "forex": 0.05})


def test_each_cell_draws_its_own_resamples() -> None:
    # two identical cells in different markets: the same seed, but each its own stream
    pair = _power(pd.concat([_cell(5, noise=1.0), _cell(5, noise=1.0, market="forex")]))
    rates = pair.pivot(index="improvement_pct", columns="market", values="detection_rate")
    assert not np.allclose(rates["crypto"], rates["forex"])


@pytest.mark.parametrize("gains", [[], [0.5, 1.0], [0.0, 2.0, 1.0], [0.0, 1.0, 1.0]], ids=str)
def test_a_grid_that_does_not_increase_from_zero_raises(gains) -> None:
    with pytest.raises(ValueError, match="improvements must increase from 0"):
        _power(_cell(6), gains)


def test_a_cell_shorter_than_the_block_raises() -> None:
    with pytest.raises(ValueError, match="block must be between 1 and the 20 values, got 24"):
        _power(_cell(6, times=20))


def test_power_on_the_rq1_cells_of_the_synthetic_cache(quarters_cache_folder: Path) -> None:
    bootstrap = Bootstrap(0.95, 100, 0, block_hours=24, min_block=4)
    settings = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, bootstrap)
    rule = _rule()
    models = ["random-walk", "climatology", "skilled", "sampled"]
    tables = power_tables(read_cache(quarters_cache_folder), models, settings, rule)
    times = tables["times"]
    assert len(times.groupby(["model", "market", "horizon"])) == len(tables["cells"])
    assert times.groupby(["model", "market", "horizon"])["time"].is_monotonic_increasing.all()
    cells = tables["cells"].set_index(["model", "market", "horizon"])
    by_cell = times.groupby(["model", "market", "horizon"])
    assert (by_cell.size() == cells["times"].loc[by_cell.size().index]).all()
    assert (by_cell["decisions"].sum() == cells["decisions"].loc[by_cell.size().index]).all()
    power = tables["power"]
    assert len(power) == len(cells) * len(GAINS)
    assert (power.loc[power["improvement_pct"] == 0, "detection_rate"] < 0.05).all()
    blocks = power.drop_duplicates(["model", "market", "horizon"]).set_index(cells.index.names)
    pd.testing.assert_series_equal(blocks["block"], cells["block"].loc[blocks.index])
    pd.testing.assert_series_equal(blocks["overlap"], cells["overlap"].loc[blocks.index])
    pd.testing.assert_series_equal(blocks["family"], cells["family"].loc[blocks.index])
    assert len(tables["mde"]) == len(cells)


# --- minimum_detectable_effects -----------------------------------------------------------


def _rates(rates, gains=(0.0, 1.0, 2.0, 5.0)):
    return pd.DataFrame({
        "model": "m", "market": "crypto", "horizon": 1, "family": "main", "quarters": 4,
        "baseline": "climatology", "times": 100,
        "improvement_pct": list(gains), "detection_rate": rates, "false_positive_level": 0.01,
    })  # fmt: skip


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ([0.0, 0.5, 0.9, 1.0], 1.75),
        ([0.0, 0.2, 0.8, 1.0], 2.0),
        ([0.0, 0.0, 0.0, 0.8], 5.0),
        ([0.9, 1.0, 1.0, 1.0], 0.0),
        ([0.0, 0.6, 0.7, 1.0], 2.0 + 3.0 / 3),
    ],
)
def test_the_mde_interpolates_linearly_to_the_first_gain_reaching_the_target(
    rates, expected
) -> None:
    mde = minimum_detectable_effects(_rates(rates), target=0.8)
    assert mde["mde_pct"].tolist() == pytest.approx([expected])
    assert mde["is_reached"].tolist() == [True]
    assert mde["zero_rate"].tolist() == pytest.approx([rates[0]])


def test_a_target_never_reached_has_no_mde() -> None:
    mde = minimum_detectable_effects(_rates([0.0, 0.1, 0.5, 0.79]), target=0.8)
    assert np.isnan(mde["mde_pct"].iloc[0])
    assert mde["is_reached"].tolist() == [False]
    assert mde["largest_pct"].tolist() == [5.0]


def test_the_mde_sorts_the_gains_first() -> None:
    rows = _rates([0.0, 0.5, 0.9, 1.0]).iloc[[3, 1, 0, 2]]
    mde = minimum_detectable_effects(rows, target=0.8)
    assert mde["mde_pct"].tolist() == pytest.approx([1.75])
    assert mde["zero_rate"].tolist() == [0.0]


def test_the_mde_of_a_planted_power_table_lies_on_the_grid() -> None:
    power = _power(pd.concat([_cell(9, noise=0.5), _cell(9, noise=3.0, model="noisy")]))
    mde = minimum_detectable_effects(power, target=0.8).set_index("model")
    assert mde.loc["m", "mde_pct"] < mde.loc["noisy", "mde_pct"]
    assert ((mde["mde_pct"] >= 0) & (mde["mde_pct"] <= 50)).all()
    assert mde["is_reached"].all()


# --- threshold_shares and powered_threshold --------------------------------------------------


def _mde(values, horizons=None):
    # an mde table: NaN means never reached
    values = np.array(values, dtype=float)
    return pd.DataFrame({
        "model": [f"m{i}" for i in range(len(values))], "market": "crypto",
        "horizon": horizons if horizons is not None else [1] * len(values),
        "mde_pct": values, "is_reached": ~np.isnan(values), "largest_pct": 10.0, "quarters": 4,
    })  # fmt: skip


def test_only_cells_with_enough_quarters_can_be_detected() -> None:
    # a cell with 2 quarters can never win 3: its detection rate is 0 at any gain, so the
    # power shares leave it out
    two = _cell(12, noise=0.05).assign(window=np.repeat(["t1", "t2"], 400))
    power = _power(two, [0.0, 50.0])
    assert power["detection_rate"].tolist() == [0.0, 0.0]
    assert power["quarters"].tolist() == [2, 2]
    mde = _mde([1.0, np.nan]).assign(quarters=[4, 2])
    assert detectable_cells(mde, min_winning_quarters=3)["model"].tolist() == ["m0"]


GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def test_threshold_shares_count_reached_cells_at_or_below_each_threshold() -> None:
    shares = threshold_shares(_mde([0.4, 1.0, 3.0, np.nan], [1, 1, 4, 4]), thresholds=GRID)
    assert shares.columns.tolist() == ["threshold_pct", "cells", "share", "share_1h", "share_4h"]
    by_threshold = shares.set_index("threshold_pct")
    assert by_threshold.loc[0.0, "share"] == 0.0
    assert by_threshold.loc[0.5, "share"] == pytest.approx(0.25)
    assert by_threshold.loc[1.0, "share"] == pytest.approx(0.5)
    assert by_threshold.loc[10.0, "share"] == pytest.approx(0.75)
    assert by_threshold.loc[10.0, "share_1h"] == 1.0
    assert by_threshold.loc[10.0, "share_4h"] == pytest.approx(0.5)
    assert (shares["cells"] == 4).all()


@given(values=st.lists(st.one_of(st.floats(0, 12), st.just(float("nan"))), min_size=1, max_size=30))
def test_the_share_never_falls_as_the_threshold_grows(values) -> None:
    shares = threshold_shares(_mde(values), thresholds=GRID)
    for column in ["share", "share_1h"]:
        assert (np.diff(shares[column].to_numpy()) >= 0).all()
        assert ((shares[column] >= 0) & (shares[column] <= 1)).all()


@pytest.mark.parametrize("thresholds", [[], [1.0, 0.5], [0.0, 1.0, 1.0]], ids=str)
def test_thresholds_that_do_not_increase_raise(thresholds) -> None:
    with pytest.raises(ValueError, match="thresholds must be increasing"):
        threshold_shares(_mde([1.0]), thresholds=thresholds)


def _shares(values):
    return pd.DataFrame({"threshold_pct": GRID[: len(values)], "share_1h": values})


@pytest.mark.parametrize(
    ("values", "share", "expected"),
    [
        ([0.0, 0.2, 0.5, 0.8, 0.85, 0.88, 0.9, 0.9], 0.8, 1.5),
        ([0.0, 0.2, 0.5, 0.8, 0.85, 0.88, 0.9, 0.9], 0.9, 5.0),
        ([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 0.5, 0.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], 0.8, 10.0),
        ([0.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1.0, 1.0),
    ],
    ids=["k80", "k90", "first-point", "last-point", "all-cells"],
)
def test_the_threshold_is_the_smallest_reaching_the_share(values, share, expected) -> None:
    assert powered_threshold(_shares(values), share=share, horizon=1) == expected


def test_a_share_never_reached_has_no_threshold() -> None:
    values = [0.0, 0.2, 0.5, 0.7, 0.75, 0.79, 0.79, 0.79]
    assert np.isnan(powered_threshold(_shares(values), share=0.8, horizon=1))


def test_the_threshold_sorts_the_grid_first() -> None:
    shares = _shares([0.0, 0.2, 0.5, 0.8, 0.85, 0.88, 0.9, 0.9]).iloc[::-1]
    assert powered_threshold(shares, share=0.8, horizon=1) == 1.5


def test_the_threshold_uses_the_cells_of_its_horizon_only() -> None:
    mde = _mde([0.5, 0.5, 8.0, 8.0], [1, 1, 24, 24])
    shares = threshold_shares(mde, thresholds=GRID)
    assert powered_threshold(shares, share=0.8, horizon=1) == 0.5
    assert powered_threshold(shares, share=0.8, horizon=24) == 10.0


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5])
def test_a_share_outside_zero_to_one_raises(share) -> None:
    with pytest.raises(ValueError, match="share must be in"):
        powered_threshold(_shares([0.0, 1.0]), share=share, horizon=1)
