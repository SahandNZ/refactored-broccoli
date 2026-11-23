import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.benchmark import Bootstrap, NullRule, StudySettings, read_cache
from src.benchmark._cache import QUANTILE_COLUMNS
from src.benchmark._decisions import (
    NO_VALID_DECISION,
    RELEASED_AFTER,
    SCORED,
    cell_timing,
    decision_points,
    scored_decisions,
)
from src.benchmark._scores import best_baseline, decision_losses, full_quarters, null_tables
from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS, quantile_crps
from src.models import QUANTILE_LEVELS
from src.stats import diebold_mariano, politis_white_block

MODELS = ["random-walk", "climatology", "skilled", "sampled"]
DEVELOPMENT = ["d1", "d2", "d3", "d4"]
TEST = ["t1", "t2", "t3", "t4"]
LEVELS = np.asarray(QUANTILE_LEVELS)
BOOTSTRAP = Bootstrap(level=0.95, n_resamples=100, seed=0, block_hours=24, min_block=4)
SETTINGS = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, BOOTSTRAP)
RULE = NullRule(
    min_winning_quarters=3, main_family_min_quarters=3, skill_resamples=400,
    improvements_pct=(0.0, 10.0), power_target=0.8, mde_horizon=1,
    threshold_grid_pct=(0.0, 10.0), power_share=0.8, power_share_sensitivity=(0.5, 0.9),
    mde_threshold_pct=2.0,
)  # fmt: skip
finite = st.floats(-1e3, 1e3, allow_nan=False)


# --- CRPS from nine quantiles -------------------------------------------------------------


@given(center=finite, target=finite)
def test_crps_of_a_degenerate_forecast_is_the_absolute_error(center, target) -> None:
    quantiles = np.full((1, len(LEVELS)), center)
    crps = quantile_crps(quantiles, LEVELS, np.array([target]))
    assert crps[0] == pytest.approx(abs(target - center), rel=1e-9, abs=1e-9)


@given(values=st.lists(finite, min_size=9, max_size=9), target=finite)
def test_crps_is_never_negative(values, target) -> None:
    quantiles = np.sort(np.array(values))[None, :]
    assert quantile_crps(quantiles, LEVELS, np.array([target]))[0] >= 0


# --- decision_losses ----------------------------------------------------------------------


def test_decision_losses_have_one_row_per_decision_with_its_market(cache_folder: Path) -> None:
    cache = read_cache(cache_folder)
    decisions, invalid = scored_decisions(cache, "skilled", windows=["l1", "d1", "t1"])
    losses = decision_losses(decisions)
    assert list(losses.columns) == [
        "time",
        "asset",
        "horizon",
        "window",
        "label_time",
        "market",
        "crps",
    ]
    released = cache.labels["time"] > pd.Timestamp("2025-01-01", tz="UTC")
    assert len(losses) == released.sum()
    markets = dict(zip(cache.assets["asset"], cache.assets["market"], strict=True))
    assert (losses["asset"].map(markets) == losses["market"]).all()
    assert (losses["crps"] >= 0).all()
    assert invalid.empty


def test_decision_losses_score_the_mean_of_the_seeds_quantiles(cache_folder: Path) -> None:
    cache = read_cache(cache_folder)
    frame = cache.forecasts("sampled")
    mean = frame.groupby(["time", "asset", "horizon"])[list(QUANTILE_COLUMNS)].mean()
    decisions, _ = scored_decisions(cache, "sampled", windows=["d1", "t1"])
    losses = decision_losses(decisions).set_index(["time", "asset", "horizon"])
    labels = cache.labels.set_index(["time", "asset", "horizon"])["label"].loc[losses.index]
    expected = quantile_crps(mean.loc[losses.index].to_numpy(), LEVELS, labels.to_numpy())
    np.testing.assert_allclose(losses["crps"], expected, rtol=1e-6)
    by_seed = frame.pivot_table(index=["time", "asset", "horizon"], columns="seed", values="q50")
    assert not np.allclose(by_seed[0], by_seed[1])


def _copy(cache_folder, tmp_path):
    folder = tmp_path / "cache"
    shutil.copytree(cache_folder, folder)
    return folder


def test_an_invalid_seed_drops_its_decision_and_is_counted(
    cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(cache_folder, tmp_path)
    path = folder / "forecasts" / "sampled.parquet"
    forecasts = pd.read_parquet(path)
    keys = ["time", "asset", "horizon"]
    first, second = np.flatnonzero(forecasts["time"] > pd.Timestamp("2025-03-02", tz="UTC"))[:2]
    forecasts.loc[first, "q50"] = np.nan
    forecasts.loc[second, "q30"] = forecasts.loc[second, "q90"] + 1.0
    forecasts.to_parquet(path)
    cache = read_cache(folder)
    decisions, invalid = scored_decisions(cache, "sampled", windows=["l1", "d1", "t1"])
    broken = forecasts.loc[[first, second], keys]
    assert decisions.merge(broken, on=keys).empty
    assert int(invalid["invalid"].sum()) == 2
    clean, _ = scored_decisions(read_cache(cache_folder), "sampled", windows=["l1", "d1", "t1"])
    assert len(clean) - len(decisions) == 2
    assert np.isfinite(decision_losses(decisions)["crps"]).all()


def test_an_asset_without_market_raises(cache_folder: Path, tmp_path: Path) -> None:
    folder = _copy(cache_folder, tmp_path)
    assets = pd.read_csv(folder / "assets.csv")
    assets[assets["asset"] != "AAPL"].to_csv(folder / "assets.csv", index=False)
    with pytest.raises(ValueError, match=r"assets \['AAPL'\] of climatology"):
        scored_decisions(read_cache(folder), "climatology", windows=["d1"])


# --- best_baseline ------------------------------------------------------------------------


def _hour(hours):
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")


def _losses(rows):
    # times in hours from 2025-01-01; a label of one bar ends at the next hour
    frame = pd.DataFrame(rows, columns=["time", "asset", "window", "crps"])
    frame["time"] = _hour(frame["time"])
    frame = frame.assign(
        horizon=1, market="crypto", label_time=frame["time"] + pd.Timedelta(hours=1)
    )
    return frame[["time", "asset", "horizon", "window", "label_time", "market", "crps"]]


def test_the_best_baseline_is_chosen_on_the_choice_windows_only() -> None:
    good_later = _losses([(0, "A", "d1", 2.0), (1, "A", "t1", 0.1)])
    good_now = _losses([(0, "A", "d1", 1.0), (1, "A", "t1", 5.0)])
    best = best_baseline(
        {"good-later": good_later, "good-now": good_now},
        choice_windows=["d1"], scored_windows=["t1"],
    )  # fmt: skip
    assert best["baseline"].tolist() == ["good-now"]
    assert best["crps"].tolist() == [5.0]


def test_in_development_each_fold_is_chosen_on_the_other_folds() -> None:
    first = _losses([(0, "A", "d1", 1.0), (1, "A", "d2", 3.0), (2, "A", "d3", 3.0)])
    second = _losses([(0, "A", "d1", 2.0), (1, "A", "d2", 1.0), (2, "A", "d3", 1.0)])
    best = best_baseline(
        {"first": first, "second": second},
        choice_windows=["d1", "d2", "d3"], scored_windows=["d1", "d2", "d3"],
    )  # fmt: skip
    chosen = best.set_index("window")["baseline"].to_dict()
    assert chosen == {"d1": "second", "d2": "second", "d3": "second"}
    first = _losses([(0, "A", "d1", 9.0), (1, "A", "d2", 1.0), (2, "A", "d3", 1.0)])
    second = _losses([(0, "A", "d1", 0.0), (1, "A", "d2", 2.0), (2, "A", "d3", 2.0)])
    best = best_baseline(
        {"first": first, "second": second},
        choice_windows=["d1", "d2", "d3"], scored_windows=["d1", "d2", "d3"],
    )  # fmt: skip
    chosen = best.set_index("window")["baseline"].to_dict()
    assert chosen == {"d1": "first", "d2": "second", "d3": "second"}


def test_a_choice_label_that_reaches_into_the_scored_window_is_not_used() -> None:
    # the d1 decisions at time 9 have labels ending at 11, inside t1 (from 10): only the
    # decisions at time 0 count, where "late" is better
    early = _losses([(0, "A", "d1", 3.0), (9, "A", "d1", 0.0), (10, "A", "t1", 1.0)])
    late = _losses([(0, "A", "d1", 2.0), (9, "A", "d1", 5.0), (10, "A", "t1", 1.0)])
    windows = {"choice_windows": ["d1"], "scored_windows": ["t1"]}
    assert best_baseline({"early": early, "late": late}, **windows)["baseline"].tolist() == [
        "early"
    ]
    for frame in (early, late):
        frame.loc[1, "label_time"] = _hour(11)
    assert best_baseline({"early": early, "late": late}, **windows)["baseline"].tolist() == ["late"]


def test_the_best_baseline_averages_over_assets_at_each_time_first() -> None:
    # "dense" wins on the mean over decisions, "sparse" on the mean over times
    dense = _losses(
        [
            (0, "A", "d1", 1.0),
            (0, "B", "d1", 1.0),
            (0, "C", "d1", 1.0),
            (1, "A", "d1", 4.0),
            (5, "A", "t1", 1.0),
        ]
    )
    sparse = _losses(
        [
            (0, "A", "d1", 2.0),
            (0, "B", "d1", 2.0),
            (0, "C", "d1", 2.0),
            (1, "A", "d1", 2.0),
            (5, "A", "t1", 1.0),
        ]
    )
    best = best_baseline(
        {"dense": dense, "sparse": sparse}, choice_windows=["d1"], scored_windows=["t1"]
    )
    assert best["baseline"].tolist() == ["sparse"]


def test_the_best_baseline_is_chosen_on_decisions_every_baseline_forecast() -> None:
    full = _losses([(0, "A", "d1", 1.0), (1, "A", "d1", 1.0), (9, "A", "t1", 1.0)])
    partial = _losses([(0, "A", "d1", 1.5), (2, "A", "d1", 0.0), (9, "A", "t1", 1.0)])
    best = best_baseline(
        {"full": full, "partial": partial}, choice_windows=["d1"], scored_windows=["t1"]
    )
    assert set(best["baseline"]) == {"full"}


def test_a_cell_without_shared_choice_decisions_raises() -> None:
    first = _losses([(0, "A", "d1", 1.0), (5, "A", "t1", 1.0)])
    second = _losses([(1, "A", "d1", 1.0), (5, "A", "t1", 1.0)])
    with pytest.raises(ValueError, match="only 0 of 1 market and horizon cells of t1"):
        best_baseline(
            {"first": first, "second": second}, choice_windows=["d1"], scored_windows=["t1"]
        )


# --- null_tables --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables(quarters_cache_folder):
    return null_tables(read_cache(quarters_cache_folder), MODELS, SETTINGS, RULE)


def test_null_tables_have_one_cell_per_model_market_and_horizon(
    tables: dict[str, pd.DataFrame],
) -> None:
    cells = tables["cells"]
    assert len(cells) == 2 * 3 * 3
    assert set(cells["model"]) == {"skilled", "sampled"}
    assert set(cells["baseline"]) == {"climatology"}
    assert (cells["status"] == SCORED).all()
    assert (cells["quarters"] == 4).all()
    assert (cells["invalid"] == 0).all()
    assert len(tables["quarters"]) == 2 * 3 * 3 * 4
    skilled = cells[cells["model"] == "skilled"]
    assert (skilled["winning_quarters"] == 4).all()
    assert (skilled["dm_statistic"] < 0).all()


def test_the_overlap_and_the_block_follow_the_labels_and_the_loss_difference(
    tables: dict[str, pd.DataFrame],
) -> None:
    cells = tables["cells"].set_index(["model", "horizon"]).sort_index()
    assert cells.loc[("skilled", 24), "spacing_hours"].tolist() == [1.0] * 3
    assert cells.loc[("skilled", 24), "overlap"].tolist() == [23] * 3
    assert cells.loc[("skilled", 1), "overlap"].tolist() == [0] * 3
    assert cells.loc[("sampled", 4), "spacing_hours"].tolist() == [4.0] * 3
    assert cells.loc[("sampled", 4), "overlap"].tolist() == [0] * 3
    assert cells.loc[("sampled", 24), "overlap"].tolist() == [5] * 3
    assert (cells["dm_lags"] >= cells["overlap"]).all()
    times = tables["times"]
    for (model, market, horizon), cell in times.groupby(["model", "market", "horizon"]):
        difference = (cell["crps_model"] - cell["crps_baseline"]).to_numpy()
        overlap = int(cell["overlap"].iloc[0])
        expected = max(politis_white_block(difference), 4 * overlap, 2)
        row = (
            tables["cells"].set_index(["model", "market", "horizon"]).loc[(model, market, horizon)]
        )
        assert row["block"] == expected


def test_the_skill_and_its_interval(tables: dict[str, pd.DataFrame]) -> None:
    cells = tables["cells"]
    np.testing.assert_allclose(
        cells["skill"], 100 * (1 - cells["model_crps"] / cells["baseline_crps"])
    )
    assert (cells["skill_low"] <= cells["skill"]).all()
    assert (cells["skill"] <= cells["skill_high"]).all()
    skilled = cells[cells["model"] == "skilled"]
    assert (skilled["skill_low"] > 0).all()
    sampled = cells[cells["model"] == "sampled"]
    assert ((sampled["skill_low"] < 0) | (sampled["skill_high"] < 0)).all()


def test_the_skill_interval_is_reproducible_and_each_cell_has_its_own_draws(
    quarters_cache_folder: Path, tables: dict[str, pd.DataFrame]
) -> None:
    again = null_tables(read_cache(quarters_cache_folder), MODELS, SETTINGS, RULE)["cells"]
    pd.testing.assert_frame_equal(again, tables["cells"])
    other = dataclasses.replace(SETTINGS, bootstrap=dataclasses.replace(BOOTSTRAP, seed=1))
    moved = null_tables(read_cache(quarters_cache_folder), MODELS, other, RULE)["cells"]
    assert not np.allclose(moved["skill_low"], tables["cells"]["skill_low"])


def test_a_model_with_three_quarters_after_its_release_is_in_the_main_family(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    assert full_quarters(pd.Timestamp("2025-11-01", tz="UTC"), TEST_FOLDS) == 4
    assert full_quarters(pd.Timestamp("2025-11-02", tz="UTC"), TEST_FOLDS) == 3
    assert full_quarters(pd.Timestamp("2026-04-14", tz="UTC"), TEST_FOLDS) == 2
    folder = _copy(quarters_cache_folder, tmp_path)
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == "sampled", "release_date"] = "2025-12-01"
    models.loc[models["model"] == "skilled", "release_date"] = "2026-04-14"
    models.to_csv(folder / "models.csv", index=False)
    tables = null_tables(read_cache(folder), MODELS, SETTINGS, RULE)
    families = tables["cells"].groupby("model")["family"].unique()
    assert families["sampled"].tolist() == ["main"]
    assert families["skilled"].tolist() == ["late"]
    assert set(tables["times"].groupby("model")["family"].first()) == {"main", "late"}


def test_a_grid_models_decisions_of_one_grid_point_are_one_point(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    # Kronos-like: every asset's first decision at or after a grid point; a market that is
    # closed at the grid point decides later, but the decision belongs to the same point
    folder = _copy(quarters_cache_folder, tmp_path)
    path = folder / "forecasts" / "sampled.parquet"
    frame = pd.read_parquet(path)
    is_stock = frame["asset"] == "AAPL"
    frame.loc[is_stock, "time"] = frame.loc[is_stock, "time"] + pd.Timedelta(hours=2)
    frame.to_parquet(path)
    cache = read_cache(folder)
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-11-01 00:00", "2025-11-01 02:00", "2025-11-01 05:00"], utc=True
            ),
            "window": "t1",
        }
    )
    points = decision_points(rows, cache, "sampled")
    assert points.tolist() == [pd.Timestamp("2025-11-01 00:00", tz="UTC")] * 2 + [
        pd.Timestamp("2025-11-01 04:00", tz="UTC")
    ]
    assert decision_points(rows, cache, "skilled").equals(rows["time"])
    shifted = null_tables(cache, MODELS, SETTINGS, RULE)["cells"]
    stock = shifted[(shifted["model"] == "sampled") & (shifted["market"] == "stock")]
    assert (stock["times"] == 200).all()


def test_a_model_with_a_subset_of_decisions_is_compared_on_that_subset(
    tables: dict[str, pd.DataFrame], quarters_cache_folder: Path
) -> None:
    cache = read_cache(quarters_cache_folder)
    climatology = decision_losses(scored_decisions(cache, "climatology", windows=TEST)[0])
    on_subset = climatology[climatology["time"].dt.hour % 4 == 0]
    by_time = on_subset.groupby(["market", "horizon", "time"])["crps"].mean()
    expected = by_time.groupby(["market", "horizon"]).agg(["mean", "size"])
    cells = tables["cells"][tables["cells"]["model"] == "sampled"].set_index(["market", "horizon"])
    np.testing.assert_allclose(cells["baseline_crps"], expected.loc[cells.index, "mean"])
    assert (cells["times"] == expected.loc[cells.index, "size"]).all()
    assert (cells["decisions"] == on_subset.groupby(["market", "horizon"]).size()).all()


def test_the_cell_crps_and_the_test_use_the_same_mean_over_assets_per_time(
    tables: dict[str, pd.DataFrame], quarters_cache_folder: Path
) -> None:
    cache = read_cache(quarters_cache_folder)
    model = decision_losses(scored_decisions(cache, "skilled", windows=TEST)[0])
    baseline = decision_losses(scored_decisions(cache, "climatology", windows=TEST)[0])
    keep = (model["market"] == "forex") & (model["horizon"] == 4)
    model_mean = model[keep].groupby("time")["crps"].mean()
    keep = (baseline["market"] == "forex") & (baseline["horizon"] == 4)
    baseline_mean = baseline[keep].groupby("time")["crps"].mean().loc[model_mean.index]
    expected = diebold_mariano(model_mean.to_numpy(), baseline_mean.to_numpy(), overlap=3)
    cells = tables["cells"]
    cell = cells[(cells["model"] == "skilled") & (cells["market"] == "forex")]
    cell = cell[cell["horizon"] == 4].iloc[0]
    assert cell["times"] == len(model_mean)
    assert cell["dm_statistic"] == pytest.approx(expected["statistic"])
    assert cell["p_value"] == pytest.approx(expected["p_less"])
    assert cell["dm_lags"] == expected["lags"]
    assert cell["model_crps"] == pytest.approx(model_mean.mean())
    assert cell["baseline_crps"] == pytest.approx(baseline_mean.mean())
    assert np.sign(cell["model_crps"] - cell["baseline_crps"]) == np.sign(cell["dm_statistic"])


def test_a_model_released_inside_the_window_is_scored_after_its_release(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(quarters_cache_folder, tmp_path)
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == "skilled", "release_date"] = "2026-03-01"
    models.to_csv(folder / "models.csv", index=False)
    tables = null_tables(read_cache(folder), MODELS, SETTINGS, RULE)
    cells = tables["cells"]
    skilled = cells[cells["model"] == "skilled"]
    assert (skilled["quarters"] == 2).all()
    assert (skilled["winning_quarters"] == 2).all()


def test_a_model_released_after_the_scored_folds_is_reported_not_scored(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(quarters_cache_folder, tmp_path)
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == "sampled", "release_date"] = "2027-01-01"
    models.to_csv(folder / "models.csv", index=False)
    tables = null_tables(read_cache(folder), MODELS, SETTINGS, RULE)
    late = tables["cells"][tables["cells"]["model"] == "sampled"]
    assert len(late) == 9
    assert (late["status"] == RELEASED_AFTER).all()
    assert (late["times"] == 0).all()
    assert late["p_value"].isna().all()
    assert "sampled" not in set(tables["times"]["model"])


def test_a_cell_with_fewer_than_five_blocks_is_untested(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    # a cell needs at least 5 blocks of decision points (owner's rule, src.stats
    # has_enough_blocks): the sampled model kept at one decision time is untested and left
    # out of Holm, with no test and no skill interval; the skilled model is still tested
    folder = _copy(quarters_cache_folder, tmp_path)
    path = folder / "forecasts" / "sampled.parquet"
    frame = pd.read_parquet(path)
    frame[frame["time"] == pd.Timestamp("2025-11-01", tz="UTC")].to_parquet(path)
    tables = null_tables(read_cache(folder), MODELS, SETTINGS, RULE)
    cells = tables["cells"].set_index("model")
    assert (
        cells.loc["sampled", "status"] == "untested: 1 decision times, fewer than 5 blocks of 2"
    ).all()
    assert cells.loc["sampled", "p_value"].isna().all()
    assert cells.loc["sampled", "skill_low"].isna().all()
    assert (cells.loc["skilled", "status"] == SCORED).all()
    assert set(tables["times"]["model"]) == {"skilled"}


def test_no_scored_cell_at_all_raises_with_the_reasons(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(quarters_cache_folder, tmp_path)
    for model in ("skilled", "sampled"):
        path = folder / "forecasts" / f"{model}.parquet"
        frame = pd.read_parquet(path)
        frame[frame["time"] == pd.Timestamp("2025-11-01", tz="UTC")].to_parquet(path)
    message = (
        r"no zero-shot model of rq1 could be scored on \['t1', 't2', 't3', 't4'\]: 18 cells "
        r"untested: 1 decision times, fewer than 5 blocks of 2"
    )
    with pytest.raises(ValueError, match=message):
        null_tables(read_cache(folder), MODELS, SETTINGS, RULE)


def test_a_cell_without_any_decision_is_reported(
    quarters_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(quarters_cache_folder, tmp_path)
    path = folder / "forecasts" / "skilled.parquet"
    forecasts = pd.read_parquet(path)
    forecasts[forecasts["asset"] != "AAPL"].to_parquet(path)
    cells = null_tables(read_cache(folder), MODELS, SETTINGS, RULE)["cells"]
    stock = cells[(cells["model"] == "skilled") & (cells["market"] == "stock")]
    assert (stock["status"] == NO_VALID_DECISION).all()
    assert len(stock) == 3


def test_null_tables_refuse_unknown_models_and_windows(cache_folder: Path) -> None:
    cache = read_cache(cache_folder)
    with pytest.raises(ValueError, match=r"models \['nope'\], windows \['d2', 'd3', 'd4'"):
        null_tables(cache, [*MODELS, "nope"], SETTINGS, RULE)


# --- cell_timing: the overlap of the labels sets the lags and the block -------------------


def _sessions(hours_by_asset, horizon):
    # decisions at each own close of every asset; a label ends `horizon` own closes later
    hourly = pd.date_range("2025-11-03", "2026-02-01", freq="1h", tz="UTC", inclusive="left")
    parts = []
    for asset, (first, last) in hours_by_asset.items():
        closes = hourly[(hourly.hour >= first) & (hourly.hour <= last) & (hourly.day_of_week < 5)]
        parts.append(
            pd.DataFrame(
                {"asset": asset, "time": closes[:-horizon], "label_time": closes[horizon:]}
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_the_lags_count_the_later_decisions_inside_a_label() -> None:
    timing = cell_timing(_sessions({"BTC": (0, 23)}, 24), BOOTSTRAP)
    assert timing["dm_lags"] == 24
    assert timing["block"] == 24
    assert cell_timing(_sessions({"BTC": (0, 23)}, 1), BOOTSTRAP)["dm_lags"] == 1


def test_a_pooled_market_with_three_sessions_gets_the_lags_of_its_overlap() -> None:
    # Tokyo, Frankfurt and New York closes pooled: the spacing is one hour but a 24-bar label
    # spans more than three days, so dozens of later decisions overlap it
    pooled = _sessions({"7203.T": (1, 6), "SAP.DE": (8, 16), "AAPL": (15, 21)}, 24)
    timing = cell_timing(pooled, BOOTSTRAP)
    assert timing["spacing_hours"] == 1.0
    times = pd.DatetimeIndex(pooled["time"].drop_duplicates().sort_values())
    later = times.searchsorted(pd.DatetimeIndex(pooled["label_time"])) - times.searchsorted(
        pd.DatetimeIndex(pooled["time"]), side="right"
    )
    assert timing["dm_lags"] == 1 + int(np.ceil(np.median(later)))
    assert timing["dm_lags"] > 40
    assert timing["block"] == timing["dm_lags"]


def test_sparse_decisions_get_one_lag_and_the_shortest_block() -> None:
    times = pd.date_range("2025-11-03", periods=13, freq="167h", tz="UTC")
    timing = cell_timing(
        pd.DataFrame({"time": times, "label_time": times + pd.Timedelta(hours=24)}), BOOTSTRAP
    )
    assert (timing["dm_lags"], timing["block"], timing["status"]) == (1, 4, SCORED)
    few = cell_timing(pd.DataFrame({"time": times[:3], "label_time": times[:3]}), BOOTSTRAP)
    assert few["status"] == "not scored: 3 decision times, fewer than the bootstrap block 4"
