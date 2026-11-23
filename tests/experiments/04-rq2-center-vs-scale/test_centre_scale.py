"""The centre and scale scores and the verdict of 04-rq2-center-vs-scale (its ``lib``)."""

import math
import shutil
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.benchmark import (
    QUANTILE_COLUMNS,
    RELEASED_AFTER,
    SCORED,
    Bootstrap,
    ForecastCache,
    StudySettings,
    read_cache,
)
from src.benchmark._cache import write_forecasts
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS, Fold, quantile_crps
from src.evaluate import rank_ic as spearman
from src.models import QUANTILE_LEVELS
from src.stats import BLOCK_OVERLAP_FLOOR, bootstrap_mean_test, holm, politis_white_block

TEST = ["t1", "t2", "t3", "t4"]
LEVELS = np.asarray(QUANTILE_LEVELS)
BELOW = [f"below_{column}" for column in QUANTILE_COLUMNS]
BOOTSTRAP = Bootstrap(level=0.95, n_resamples=200, seed=0, block_hours=24, min_block=4)
CELL = ["model", "market", "horizon"]

type Tables = dict[str, pd.DataFrame]


def _settings(
    folds: tuple[Fold, ...] = TEST_FOLDS, bootstrap: Bootstrap = BOOTSTRAP
) -> StudySettings:
    return StudySettings(DEVELOPMENT_FOLDS, tuple(folds), 0.05, bootstrap)


def _rule(
    rq2: ModuleType,
    *,
    scale: str = "ewma-normal",
    centre: str = "climatology",
    coverage_tolerance: float = 0.1,
) -> Any:
    return rq2.CentreScaleRule(scale, centre, 3, 3, coverage_tolerance, 1)


def _tables(
    rq2: ModuleType,
    folder: Path,
    models: list[str],
    *,
    scale: str = "ewma-normal",
    folds: tuple[Fold, ...] = TEST_FOLDS,
) -> Tables:
    rule = _rule(rq2, scale=scale)
    return rq2.centre_scale_tables(read_cache(folder), models, _settings(folds), rule=rule)


def _copy(folder: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "cache"
    shutil.copytree(folder, copy)
    return copy


def _set_release(folder: Path, model: str, date: str) -> None:
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == model, "release_date"] = date
    models.to_csv(folder / "models.csv", index=False)


def _verdict(rq2: ModuleType, tables: Tables, *, coverage_tolerance: float = 0.1) -> pd.DataFrame:
    cells, quarters = tables["cells"], tables["quarters"]
    return rq2.centre_scale_verdict(
        cells[cells["kind"] == "zero-shot"], quarters[quarters["kind"] == "zero-shot"],
        alpha=0.05, rule=_rule(rq2, coverage_tolerance=coverage_tolerance),
    )  # fmt: skip


# --- a small cache written directly ---------------------------------------------------------


def _forecast_rows(keys: pd.DataFrame, quantiles: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(quantiles, columns=list(QUANTILE_COLUMNS))
    return pd.concat([keys.reset_index(drop=True).assign(seed=0), frame], axis=1)


def _small_cache(
    folder: Path, labels: pd.DataFrame, forecasts: dict[str, np.ndarray], *, every_hours: int = 0
) -> Any:
    # `labels` has time, asset, horizon, window and label; every model forecasts every label,
    # "model" is zero-shot (deciding on a grid of `every_hours`, 0 for none) and the others
    # are baselines; all assets are crypto
    folder.mkdir(parents=True)
    table = labels.assign(
        label_time=labels["time"] + pd.to_timedelta(labels["horizon"], unit="h"),
        log_return=labels["label"], sigma=1.0,
    )  # fmt: skip
    table.to_parquet(folder / "labels.parquet")
    names = list(forecasts)
    pd.DataFrame({
        "model": names, "kind": ["zero-shot" if name == "model" else "baseline" for name in names],
        "release_date": ["2025-01-01" if name == "model" else "" for name in names],
        "pretraining_cutoff": "",
        "every_hours": [every_hours if name == "model" else 0 for name in names],
    }).to_csv(folder / "models.csv", index=False)  # fmt: skip
    assets = sorted(set(labels["asset"]))
    pd.DataFrame({"asset": assets, "market": "crypto", "cost_per_side_bps": 5.0}).to_csv(
        folder / "assets.csv", index=False
    )
    keys = labels[["time", "asset", "horizon"]]
    for name, quantiles in forecasts.items():
        write_forecasts(folder, name, _forecast_rows(keys, quantiles))
    return read_cache(folder)


def _hourly_labels(values: np.ndarray, *, start: str = "2025-11-01", asset: str = "A") -> Any:
    times = pd.date_range(start, periods=len(values), freq="1h", tz="UTC")
    return pd.DataFrame({"time": times, "asset": asset, "horizon": 1, "window": "t1"}).assign(
        label=values
    )


def _normal(center: np.ndarray, scale: float) -> np.ndarray:
    standard = np.array([NormalDist().inv_cdf(level) for level in QUANTILE_LEVELS])
    return center[:, None] + scale * standard[None, :]


quarter_steps = st.integers(-8, 8).map(lambda value: value / 4)


@given(data=st.data())
@settings(max_examples=40, deadline=None)
def test_one_cell_matches_a_direct_computation(
    rq2: ModuleType, data: st.DataObject, tmp_path_factory: pytest.TempPathFactory
) -> None:
    n = data.draw(st.integers(2, 30))
    labels = data.draw(arrays(float, n, elements=quarter_steps))
    quantiles = np.sort(data.draw(arrays(float, (n, 9), elements=quarter_steps)), axis=1)
    climatology = np.full((n, 9), 1.0)
    forecasts = {"model": quantiles, "reference": quantiles, "climatology": climatology}
    folder = tmp_path_factory.mktemp("small") / "cache"
    cache = _small_cache(folder, _hourly_labels(labels), forecasts)
    bootstrap = Bootstrap(level=0.9, n_resamples=50, seed=0, block_hours=1, min_block=1)
    rule = _rule(rq2, scale="reference")
    tables = rq2.centre_scale_tables(
        cache, ["model"], _settings(TEST_FOLDS[:1], bootstrap), rule=rule
    )
    cell = tables["cells"].set_index("model").loc["model"]
    shares = (labels[:, None] <= quantiles).mean(axis=0)
    np.testing.assert_allclose(cell[BELOW].to_numpy(dtype=float), shares)
    assert cell["coverage"] == pytest.approx(shares[-1] - shares[0])
    assert cell["coverage_error"] == pytest.approx(abs(shares[-1] - shares[0] - 0.8))
    assert cell["ks_distance"] == pytest.approx(np.abs(shares - LEVELS).max())
    assert cell["reference_coverage_error"] == pytest.approx(cell["coverage_error"])
    assert cell["reference_ks_distance"] == pytest.approx(cell["ks_distance"])
    median = quantiles[:, 4]
    signed = (median != 0) & (labels != 0)
    hits = np.sign(median[signed]) == np.sign(labels[signed])
    up = labels[signed] > 0
    expected_hit = hits.mean() if signed.any() else math.nan
    expected_reference = up.mean() if signed.any() else math.nan
    np.testing.assert_allclose(cell["hit_rate"], expected_hit)
    np.testing.assert_allclose(cell["reference_hit_rate"], expected_reference)
    np.testing.assert_allclose(cell["hit_difference"], expected_hit - expected_reference)
    assert cell["signed_decisions"] == signed.sum()
    np.testing.assert_allclose(cell["rank_ic"], spearman(pd.Series(median), pd.Series(labels)))
    crps = quantile_crps(quantiles, LEVELS, labels).mean()
    centred = quantile_crps(quantiles - median[:, None], LEVELS, labels).mean()
    assert cell["crps"] == pytest.approx(crps)
    assert cell["centred_crps"] == pytest.approx(centred)
    if centred > 0:
        assert cell["centre_share"] == pytest.approx(1 - crps / centred)
    differences = hits.astype(float) - up
    inside = ((labels > quantiles[:, 0]) & (labels <= quantiles[:, -1])).astype(float)
    series = differences if signed.any() else inside
    assert cell["overlap"] == 0
    assert cell["block"] == max(politis_white_block(series), 2)
    assert cell["decisions"] == n
    quarters = tables["quarters"][tables["quarters"]["model"] == "model"]
    assert quarters[BELOW].to_numpy(dtype=float) == pytest.approx(shares[None, :])


def test_every_decision_time_weighs_the_same_in_the_crps_and_the_hit_rate(
    rq2: ModuleType, tmp_path: Path
) -> None:
    # asset A at every hour, asset B only at the first hour: a plain mean over decisions
    # would weigh the first hour twice
    first = _hourly_labels(np.array([1.0, 1.0, -1.0, 1.0]), asset="A")
    second = _hourly_labels(np.array([-1.0]), asset="B")
    labels = pd.concat([first, second], ignore_index=True)
    medians = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    forecasts = {
        "model": _normal(medians, 1.0),
        "reference": _normal(np.zeros(5), 1.0),
        "climatology": _normal(np.full(5, -1.0), 1.0),
    }
    cache = _small_cache(tmp_path / "cache", labels, forecasts)
    bootstrap = Bootstrap(level=0.9, n_resamples=20, seed=0, block_hours=1, min_block=1)
    rule = _rule(rq2, scale="reference")
    tables = rq2.centre_scale_tables(
        cache, ["model"], _settings(TEST_FOLDS[:1], bootstrap), rule=rule
    )
    cell = tables["cells"].set_index("model").loc["model"]
    # first hour: A right, B wrong (share 0.5); then right, wrong, right
    assert cell["hit_rate"] == pytest.approx((0.5 + 1 + 0 + 1) / 4)
    assert cell["reference_hit_rate"] == pytest.approx((0.5 + 0 + 1 + 0) / 4)
    crps = quantile_crps(forecasts["model"], LEVELS, labels["label"].to_numpy())
    per_time = [(crps[0] + crps[4]) / 2, crps[1], crps[2], crps[3]]
    assert cell["crps"] == pytest.approx(np.mean(per_time))
    assert cell["crps"] != pytest.approx(crps.mean())


def test_the_assets_of_one_grid_point_are_one_decision_time(
    rq2: ModuleType, tmp_path: Path
) -> None:
    # a grid model (every 4 hours, like Kronos) whose assets decide at different times of one
    # grid point: A at 00:00 of every point, B at 01:00 of every other point; A is always
    # right and B always wrong, so the points share 0.5 and 1 in turn (mean 0.75), while a
    # mean over raw times would give 2 / 3
    points = pd.date_range("2025-11-01", periods=12, freq="4h", tz="UTC")
    first = pd.DataFrame({"time": points, "asset": "A"})
    second = pd.DataFrame({"time": points[::2] + pd.Timedelta(hours=1), "asset": "B"})
    labels = pd.concat([first, second], ignore_index=True).assign(
        horizon=1, window="t1", label=lambda frame: np.where(frame["asset"] == "A", 1.0, -1.0)
    )
    n = len(labels)
    forecasts = {
        "model": _normal(np.ones(n), 1.0),
        "reference": _normal(np.zeros(n), 1.0),
        "climatology": _normal(np.full(n, 1.0), 1.0),
    }
    cache = _small_cache(tmp_path / "cache", labels, forecasts, every_hours=4)
    bootstrap = Bootstrap(level=0.9, n_resamples=20, seed=0, block_hours=1, min_block=1)
    rule = _rule(rq2, scale="reference")
    tables = rq2.centre_scale_tables(
        cache, ["model"], _settings(TEST_FOLDS[:1], bootstrap), rule=rule
    )
    cell = tables["cells"].set_index("model").loc["model"]
    assert cell["times"] == 12
    assert cell["hit_rate"] == pytest.approx(0.75)


def test_every_test_and_interval_of_a_cell_uses_its_block_and_its_seed(
    rq2: ModuleType, tmp_path: Path
) -> None:
    # one asset, so a decision time is a decision; a tenth of the labels are zero, so some
    # times have no sign and the test runs on fewer times than the coverage interval
    rng = np.random.default_rng(5)
    label = np.round(rng.standard_normal(300) + 0.1, 1)
    label[rng.random(300) < 0.1] = 0.0
    median = 0.3 * label + rng.standard_normal(300)
    forecasts = {
        "model": _normal(median, 1.0),
        "reference": _normal(np.zeros(300), 1.0),
        "climatology": _normal(np.full(300, 0.1), 1.0),
    }
    cache = _small_cache(tmp_path / "cache", _hourly_labels(label), forecasts)
    bootstrap = Bootstrap(level=0.95, n_resamples=300, seed=11, block_hours=1, min_block=1)
    rule = _rule(rq2, scale="reference")
    tables = rq2.centre_scale_tables(
        cache, ["model"], _settings(TEST_FOLDS[:1], bootstrap), rule=rule
    )
    cell = tables["cells"].set_index("model").loc["model"]
    quantiles = forecasts["model"]
    signed = (label != 0) & (median != 0)
    assert cell["signed_times"] == signed.sum() < cell["times"] == 300
    hits = (np.sign(median[signed]) == np.sign(label[signed])).astype(float)
    differences = hits - (label[signed] > 0)
    block = int(cell["block"])
    assert block == max(politis_white_block(differences), 2)
    seed = int(rq2.cell_seed(bootstrap, "model", "crypto", 1))
    draws: dict[str, Any] = {"level": 0.95, "n_resamples": 300, "seed": seed}
    test = bootstrap_mean_test(differences, null=0.0, block=block, **draws)
    assert cell["p_value"] == pytest.approx(test["p_greater"])
    assert (cell["diff_low"], cell["diff_high"]) == pytest.approx((test["low"], test["high"]))
    hit_test = bootstrap_mean_test(hits, null=0.0, block=block, **draws)
    assert (cell["hit_low"], cell["hit_high"]) == pytest.approx((hit_test["low"], hit_test["high"]))
    inside = ((label > quantiles[:, 0]) & (label <= quantiles[:, -1])).astype(float)
    coverage = rq2.ratio_interval(inside, np.ones(300), block=block, **draws)
    assert (cell["coverage_low"], cell["coverage_high"]) == pytest.approx(coverage)


def test_a_decision_the_centre_reference_did_not_forecast_is_left_out_and_counted(
    rq2: ModuleType, tmp_path: Path
) -> None:
    label = np.linspace(-1, 1, 40) + 0.01
    forecasts = {
        "model": _normal(label, 1.0),
        "reference": _normal(np.zeros(40), 1.0),
        "climatology": _normal(np.full(40, 0.1), 1.0),
    }
    folder = tmp_path / "cache"
    _small_cache(folder, _hourly_labels(label), forecasts)
    path = folder / "forecasts" / "climatology.parquet"
    pd.read_parquet(path).iloc[3:].to_parquet(path)
    bootstrap = Bootstrap(level=0.9, n_resamples=20, seed=0, block_hours=1, min_block=1)
    tables = rq2.centre_scale_tables(
        read_cache(folder), ["model"], _settings(TEST_FOLDS[:1], bootstrap),
        rule=_rule(rq2, scale="reference"),
    )  # fmt: skip
    cell = tables["cells"].set_index("model").loc["model"]
    assert cell["unpaired"] == 3
    assert cell["decisions"] == 37


def test_a_cells_draws_do_not_depend_on_the_other_models_of_the_run(
    rq2: ModuleType, centre_scale_cache_folder: Path
) -> None:
    alone = _tables(rq2, centre_scale_cache_folder, ["centre-only"])["cells"]
    together = _tables(rq2, centre_scale_cache_folder, ["scale-only", "centre-only"])["cells"]
    columns = ["p_value", "diff_low", "hit_high", "coverage_low"]
    first = alone[alone["model"] == "centre-only"][columns].to_numpy(dtype=float)
    second = together[together["model"] == "centre-only"][columns].to_numpy(dtype=float)
    np.testing.assert_allclose(first, second)


def test_a_run_of_baselines_only_raises(rq2: ModuleType, centre_scale_cache_folder: Path) -> None:
    with pytest.raises(ValueError, match="are all baselines"):
        _tables(rq2, centre_scale_cache_folder, ["climatology"])


# --- the hit rate is judged against the centre reference -------------------------------------


@pytest.fixture(scope="module")
def drift_cache(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Four test quarters of one crypto asset whose label drifts up (mean 0.3).

    ``model``'s median is always positive and only slightly varying (it rises with the
    label, so its rank IC is positive), climatology's median is the drift, 0.3.
    """
    rng = np.random.default_rng(3)
    parts = [
        _hourly_labels(np.zeros(300), start=str(fold.evaluation_start.date())).assign(
            window=fold.name
        )
        for fold in TEST_FOLDS
    ]
    labels = pd.concat(parts, ignore_index=True)
    label = 0.3 + rng.standard_normal(len(labels))
    labels = labels.assign(label=label)
    forecasts = {
        "model": _normal(0.05 + 0.02 * np.tanh(label), 1.0),
        "ewma-normal": _normal(np.zeros(len(labels)), 1.0),
        "climatology": _normal(np.full(len(labels), 0.3), 1.0),
    }
    return _small_cache(tmp_path_factory.mktemp("drift") / "cache", labels, forecasts)


def test_a_median_that_only_follows_the_drift_has_no_centre_skill(
    rq2: ModuleType, drift_cache: ForecastCache
) -> None:
    # tested against 0.5, this always-positive median "predicts the sign" (hit rate about
    # 0.62, adjusted p about 1e-26, positive rank IC in every quarter); climatology, always
    # positive too, has the same hits, so the difference is always 0: no test, the cell fails
    rule = _rule(rq2)
    tables = rq2.centre_scale_tables(drift_cache, ["model"], _settings(), rule=rule)
    cells, quarters = tables["cells"], tables["quarters"]
    verdict = rq2.centre_scale_verdict(
        cells[cells["model"] == "model"], quarters[quarters["model"] == "model"],
        alpha=0.05, rule=rule,
    ).iloc[0]  # fmt: skip
    assert verdict["hit_rate"] > 0.6
    assert verdict["positive_ic_quarters"] == 4
    assert verdict["reference_hit_rate"] == pytest.approx(verdict["hit_rate"])
    assert verdict["hit_difference"] == pytest.approx(0.0)
    assert np.isnan(verdict["p_value"])
    assert verdict["centre_test"] == "untested: no spread in the hit difference"
    assert verdict["hit_low"] <= verdict["hit_rate"] <= verdict["hit_high"]
    assert verdict["diff_low"] == pytest.approx(0.0)
    assert verdict["diff_high"] == pytest.approx(0.0)
    assert not verdict["has_centre_skill"]
    assert not verdict["has_worse_centre"]


def test_the_share_of_decisions_with_a_zero_reference_median_is_reported(
    rq2: ModuleType, tmp_path: Path
) -> None:
    # climatology's median is exactly zero on 30 of 100 decisions: those leave the centre
    # test, and the cell reports their share
    rng = np.random.default_rng(8)
    label = rng.standard_normal(100) + 0.001
    centre = np.full(100, 0.2)
    centre[:30] = 0.0
    forecasts = {
        "model": _normal(label + rng.standard_normal(100), 1.0),
        "reference": _normal(np.zeros(100), 1.0),
        "climatology": _normal(centre, 1.0),
    }
    cache = _small_cache(tmp_path / "cache", _hourly_labels(label), forecasts)
    bootstrap = Bootstrap(level=0.95, n_resamples=50, seed=0, block_hours=1, min_block=1)
    tables = rq2.centre_scale_tables(
        cache, ["model"], _settings(TEST_FOLDS[:1], bootstrap), rule=_rule(rq2, scale="reference")
    )
    cell = tables["cells"].set_index("model").loc["model"]
    assert cell["zero_reference_share"] == pytest.approx(0.3)
    assert cell["signed_decisions"] == 70
    quarters = tables["quarters"].set_index("model").loc["model"]
    assert quarters["zero_reference_share"] == pytest.approx(0.3)


# --- centre skill and scale skill are told apart --------------------------------------------


MODELS = ["climatology", "sampled", "ewma-normal", "centre-only", "scale-only"]


@pytest.fixture(name="tables", scope="module")
def _tables_fixture(rq2: ModuleType, centre_scale_cache_folder: Path) -> Tables:
    return _tables(rq2, centre_scale_cache_folder, MODELS)


def test_a_model_with_centre_skill_and_bad_scale_has_only_centre_skill(
    rq2: ModuleType, tables: Tables
) -> None:
    verdict = _verdict(rq2, tables)
    centre = verdict[verdict["model"] == "centre-only"]
    assert len(centre) == 9
    assert centre["has_centre_skill"].all()
    assert not centre["is_calibrated"].any()
    assert not centre["beats_reference_scale"].any()
    assert (centre["hit_rate"] > 0.65).all()
    assert (centre["hit_low"] > centre["reference_hit_rate"]).all()
    assert (centre["diff_low"] > 0).all()
    assert (centre["family"] == "main").all()


def test_a_calibrated_model_without_centre_skill_has_only_scale_skill(
    rq2: ModuleType, tables: Tables
) -> None:
    verdict = _verdict(rq2, tables)
    scale = verdict[verdict["model"] == "scale-only"]
    assert len(scale) == 9
    assert not scale["has_centre_skill"].any()
    assert scale["is_calibrated"].all()
    assert scale["beats_reference_scale"].all()
    assert (scale["coverage_error"] < scale["reference_coverage_error"]).all()
    assert (scale["ks_distance"] < scale["reference_ks_distance"]).all()
    assert (scale["coverage_low"] < 0.8).all()
    assert (scale["coverage_high"] > 0.8).all()


def test_the_tables_have_one_row_per_model_cell_and_window(tables: Tables) -> None:
    cells, quarters = tables["cells"], tables["quarters"]
    assert len(cells) == len(MODELS) * 3 * 3
    assert len(quarters) == len(MODELS) * 3 * 3 * 4
    assert set(cells["kind"]) == {"baseline", "zero-shot"}
    assert (cells["quarters"] == 4).all()
    assert (cells["full_quarters"] == 4).all()
    assert not cells.duplicated(CELL).any()
    assert (cells.loc[cells["model"] != "sampled", "decisions"] == 800).all()
    for low, high in [("hit_low", "hit_high"), ("diff_low", "diff_high")]:
        assert (cells[low].isna() | (cells[low] <= cells[high])).all()
    assert (cells["coverage_low"] <= cells["coverage"]).all()
    assert (cells["coverage"] <= cells["coverage_high"]).all()


def test_every_test_block_holds_four_times_the_overlap_of_the_labels(tables: Tables) -> None:
    cells = tables["cells"]
    assert BLOCK_OVERLAP_FLOOR == 4
    assert (cells["block"] >= 4 * cells["overlap"]).all()
    hourly = cells[(cells["model"] != "sampled") & (cells["horizon"] == 24)]
    assert (hourly["overlap"] == 23).all()
    assert (hourly["block"] >= 92).all()


def test_the_reference_compared_with_itself_never_beats_itself(tables: Tables) -> None:
    cells = tables["cells"]
    reference = cells[cells["model"] == "ewma-normal"]
    np.testing.assert_allclose(reference["coverage_error"], reference["reference_coverage_error"])
    np.testing.assert_allclose(reference["ks_distance"], reference["reference_ks_distance"])
    climatology = cells[cells["model"] == "climatology"]
    np.testing.assert_allclose(climatology["hit_difference"], 0.0)
    assert climatology["p_value"].isna().all()


def test_a_median_that_is_always_zero_has_no_sign_test(rq2: ModuleType, tables: Tables) -> None:
    cells = tables["cells"]
    zero = cells[cells["model"] == "ewma-normal"]
    columns = ["hit_rate", "reference_hit_rate", "hit_low", "diff_high", "p_value"]
    assert zero[columns].isna().all().all()
    assert (zero["signed_decisions"] == 0).all()
    assert zero["rank_ic"].isna().all()
    assert zero["coverage_low"].notna().all()
    both = cells[cells["model"].isin(["ewma-normal", "centre-only"])]
    quarters = tables["quarters"][tables["quarters"]["model"].isin(["ewma-normal", "centre-only"])]
    verdict = rq2.centre_scale_verdict(both, quarters, alpha=0.05, rule=_rule(rq2))
    untested = verdict[verdict["model"] == "ewma-normal"]
    assert untested["adjusted_p"].isna().all()
    assert not untested["has_centre_skill"].any()
    tested = verdict[verdict["model"] == "centre-only"]
    np.testing.assert_allclose(tested["adjusted_p"], holm(tested["p_value"].to_numpy()))


def test_the_seeds_of_a_sampling_model_are_averaged_quantile_by_quantile(
    tables: Tables, centre_scale_cache_folder: Path
) -> None:
    labels: pd.DataFrame = read_cache(centre_scale_cache_folder).labels
    hours = pd.DatetimeIndex(labels["time"]).hour.to_numpy()
    is_kept = labels["window"].isin(TEST).to_numpy() & (hours % 4 == 0)
    markets = {"BTCUSDT": "crypto", "EURUSD": "forex", "AAPL": "stock"}
    kept = labels.loc[is_kept, ["asset", "horizon", "label"]].reset_index(drop=True)
    label = kept["label"].to_numpy(dtype=float)
    # seeds shift the centre by 0.1 and -0.3, so the averaged median is -0.1 at every decision;
    # one asset per market, so the per-time share is the plain share
    shares = pd.DataFrame({
        "market": [markets[str(asset)] for asset in kept["asset"]], "horizon": kept["horizon"],
        "below": label <= -0.1, "hit": label < 0, "reference": label > 0,
    })  # fmt: skip
    expected: pd.DataFrame = shares.groupby(["market", "horizon"]).mean()
    cells = tables["cells"]
    sampled = cells[cells["model"] == "sampled"].set_index(["market", "horizon"])
    np.testing.assert_allclose(sampled["below_q50"], expected.loc[sampled.index, "below"])
    np.testing.assert_allclose(sampled["hit_rate"], expected.loc[sampled.index, "hit"])
    np.testing.assert_allclose(
        sampled["reference_hit_rate"], expected.loc[sampled.index, "reference"]
    )
    assert (sampled["decisions"] == 200).all()
    assert sampled["rank_ic"].isna().all()


def test_the_references_are_scored_only_on_the_decisions_of_the_model(
    rq2: ModuleType, centre_scale_cache_folder: Path
) -> None:
    tables = _tables(rq2, centre_scale_cache_folder, ["scale-only"], scale="sampled")
    assert (tables["cells"]["decisions"] == 200).all()
    scale_only = tables["cells"][tables["cells"]["model"] == "scale-only"]
    assert (scale_only["unpaired"] == 600).all()
    full = _tables(rq2, centre_scale_cache_folder, ["sampled"], scale="scale-only")
    cells = full["cells"]
    assert (cells.loc[cells["model"] == "sampled", "decisions"] == 200).all()
    assert (cells.loc[cells["model"] == "sampled", "unpaired"] == 0).all()


def test_a_model_released_inside_the_window_is_scored_after_its_release(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "centre-only", "2026-03-01")
    tables = _tables(rq2, folder, ["centre-only", "ewma-normal"])
    cells = tables["cells"]
    late = cells[cells["model"] == "centre-only"]
    assert (late["quarters"] == 2).all()
    assert (late["full_quarters"] == 2).all()
    assert (late["decisions"] == 400).all()
    assert (cells.loc[cells["model"] == "ewma-normal", "quarters"] == 4).all()
    assert not _verdict(rq2, tables)["has_centre_skill"].any()


def test_a_model_with_fewer_full_quarters_is_corrected_in_its_own_family(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "scale-only", "2026-06-01")
    tables = _tables(rq2, folder, ["centre-only", "scale-only"])
    verdict = _verdict(rq2, tables)
    main = verdict[verdict["model"] == "centre-only"]
    late = verdict[verdict["model"] == "scale-only"]
    assert (main["family"] == "main").all()
    assert (late["family"] == "late").all()
    assert (late["full_quarters"] == 1).all()
    np.testing.assert_allclose(main["adjusted_p"], holm(main["p_value"].to_numpy()))
    np.testing.assert_allclose(late["adjusted_p"], holm(late["p_value"].to_numpy()))
    assert main["has_centre_skill"].all()
    assert main["family_has_centre_skill"].all()
    assert not late["family_has_centre_skill"].any()


# --- broken input ---------------------------------------------------------------------------


def test_unknown_models_and_windows_raise(rq2: ModuleType, centre_scale_cache_folder: Path) -> None:
    with pytest.raises(ValueError, match=r"models \['nope'\], windows \[\]"):
        _tables(rq2, centre_scale_cache_folder, ["scale-only"], scale="nope")
    with pytest.raises(ValueError, match=r"models \[\], windows \['l1'\]"):
        _tables(
            rq2, centre_scale_cache_folder, ["scale-only"], folds=(TEST_FOLDS[0], LEAKAGE_FOLDS[0])
        )


def test_a_model_without_a_decision_after_its_release_is_not_scored(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "scale-only", "2027-01-01")
    tables = _tables(rq2, folder, ["scale-only", "centre-only"])
    late = tables["cells"][tables["cells"]["model"] == "scale-only"]
    assert len(late) == 9
    assert (late["status"] == RELEASED_AFTER).all()
    assert (late["full_quarters"] == 0).all()
    verdict = _verdict(rq2, tables)
    assert not verdict.loc[verdict["model"] == "scale-only", "is_calibrated"].any()
    tested = verdict[verdict["model"] == "centre-only"]
    np.testing.assert_allclose(tested["adjusted_p"], holm(tested["p_value"].to_numpy()))


def test_no_scored_cell_at_all_raises_with_the_reasons(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "scale-only", "2027-01-01")
    with pytest.raises(
        ValueError,
        match=r"no zero-shot model of rq2 could be scored on .*: 9 cells not scored: released",
    ):
        _tables(rq2, folder, ["scale-only", "ewma-normal"])


def test_an_invalid_forecast_is_left_out_and_counted(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    path = folder / "forecasts" / "scale-only.parquet"
    forecasts = pd.read_parquet(path)
    row = np.flatnonzero(forecasts["time"] >= pd.Timestamp("2025-11-01", tz="UTC"))[0]
    forecasts.loc[row, "q90"] = np.nan
    forecasts.to_parquet(path)
    tables = _tables(rq2, folder, ["scale-only"])
    cells = tables["cells"][tables["cells"]["model"] == "scale-only"]
    assert cells["invalid"].sum() == 1
    assert cells["decisions"].sum() == 9 * 800 - 1


def test_an_asset_without_market_raises(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    assets = pd.read_csv(folder / "assets.csv")
    assets[assets["asset"] != "EURUSD"].to_csv(folder / "assets.csv", index=False)
    with pytest.raises(ValueError, match=r"assets \['EURUSD'\] of ewma-normal"):
        _tables(rq2, folder, ["centre-only"])


def test_a_cell_shorter_than_its_bootstrap_block_is_not_scored(
    rq2: ModuleType, centre_scale_cache_folder: Path, tmp_path: Path
) -> None:
    # released on the last day of t4, "sampled" keeps one decision time per cell (04:00),
    # fewer than the shortest block, 2
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "sampled", "2026-08-09")
    tables = _tables(rq2, folder, ["sampled", "scale-only"])
    cells = tables["cells"]
    sampled = cells[cells["model"] == "sampled"]
    assert (sampled["times"] == 1).all()
    expected = "not scored: 1 decision times, fewer than the bootstrap block 2"
    assert (sampled["status"] == expected).all()
    assert sampled[["p_value", "coverage_low"]].isna().all().all()
    assert (cells.loc[cells["model"] == "scale-only", "status"] == SCORED).all()


# --- the decision rule ----------------------------------------------------------------------


def _rule_tables(
    *,
    difference: float = 0.1,
    p_value: float = 1e-6,
    rank_ics: tuple[float, ...] = (0.1,) * 4,
    errors: tuple[float, ...] = (0.0,) * 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.DataFrame(
        {"model": ["m"], "market": ["crypto"], "horizon": [1], "hit_difference": [difference],
         "diff_high": [difference + 0.05], "p_value": [p_value], "status": [SCORED],
         "full_quarters": [4]}
    )  # fmt: skip
    quarters = pd.DataFrame({
        "model": "m", "market": "crypto", "horizon": 1, "rank_ic": list(rank_ics),
        "coverage_error": list(errors), "ks_distance": 0.01,
        "reference_coverage_error": 0.04, "reference_ks_distance": 0.02,
    })  # fmt: skip
    return cells, quarters


def _one_verdict(
    rq2: ModuleType,
    cells: pd.DataFrame,
    quarters: pd.DataFrame,
    *,
    coverage_tolerance: float = 0.05,
) -> pd.Series:
    rule = _rule(rq2, coverage_tolerance=coverage_tolerance)
    return rq2.centre_scale_verdict(cells, quarters, alpha=0.05, rule=rule).iloc[0]


def test_a_cell_not_scored_passes_no_rule(rq2: ModuleType) -> None:
    cells, quarters = _rule_tables()
    status = "not scored: 3 decision times, fewer than the bootstrap block 24"
    row = _one_verdict(rq2, cells.assign(status=status, p_value=np.nan), quarters)
    assert not row["has_centre_skill"]
    assert not row["has_worse_centre"]
    assert not row["is_calibrated"]
    assert not row["beats_reference_scale"]


@pytest.mark.parametrize(
    ("difference", "p_value", "rank_ics", "expected"),
    [
        (0.1, 1e-6, (0.1, 0.1, 0.1, 0.1), True),
        (0.1, 1e-6, (0.1, 0.1, 0.1, -0.1), True),
        (0.1, 1e-6, (0.1, 0.1, -0.1, -0.1), False),
        (0.1, 1e-6, (0.1, 0.1, 0.1, np.nan), True),
        (0.1, 1e-6, (0.1, 0.1, 0.0, np.nan), False),
        (0.0, 1e-6, (0.1, 0.1, 0.1, 0.1), False),
        (-0.1, 1e-6, (0.1, 0.1, 0.1, 0.1), False),
        (0.1, 0.05, (0.1, 0.1, 0.1, 0.1), False),
        (0.1, np.nan, (0.1, 0.1, 0.1, 0.1), False),
    ],
)
def test_centre_skill_needs_a_significant_positive_difference_and_three_positive_ics(
    rq2: ModuleType,
    difference: float,
    p_value: float,
    rank_ics: tuple[float, ...],
    expected: bool,
) -> None:
    cells, quarters = _rule_tables(difference=difference, p_value=p_value, rank_ics=rank_ics)
    assert _one_verdict(rq2, cells, quarters)["has_centre_skill"] == expected


def test_a_worse_centre_comes_from_the_interval_of_the_difference(rq2: ModuleType) -> None:
    cells, quarters = _rule_tables(difference=-0.1, p_value=0.99)
    assert _one_verdict(rq2, cells, quarters)["has_worse_centre"]
    straddling = cells.assign(diff_high=0.01)
    assert not _one_verdict(rq2, straddling, quarters)["has_worse_centre"]


@pytest.mark.parametrize(
    ("errors", "calibrated", "better"),
    [
        ((0.0, 0.01, 0.02, 0.03), True, True),
        ((0.05, 0.05, 0.05, 0.2), True, False),
        ((0.04, 0.04, 0.04, 0.2), True, False),
        ((0.03, 0.03, 0.06, 0.06), False, False),
        ((0.06, 0.06, 0.06, 0.06), False, False),
    ],
)
def test_calibration_and_a_better_scale_need_three_quarters(
    rq2: ModuleType, errors: tuple[float, ...], calibrated: bool, better: bool
) -> None:
    cells, quarters = _rule_tables(errors=errors)
    row = _one_verdict(rq2, cells, quarters)
    assert row["is_calibrated"] == calibrated
    assert row["beats_reference_scale"] == better


@pytest.mark.parametrize("coverage", [0.75, 0.85])
def test_a_coverage_exactly_at_the_tolerance_is_calibrated_on_both_sides(
    rq2: ModuleType, coverage: float
) -> None:
    cells, quarters = _rule_tables(errors=(abs(coverage - (0.9 - 0.1)),) * 4)
    assert _one_verdict(rq2, cells, quarters)["is_calibrated"]


def test_a_better_scale_needs_both_the_coverage_error_and_the_ks_distance_lower(
    rq2: ModuleType,
) -> None:
    cells, quarters = _rule_tables()
    assert _one_verdict(rq2, cells, quarters)["beats_reference_scale"]
    worse_ks = quarters.assign(ks_distance=[0.01, 0.01, 0.03, 0.03])
    assert not _one_verdict(rq2, cells, worse_ks)["beats_reference_scale"]
    equal_ks = quarters.assign(ks_distance=0.02)
    assert not _one_verdict(rq2, cells, equal_ks)["beats_reference_scale"]


probabilities = st.floats(0, 1, allow_nan=False)


@given(
    rank_ics=st.lists(st.floats(-1, 1, allow_nan=False), min_size=1, max_size=6),
    errors=st.lists(st.floats(0, 0.3, allow_nan=False), min_size=6, max_size=6),
    tolerance=st.floats(0, 0.3, allow_nan=False),
    p_values=st.lists(probabilities, min_size=4, max_size=4),
    quarters_after=st.lists(st.integers(0, 4), min_size=4, max_size=4),
)
def test_the_rule_counts_the_quarters_and_adjusts_within_each_family(
    rq2: ModuleType,
    rank_ics: list[float],
    errors: list[float],
    tolerance: float,
    p_values: list[float],
    quarters_after: list[int],
) -> None:
    n = len(rank_ics)
    quarters = pd.DataFrame({
        "model": "m", "market": "crypto", "horizon": 1, "rank_ic": rank_ics,
        "coverage_error": errors[:n], "ks_distance": errors[-n:],
        "reference_coverage_error": 0.1, "reference_ks_distance": 0.1,
    })  # fmt: skip
    cells = pd.DataFrame(
        {"model": "m", "market": ["crypto", "crypto", "forex", "forex"], "horizon": [1, 4, 1, 4],
         "hit_difference": 0.1, "diff_high": 0.2, "p_value": p_values, "status": SCORED,
         "full_quarters": quarters_after}
    )  # fmt: skip
    cell_keys = zip(cells["market"], cells["horizon"], strict=True)
    every = pd.concat(
        [quarters.assign(market=market, horizon=horizon) for market, horizon in cell_keys]
    )
    verdict = rq2.centre_scale_verdict(
        cells, every, alpha=0.05, rule=_rule(rq2, coverage_tolerance=tolerance)
    )
    assert (verdict["positive_ic_quarters"] == sum(ic > 0 for ic in rank_ics)).all()
    # the rule allows FLOAT_SLACK (1e-9) above the tolerance
    calibrated = sum(error <= tolerance + 1e-9 for error in errors[:n])
    assert (verdict["calibrated_quarters"] == calibrated).all()
    better = sum(a < 0.1 and b < 0.1 for a, b in zip(errors[:n], errors[-n:], strict=True))
    assert (verdict["better_scale_quarters"] == better).all()
    is_main = np.array(quarters_after) >= 3
    assert (verdict["family"].to_numpy() == np.where(is_main, "main", "late")).all()
    p = np.array(p_values)
    for members in (is_main, ~is_main):
        np.testing.assert_allclose(verdict["adjusted_p"][members], holm(p[members]))
    assert (verdict["is_calibrated"] == (calibrated >= 3)).all()
    skilled = verdict["has_centre_skill"]
    assert (~skilled | (verdict["adjusted_p"] < 0.05)).all()


# --- the five-block rule and the ratio interval ----------------------------------------------


def _cell_of(rq2: ModuleType, folder: Path, label: np.ndarray) -> pd.Series:
    n = len(label)
    rng = np.random.default_rng(9)
    forecasts = {
        "model": _normal(0.5 * label + rng.standard_normal(n) + 0.01, 1.0),
        "reference": _normal(np.zeros(n), 1.0),
        "climatology": _normal(np.full(n, 0.1), 1.0),
    }
    cache = _small_cache(folder, _hourly_labels(label), forecasts)
    settings = _settings(
        TEST_FOLDS[:1], Bootstrap(level=0.95, n_resamples=50, seed=0, block_hours=1, min_block=1)
    )
    tables = rq2.centre_scale_tables(cache, ["model"], settings, rule=_rule(rq2, scale="reference"))
    return tables["cells"].set_index("model").loc["model"]


def test_a_test_or_interval_needs_five_blocks_of_its_own_decision_times(
    rq2: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every cell gets block 2 (1-bar labels do not overlap), so 5 blocks are 10 times
    def fixed_block(series: np.ndarray) -> int:
        return 2

    monkeypatch.setattr(rq2._centre_scale, "politis_white_block", fixed_block)
    rng = np.random.default_rng(4)
    exactly = _cell_of(rq2, tmp_path / "a", np.round(rng.standard_normal(10), 2) + 0.005)
    assert exactly["block"] == 2
    assert exactly["centre_test"] == "tested"
    assert np.isfinite(exactly[["p_value", "hit_low", "coverage_low"]].to_numpy(float)).all()
    # 10 times but two labels exactly zero: the coverage has 5 blocks, the centre 8 times
    label = np.round(rng.standard_normal(10), 2) + 0.005
    label[[3, 7]] = 0.0
    split = _cell_of(rq2, tmp_path / "b", label)
    assert (
        split["centre_test"] == "untested: 8 decision times with a sign, fewer than 5 blocks of 2"
    )
    assert np.isnan(split["p_value"])
    assert np.isfinite(split[["coverage_low", "coverage_high"]].to_numpy(float)).all()
    # 9 times: scored (at least one block) but no test and no interval
    short = _cell_of(rq2, tmp_path / "c", np.round(rng.standard_normal(9), 2) + 0.005)
    assert short["status"] == SCORED
    assert short[["p_value", "hit_low", "diff_high", "coverage_low"]].isna().all()


def test_the_overlap_floor_keeps_the_size_of_overlapping_labels() -> None:
    # a size sanity check of the floor, not a comparison with the earlier one: 24-bar labels
    # overlap 23 later ones; at exactly 5 blocks of the floor (4 x 23 = 92) the one-sided 5 %
    # test rejects a true null at most 9 % of the time (5.8 % on 4000 series); the block
    # itself is pinned by test_every_test_block_holds_four_times_the_overlap_of_the_labels
    rng = np.random.default_rng(0)
    block = BLOCK_OVERLAP_FLOOR * 23
    rejected = 0
    for seed in range(300):
        steps = rng.standard_normal(5 * block + 23)
        labels = np.convolve(steps, np.ones(24), "valid")
        test = bootstrap_mean_test(
            labels, null=0.0, level=0.95, n_resamples=500, seed=seed, block=block
        )
        rejected += test["p_greater"] < 0.05
    assert rejected / 300 <= 0.09


def test_the_ratio_interval_is_the_mean_t_interval_when_every_time_has_one_decision(
    rq2: ModuleType,
) -> None:
    rng = np.random.default_rng(2)
    inside = (rng.random(200) < 0.8).astype(float)
    ratio = rq2.ratio_interval(inside, np.ones(200), level=0.95, n_resamples=300, seed=0, block=5)
    expected = bootstrap_mean_test(inside, null=0.0, level=0.95, n_resamples=300, seed=0, block=5)
    assert ratio == pytest.approx((expected["low"], expected["high"]))
    weighted = rq2.ratio_interval(
        inside * 2, np.full(200, 2.0), level=0.95, n_resamples=300, seed=0, block=5
    )
    assert weighted == pytest.approx(ratio)


# --- the figure and its table ---------------------------------------------------------------


def test_the_figure_table_puts_the_reference_first_then_the_models_by_release(
    rq2: ModuleType,
    centre_scale_cache_folder: Path,
    tmp_path: Path,
) -> None:
    folder = _copy(centre_scale_cache_folder, tmp_path)
    _set_release(folder, "centre-only", "2025-06-01")
    tables = _tables(rq2, folder, ["scale-only", "centre-only", "ewma-normal"])
    cells, quarters = tables["cells"], tables["quarters"]
    rule = _rule(rq2)
    reference = rq2.centre_scale_verdict(
        cells[cells["model"] == "ewma-normal"], quarters[quarters["model"] == "ewma-normal"],
        alpha=0.05, rule=rule,
    )  # fmt: skip
    combined = pd.concat([reference, _verdict(rq2, tables)])
    table = rq2.figure_table(combined, horizon=1)
    assert list(dict.fromkeys(table["model"])) == ["ewma-normal", "scale-only", "centre-only"]
    assert list(table["market"][:3]) == ["Crypto", "FX", "US stocks"]
    one_bar = combined[combined["horizon"] == 1]
    np.testing.assert_allclose(
        np.sort(table["coverage_gap"].to_numpy() + 0.8), np.sort(one_bar["coverage"].to_numpy())
    )
    unscored = combined.assign(status="not scored: 3 decision times, fewer than the block 4")
    hidden = rq2.figure_table(unscored, horizon=1)
    assert hidden[["hit_rate", "reference_hit_rate", "coverage_gap"]].isna().all().all()
    assert (hidden["status"] != SCORED).all()
    with pytest.raises(ValueError, match="no cell at horizon 2"):
        rq2.figure_table(reference, horizon=2)
    # the table has every column the figure reads (src.figures.read_centre_scale)
    read = {"model", "market", "horizon", "hit_rate", "hit_low", "hit_high",
            "reference_hit_rate", "is_centre_tested", "coverage_gap", "coverage_gap_low",
            "coverage_gap_high", "is_calibrated"}  # fmt: skip
    assert read <= set(table.columns)
