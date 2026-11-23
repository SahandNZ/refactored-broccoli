"""Tests of Q3 of 01-universe (``lib/_ranking.py``): the stability of the model ranking."""

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.benchmark import read_cache

MODELS = ["good", "middle", "bad"]
RANKING = {
    "forecast_cache": "test-run",
    "windows": ["t1", "t2", "t3", "t4"],
    "horizon": 1,
    "splits": 50,
    "tau_threshold": 0.7,
    "range_level": 0.95,
}


def _losses(rng: np.random.Generator, *, assets: int = 8, shuffled: bool = False) -> pd.DataFrame:
    # decisions of `assets` assets in two markets and four windows; "good" < "middle" < "bad"
    # on every asset, unless `shuffled` draws the three models' losses alike
    names = [f"A{index}" for index in range(assets)]
    times = pd.date_range("2025-11-01", periods=40, freq="1h", tz="UTC")
    grid = pd.MultiIndex.from_product([times, names], names=["time", "asset"]).to_frame(index=False)
    grid["window"] = np.repeat(["t1", "t2", "t3", "t4"], len(grid) // 4)
    grid["market"] = np.where(grid["asset"].isin(names[: assets // 2]), "crypto", "forex")
    noise = rng.uniform(0.0, 0.1, size=(len(grid), 3))
    offsets = np.zeros(3) if shuffled else np.array([0.0, 1.0, 2.0])
    for column, (model, offset) in enumerate(zip(MODELS, offsets, strict=True)):
        grid[model] = 1.0 + offset + noise[:, column]
    return grid


def _pools(means: pd.DataFrame) -> pd.Series:
    # two pools of half the assets each
    half = len(means) // 2
    return pd.Series(["crypto"] * half + ["forex"] * (len(means) - half), index=means.index)


def test_common_losses_keep_only_decisions_every_model_scored(
    universe: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    models = ["random-walk", "climatology", "skilled", "sampled"]
    windows = ["t1", "t2", "t3", "t4"]
    losses = universe.common_losses(cache, models, windows=windows, horizon=1)
    sampled = cache.forecasts("sampled")
    expected = sampled[(sampled["horizon"] == 1) & sampled["window"].isin(windows)]
    assert len(losses) == len(expected[["time", "asset"]].drop_duplicates())
    assert losses["time"].dt.hour.mod(4).eq(0).all()
    assert np.isfinite(losses[models].to_numpy()).all()
    assert set(losses["window"]) == set(windows)


def test_common_losses_score_the_skilled_model_below_climatology(
    universe: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    models = ["climatology", "skilled"]
    losses = universe.common_losses(cache, models, windows=["t1", "t2"], horizon=1)
    scores = universe.model_scores(losses, models).set_index("model")
    assert scores.loc["skilled", "crps"] < scores.loc["climatology", "crps"]
    assert scores.loc["skilled", "rank"] == 1


def test_common_losses_refuse_a_horizon_without_decisions(
    universe: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    with pytest.raises(ValueError, match="no decision"):
        universe.common_losses(cache, ["climatology"], windows=["t1"], horizon=2)


def test_kendall_is_one_for_the_same_order_and_minus_one_for_the_reverse(universe: ModuleType):
    first = pd.Series([0.1, 0.2, 0.3], index=MODELS)
    assert universe.kendall(first, first * 2) == pytest.approx(1.0)
    reverse = pd.Series([0.3, 0.2, 0.1], index=MODELS)
    assert universe.kendall(first, reverse) == pytest.approx(-1.0)


def test_kendall_aligns_the_models_by_name(universe: ModuleType):
    first = pd.Series([0.1, 0.2, 0.3], index=MODELS)
    same_order = pd.Series([0.6, 0.2, 0.4], index=["bad", "good", "middle"])
    assert universe.kendall(first, same_order) == pytest.approx(1.0)


def test_kendall_refuses_different_models(universe: ModuleType):
    with pytest.raises(ValueError, match="same two models"):
        universe.kendall(pd.Series([1.0, 2.0], index=["a", "b"]), pd.Series([1.0], index=["a"]))


def test_a_ranking_that_holds_on_every_asset_has_tau_one_in_every_split(universe: ModuleType):
    rng = np.random.default_rng(0)
    means = universe.asset_means(_losses(rng), MODELS)
    draws = universe.split_draws(means, _pools(means), splits=20, rng=np.random.default_rng(1))
    assert len(draws) == 20
    np.testing.assert_allclose(draws["tau"], 1.0)
    assert draws["same_top"].all()
    assert draws["same_top3"].all()


def test_models_of_the_same_skill_give_taus_around_zero(universe: ModuleType):
    rng = np.random.default_rng(0)
    means = universe.asset_means(_losses(rng, assets=40, shuffled=True), MODELS)
    draws = universe.split_draws(means, _pools(means), splits=200, rng=np.random.default_rng(1))
    taus = draws["tau"].to_numpy()
    assert abs(float(np.median(taus))) < 0.5
    assert taus.min() < 0 < taus.max()


def test_splits_are_reproducible_with_the_same_seed(universe: ModuleType):
    rng = np.random.default_rng(0)
    means = universe.asset_means(_losses(rng, assets=12, shuffled=True), MODELS)
    pools = _pools(means)
    first = universe.split_draws(means, pools, splits=30, rng=np.random.default_rng(5))
    second = universe.split_draws(means, pools, splits=30, rng=np.random.default_rng(5))
    pd.testing.assert_frame_equal(first, second)


def test_a_random_half_keeps_half_of_each_pool(universe: ModuleType):
    pools = pd.Series(["crypto"] * 4 + ["forex"] * 7, index=[f"A{i}" for i in range(11)])
    rng = np.random.default_rng(2)
    for _ in range(20):
        half: pd.Index = universe.random_half(pools, rng)
        assert half.is_unique
        chosen = pools.loc[half].tolist()
        assert chosen.count("crypto") == 2
        assert chosen.count("forex") == 3


def test_a_pool_of_one_asset_cannot_be_split(universe: ModuleType):
    rng = np.random.default_rng(0)
    means = universe.asset_means(_losses(rng, assets=4), MODELS)
    pools = pd.Series(["crypto", "forex", "forex", "forex"], index=means.index)
    with pytest.raises(ValueError, match="fewer than two"):
        universe.split_draws(means, pools, splits=1, rng=np.random.default_rng(0))


def test_an_asset_without_a_pool_raises(universe: ModuleType):
    rng = np.random.default_rng(0)
    means = universe.asset_means(_losses(rng, assets=4), MODELS)
    pools = pd.Series(["crypto", "crypto", "forex"], index=means.index[:3])
    with pytest.raises(ValueError, match="without a pool"):
        universe.split_draws(means, pools, splits=1, rng=np.random.default_rng(0))


def test_part_taus_compare_each_market_and_window_with_the_rest(universe: ModuleType):
    losses = _losses(np.random.default_rng(0))
    markets = universe.part_taus(losses, MODELS, column="market")
    assert markets["part"].tolist() == ["crypto", "forex"]
    assert markets["assets"].tolist() == [4, 4]
    np.testing.assert_allclose(markets["tau"], 1.0)
    windows = universe.part_taus(losses, MODELS, column="window")
    assert windows["part"].tolist() == ["t1", "t2", "t3", "t4"]
    np.testing.assert_allclose(windows["tau"], 1.0)


def test_read_ranking_settings_types_every_key(universe: ModuleType):
    settings = universe.read_ranking_settings({"ranking": RANKING})
    assert settings.windows == ("t1", "t2", "t3", "t4")
    assert settings.horizon == 1
    assert settings.tau_threshold == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"splits": None}, KeyError),
        ({"extra": 1}, KeyError),
        ({"windows": ["t1"]}, ValueError),
        ({"range_level": 1.0}, ValueError),
        ({"splits": 0}, ValueError),
    ],
)
def test_read_ranking_settings_refuses_a_bad_table(
    universe: ModuleType, change: dict[str, Any], error: type[Exception]
):
    table = {**RANKING, **change}
    table = {name: value for name, value in table.items() if value is not None}
    with pytest.raises(error):
        universe.read_ranking_settings({"ranking": table})


def test_the_selection_settings_leave_the_ranking_table_out(
    universe: ModuleType, options: dict[str, Any]
):
    settings = universe.read_settings({**options, "ranking": RANKING})
    assert settings.quotas == universe.read_settings(options).quotas


def test_ranking_tables_rank_the_hourly_models_and_every_model_on_the_grid(
    universe: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    pools = pd.Series("all", index=cache.assets["asset"].to_numpy())
    settings = universe.read_ranking_settings({"ranking": {**RANKING, "splits": 7}})
    models = ["random-walk", "climatology", "skilled", "sampled"]
    tables = universe.ranking_tables(cache, models, pools, settings, seed=0)
    scores = tables["ranking_scores"]
    assert set(scores.loc[scores["group"] == "hourly", "model"]) == {
        "random-walk", "climatology", "skilled",
    }  # fmt: skip
    assert set(scores.loc[scores["group"] == "grid", "model"]) == set(models)
    hourly = scores[scores["group"] == "hourly"]["decisions"].iloc[0]
    grid = scores[scores["group"] == "grid"]["decisions"].iloc[0]
    assert grid < hourly
    assert len(tables["ranking_splits"]) == 14
    parts = tables["ranking_parts"]
    assert set(parts["check"]) == {"market", "quarter"}
    answer, median = universe.stability_answer(tables["ranking_splits"], settings)
    hourly = tables["ranking_splits"]
    expected = float(np.median(hourly.loc[hourly["group"] == "hourly", "tau"]))
    assert median == pytest.approx(expected)
    assert answer == ("yes" if expected >= 0.7 else "no")


def test_ranking_tables_refuse_a_universe_other_than_the_cache(
    universe: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    pools = pd.Series("all", index=["BTCUSDT", "EURUSD", "OTHER"])
    settings = universe.read_ranking_settings({"ranking": RANKING})
    with pytest.raises(ValueError, match="differ"):
        universe.ranking_tables(cache, ["random-walk", "climatology"], pools, settings, seed=0)


def test_the_answer_is_yes_at_the_threshold_and_no_below(universe: ModuleType):
    settings = universe.read_ranking_settings({"ranking": RANKING})
    at = pd.DataFrame({"group": "hourly", "split": [0, 1, 2], "tau": [0.6, 0.7, 0.9]})
    assert universe.stability_answer(at, settings) == ("yes", pytest.approx(0.7))
    below = at.assign(tau=[0.5, 0.6, 0.9])
    assert universe.stability_answer(below, settings)[0] == "no"


def _means(values: dict[str, list[float]]) -> pd.DataFrame:
    # per-asset mean CRPS of the three models, one list (good, middle, bad) per asset
    return pd.DataFrame(values, index=MODELS).T.rename_axis("asset")


def test_kendall_of_one_swapped_pair_of_three_is_one_third(universe: ModuleType):
    first = pd.Series([1.0, 2.0, 3.0], index=MODELS)
    swapped = pd.Series([2.0, 1.0, 3.0], index=MODELS)
    assert universe.kendall(first, swapped) == pytest.approx(1 / 3)
    agreement = universe.agreement(first, swapped)
    assert not agreement["same_top"]
    assert not agreement["same_top3"]


def test_split_draws_compare_the_two_halves_not_a_half_with_the_whole(universe: ModuleType):
    # every half holds one crypto and one forex asset; {C1, F1} ranks good < bad < middle and
    # {C2, F2} good < middle < bad, so the two halves have tau 1/3, while either half against
    # all four assets would give another value: the draws must be the halves' own taus
    means = _means(
        {
            "C1": [1.0, 2.0, 3.0],
            "C2": [1.0, 2.0, 3.0],
            "F1": [1.0, 9.0, 2.0],
            "F2": [1.0, 3.0, 8.0],
        }
    )
    pools = pd.Series(["crypto", "crypto", "forex", "forex"], index=means.index)
    draws = universe.split_draws(means, pools, splits=10, rng=np.random.default_rng(0))
    for draw in draws.itertuples():
        rng = np.random.default_rng(0)
        half: pd.Index = universe.random_half(pools, rng)
        for _ in range(int(draw.split)):
            half = universe.random_half(pools, rng)
        first = means.loc[half].mean()
        second = means.drop(index=half).mean()
        assert draw.tau == pytest.approx(universe.kendall(first, second))
    by_forex = {
        "F1": universe.kendall(means.loc[["C1", "F1"]].mean(), means.loc[["C2", "F2"]].mean()),
        "F2": universe.kendall(means.loc[["C1", "F2"]].mean(), means.loc[["C2", "F1"]].mean()),
    }
    assert by_forex["F1"] == pytest.approx(1 / 3)
    taus = {round(float(value), 6) for value in draws["tau"].to_numpy()}
    assert taus <= {round(value, 6) for value in by_forex.values()}


def test_part_taus_give_the_tau_of_a_hand_computed_case(universe: ModuleType):
    # crypto: good < middle < bad; forex: middle < good < bad; one decision per asset
    losses = pd.DataFrame(
        {
            "time": pd.Timestamp("2025-11-01", tz="UTC"),
            "asset": ["C1", "F1"],
            "window": "t1",
            "market": ["crypto", "forex"],
            "good": [1.0, 2.0],
            "middle": [2.0, 1.0],
            "bad": [3.0, 3.0],
        }
    )
    parts = universe.part_taus(losses, MODELS, column="market")
    np.testing.assert_allclose(parts["tau"], [1 / 3, 1 / 3])
    assert parts["decisions"].tolist() == [1, 1]
    assert not parts["same_top"].any()


def test_split_summary_gives_the_median_and_the_central_range(universe: ModuleType):
    splits = pd.DataFrame({"group": "hourly", "tau": np.linspace(0.0, 1.0, 101)})
    median, low, high = universe.split_summary(splits, "hourly", level=0.9)
    assert median == pytest.approx(0.5)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.95)
    with pytest.raises(ValueError, match="no split"):
        universe.split_summary(splits, "grid", level=0.9)
