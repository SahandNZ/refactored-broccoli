"""07-rq5-off-hours, Q2: the reopening gap on Friday's information and weekend crypto."""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import stdtr, stdtrit

from src.enums import Timeframe
from src.panel import hourly_closes
from src.stats import holm

START, END = "2022-01-01", "2024-01-01"


def _weekends(
    off_hours: ModuleType,
    settings: Any,
    off_hours_market: Callable[..., Any],
    *,
    beta: float,
    seed: int,
):
    bars, crypto_bars = off_hours_market(beta=beta, seed=seed, start=START, end=END)
    closes = hourly_closes({asset: {Timeframe.H1: frame} for asset, frame in crypto_bars.items()})
    crypto = pd.DataFrame({asset: np.log(series) for asset, series in closes.items()})
    rows = off_hours.weekend_rows(bars, crypto, settings)
    return rows.assign(asset="FX", market="forex")


def test_a_planted_crypto_beta_is_detected_and_recovered(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.8, seed=2)
    assets = off_hours.gap_tables(weekends, settings)["gap_assets"].iloc[0]
    assert assets["status"] == "tested"
    assert assets["is_yes"]
    assert assets["sign"] == "positive"
    assert assets["beta"] == pytest.approx(0.8, abs=0.05)
    assert assets["beta_low"] < 0.8 < assets["beta_high"]
    assert assets["r2_increment"] > 0.5
    assert assets["weekends"] + assets["incomplete"] == len(weekends)


def test_a_negative_beta_says_yes_with_a_negative_sign(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    # the test is two-sided: a safe-haven currency can move against crypto
    weekends = _weekends(off_hours, settings, off_hours_market, beta=-0.8, seed=0)
    assets = off_hours.gap_tables(weekends, settings)["gap_assets"].iloc[0]
    assert assets["t"] < -10
    assert assets["is_yes"]
    assert assets["sign"] == "negative"
    assert assets["beta_high"] < 0


def test_the_interval_and_the_p_value_come_from_one_hac_t(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.05, seed=4)
    row = off_hours.gap_statistics(weekends.dropna(), settings)
    degrees = row["weekends"] - 4
    error = row["beta"] / row["t"]
    critical = stdtrit(degrees, 0.975)
    assert row["beta_low"] == pytest.approx(row["beta"] - critical * abs(error))
    assert row["beta_high"] == pytest.approx(row["beta"] + critical * abs(error))
    assert row["p_value"] == pytest.approx(2 * stdtr(degrees, -abs(row["t"])))
    # the interval leaves out zero exactly when the unadjusted p is below 1 - level
    leaves_out_zero = row["beta_low"] > 0 or row["beta_high"] < 0
    assert leaves_out_zero == (row["p_value"] < 0.05)


def test_without_a_relation_the_rule_rarely_says_yes(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    # two-sided at alpha 0.05 (a little liberal with 4 HAC lags on about 100 weekends, see
    # the size check) and a sign-consistent rank correlation
    answers: list[bool] = []
    for seed in range(40):
        weekends = _weekends(off_hours, settings, off_hours_market, beta=0.0, seed=seed)
        answers.append(bool(off_hours.gap_tables(weekends, settings)["gap_assets"]["is_yes"][0]))
    assert sum(answers) <= 6


def test_the_two_sided_hac_t_holds_its_size(off_hours: ModuleType, settings: Any):
    # size check: no relation, 100 weekends with autocorrelated gaps; the two-sided test at
    # alpha 0.05 rejects about 0.07 of the time (Bartlett HAC with 4 lags is a little liberal
    # in small samples; the plan reports it as a limitation)
    rng = np.random.default_rng(7)
    rejections: list[bool] = []
    for _ in range(400):
        noise = rng.standard_normal(101)
        weekends = pd.DataFrame(
            {"friday_target": rng.standard_normal(100), "friday_crypto": rng.standard_normal(100),
             "weekend_crypto": rng.standard_normal(100), "gap": noise[1:] + 0.3 * noise[:-1]}
        )  # fmt: skip
        rejections.append(
            bool(off_hours.gap_statistics(weekends, settings)["p_value"] < settings.alpha)
        )
    assert 0.03 <= np.mean(rejections) <= 0.10


def test_the_trimmed_rank_correlation_leaves_out_the_largest_crypto_weekends(
    off_hours: ModuleType, settings: Any
):
    rng = np.random.default_rng(0)
    crypto = rng.uniform(-0.01, 0.01, 40)
    gap = -crypto
    largest = np.argsort(np.abs(crypto))[-settings.trim_largest :]
    crypto[largest] *= 30
    gap[largest] = crypto[largest]
    weekends = pd.DataFrame(
        {"friday_target": rng.normal(0, 1e-3, 40), "friday_crypto": rng.normal(0, 1e-3, 40),
         "weekend_crypto": crypto, "gap": gap + rng.normal(0, 1e-4, 40)}
    )  # fmt: skip
    row = off_hours.gap_statistics(weekends, settings)
    assert row["rank_corr_trimmed"] == pytest.approx(-1.0, abs=0.01)
    assert row["rank_corr"] > row["rank_corr_trimmed"]
    assert row["corr"] > 0.9


def test_groups_with_too_few_weekends_are_counted_not_tested(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.8, seed=0)
    tables = off_hours.gap_tables(weekends, settings)
    halves = tables["gap_half_years"]
    assert set(halves["half_year"]) == {"2022H1", "2022H2", "2023H1", "2023H2"}
    assert (halves["status"] == "tested").all()
    short = off_hours.gap_tables(weekends.iloc[:15], settings)["gap_assets"].iloc[0]
    assert short["status"] == "not tested: 15 weekends, fewer than 20"
    assert set(tables["gap_directions"]["direction"]) == {"up", "down"}
    np.testing.assert_allclose(tables["gap_directions"]["beta"], 0.8, atol=0.1)


def test_a_weekend_without_crypto_move_is_in_neither_direction(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.8, seed=0)
    weekends.loc[:4, "weekend_crypto"] = 0.0
    directions = off_hours.gap_tables(weekends, settings)["gap_directions"]
    assert directions["weekends"].sum() == len(weekends) - 5


def test_the_rule_needs_a_trimmed_rank_correlation_of_the_coefficients_sign(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.8, seed=2)
    rng = np.random.default_rng(0)
    # a few huge weekends carry the coefficient; the rest move against crypto
    kept = np.argsort(np.abs(weekends["weekend_crypto"].to_numpy()))[: len(weekends) - 5]
    weekends.loc[kept, "gap"] = -weekends.loc[kept, "weekend_crypto"] + rng.normal(
        0, 1e-5, len(kept)
    )
    large = weekends.index.difference(kept)
    weekends.loc[large, "weekend_crypto"] *= 100
    weekends.loc[large, "gap"] = weekends.loc[large, "weekend_crypto"]
    assets = off_hours.gap_tables(weekends, settings)["gap_assets"].iloc[0]
    assert assets["beta"] > 0
    assert assets["rank_corr_trimmed"] < 0
    assert not assets["is_yes"]


def test_holm_adjusts_the_assets_together(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = pd.concat(
        [_weekends(off_hours, settings, off_hours_market, beta=beta, seed=seed).assign(asset=name)
         for name, beta, seed in [("A", 0.8, 0), ("B", 0.0, 1), ("C", 0.0, 2)]],
        ignore_index=True,
    )  # fmt: skip
    assets = off_hours.gap_tables(weekends, settings)["gap_assets"].set_index("asset")
    np.testing.assert_allclose(assets["adjusted_p"], holm(assets["p_value"].to_numpy()))
    assert assets.loc["A", "is_yes"]
    assert list(assets.index) == ["A", "B", "C"]


def test_an_asset_without_a_complete_weekend_is_listed_not_tested(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    weekends = _weekends(off_hours, settings, off_hours_market, beta=0.8, seed=2)
    empty = weekends.assign(asset="EMPTY", weekend_crypto=np.nan)
    assets = off_hours.gap_tables(pd.concat([weekends, empty]), settings)["gap_assets"]
    row = assets.set_index("asset").loc["EMPTY"]
    assert row["status"] == "not tested: no complete weekend"
    assert row["weekends"] == 0
    assert row["incomplete"] == len(weekends)
    assert not row["is_yes"]
