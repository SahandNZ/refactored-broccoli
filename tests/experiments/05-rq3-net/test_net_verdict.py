"""The decision rule of 05-rq3-net (its ``lib``): `net_verdict`."""

import dataclasses
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.benchmark import Bootstrap, ForecastCache, StudySettings, read_cache
from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS
from src.stats import holm

SETTINGS = StudySettings(
    DEVELOPMENT_FOLDS,
    TEST_FOLDS,
    0.05,
    Bootstrap(level=0.95, n_resamples=200, seed=0, block_hours=24, min_block=4),
)
RULE = {"alpha": 0.05, "main_min_quarters": 3, "min_blocks": 10.0, "min_turnover": 0.05}


def _cells(
    *,
    quarters: list[int],
    horizons: list[int],
    p_values: list[float],
    sharpes: list[float] | None = None,
    blocks: list[float] | None = None,
    turnover: list[float] | None = None,
) -> pd.DataFrame:
    n = len(p_values)
    return pd.DataFrame(
        {
            "post_release_quarters": quarters,
            "horizon": horizons,
            "p_value": p_values,
            "net_sharpe": sharpes if sharpes is not None else [1.0] * n,
            "blocks": blocks if blocks is not None else [50.0] * n,
            "turnover": turnover if turnover is not None else [0.5] * n,
            "times": [100] * n,
            "effective_times": [50] * n,
            "status": ["not scored: never trades" if math.isnan(p) else "scored" for p in p_values],
        }
    )


def test_holm_runs_within_each_family_over_all_horizons(rq3: ModuleType) -> None:
    cells = _cells(
        quarters=[4, 4, 4, 3, 2, 1],
        horizons=[1, 1, 4, 24, 1, 4],
        p_values=[0.01, 0.03, 0.012, 0.02, 0.02, 0.06],
    )
    verdict = rq3.net_verdict(cells, **RULE)
    assert verdict["family"].tolist() == ["main", "main", "main", "main", "late", "late"]
    main = holm(np.array([0.01, 0.03, 0.012, 0.02]))
    np.testing.assert_allclose(verdict["adjusted_p"].iloc[:4], main)
    np.testing.assert_allclose(verdict["adjusted_p"].iloc[4:], holm(np.array([0.02, 0.06])))
    # main: 0.04, 0.04, 0.04, 0.04 (one family of 4 across the horizons); late: 0.04, 0.06
    assert verdict["earns_after_costs"].tolist() == [True, True, True, True, True, False]


def test_a_rarely_trading_book_is_flagged_without_changing_the_verdict(rq3: ModuleType) -> None:
    # turnover per effective time = turnover x times / effective times = turnover x 2
    cells = _cells(quarters=[4, 4], horizons=[1, 1], p_values=[0.01, 0.01], turnover=[0.02, 0.03])
    verdict = rq3.net_verdict(cells, **RULE)
    np.testing.assert_allclose(verdict["turnover_per_effective"], [0.04, 0.06])
    assert verdict["is_low_turnover"].tolist() == [True, False]
    assert verdict["earns_after_costs"].tolist() == [True, True]
    # a cell that is not scored (never trades, or too short) is never flagged
    unscored = _cells(quarters=[4], horizons=[1], p_values=[np.nan], turnover=[0.0])
    assert not rq3.net_verdict(unscored, **RULE)["is_low_turnover"].any()


def test_a_cell_needs_a_positive_sharpe_and_a_small_adjusted_p(rq3: ModuleType) -> None:
    cells = _cells(
        quarters=[4, 4, 4],
        horizons=[1, 4, 24],
        p_values=[0.01, 0.06, 0.001],
        sharpes=[1.0, 1.0, -1.0],
    )
    verdict = rq3.net_verdict(cells, **RULE)
    assert verdict["earns_after_costs"].tolist() == [True, False, False]


def test_a_cell_without_a_test_is_left_out_of_holm(rq3: ModuleType) -> None:
    cells = _cells(quarters=[4, 4], horizons=[1, 1], p_values=[np.nan, 0.03])
    verdict = rq3.net_verdict(cells, **RULE)
    assert np.isnan(verdict["adjusted_p"].iloc[0])
    assert verdict["adjusted_p"].iloc[1] == pytest.approx(0.03)
    assert verdict["earns_after_costs"].tolist() == [False, True]


def test_few_blocks_are_flagged_without_changing_the_verdict(rq3: ModuleType) -> None:
    cells = _cells(
        quarters=[4, 4, 4],
        horizons=[1, 4, 24],
        p_values=[0.01, 0.01, np.nan],
        blocks=[9.5, 10.0, np.nan],
    )
    verdict = rq3.net_verdict(cells, **RULE)
    assert verdict["has_few_blocks"].tolist() == [True, False, False]
    assert verdict["earns_after_costs"].tolist() == [True, True, False]


# --- on the synthetic cache -----------------------------------------------------------------


@pytest.fixture(name="cache", scope="module")
def _cache(quarters_cache_folder: Path) -> ForecastCache:
    return read_cache(quarters_cache_folder)


@pytest.fixture(name="zero_shot", scope="module")
def _zero_shot(rq3: ModuleType, cache: ForecastCache) -> pd.DataFrame:
    cells = rq3.net_tables(cache, ["skilled", "sampled"], SETTINGS)["cells"]
    return cells[cells["kind"] == "zero-shot"]


def test_the_skilled_model_earns_after_costs_and_the_noise_model_does_not(
    rq3: ModuleType, zero_shot: pd.DataFrame
) -> None:
    verdict = rq3.net_verdict(zero_shot, **RULE)
    skilled = verdict[verdict["model"] == "skilled"]
    assert skilled["earns_after_costs"].all()
    assert (skilled["net_low"] > 0).all()
    assert (skilled["break_even_low"] > skilled["cost_bps"]).all()
    assert not verdict.loc[verdict["model"] == "sampled", "earns_after_costs"].any()
    assert (verdict["family"] == "main").all()


def test_costs_above_the_break_even_turn_the_answer_to_no(
    rq3: ModuleType, cache: ForecastCache, zero_shot: pd.DataFrame
) -> None:
    break_even = zero_shot.loc[zero_shot["model"] == "skilled", "break_even_bps"]
    factor = 2 * break_even.max() / 0.4
    assets = cache.assets.assign(cost_per_side_bps=cache.assets["cost_per_side_bps"] * factor)
    costly = dataclasses.replace(cache, assets=assets)
    cells = rq3.net_tables(costly, ["skilled"], SETTINGS)["cells"]
    verdict = rq3.net_verdict(cells[cells["kind"] == "zero-shot"], **RULE)
    assert not verdict["earns_after_costs"].any()
    assert (verdict["net_sharpe"] < 0).all()
    assert (verdict["p_value"] > 0.5).all()


def test_a_late_release_gets_its_own_family(rq3: ModuleType, cache: ForecastCache) -> None:
    models = cache.models.copy()
    models.loc[models["model"] == "sampled", "release_date"] = pd.Timestamp("2026-03-01", tz="UTC")
    later = dataclasses.replace(cache, models=models)
    cells = rq3.net_tables(later, ["skilled", "sampled"], SETTINGS)["cells"]
    verdict = rq3.net_verdict(cells[cells["kind"] == "zero-shot"], **RULE)
    families = verdict.groupby("model")["family"].unique().to_dict()
    assert families["skilled"].tolist() == ["main"]
    assert families["sampled"].tolist() == ["late"]
    for model in ("skilled", "sampled"):
        cells_of = verdict[verdict["model"] == model]
        np.testing.assert_allclose(cells_of["adjusted_p"], holm(cells_of["p_value"].to_numpy()))
