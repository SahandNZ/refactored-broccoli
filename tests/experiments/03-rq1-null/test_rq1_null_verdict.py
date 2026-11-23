"""The decision rule of 03-rq1-null (its lib's `null_verdict`) on hand-made and cache cells."""

import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.benchmark import Bootstrap, NullRule, StudySettings, read_cache
from src.benchmark._scores import null_tables
from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS
from src.stats import holm

MODELS = ["random-walk", "climatology", "skilled", "sampled"]
BOOTSTRAP = Bootstrap(level=0.95, n_resamples=100, seed=0, block_hours=24, min_block=4)
SETTINGS = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, BOOTSTRAP)
RULE = NullRule(
    min_winning_quarters=3, main_family_min_quarters=3, skill_resamples=100,
    improvements_pct=(0.0, 10.0), power_target=0.8, mde_horizon=1,
    threshold_grid_pct=(0.0, 10.0), power_share=0.8, power_share_sensitivity=(0.5, 0.9),
    mde_threshold_pct=2.0,
)  # fmt: skip


def _cells(p_values, winning_quarters, statistics, families=None):
    return pd.DataFrame(
        {
            "p_value": p_values,
            "winning_quarters": winning_quarters,
            "dm_statistic": statistics,
            "family": families if families is not None else ["main"] * len(p_values),
        }
    )


@pytest.mark.parametrize(("wins", "expected"), [(4, True), (3, True), (2, False), (0, False)])
def test_a_cell_needs_at_least_three_winning_quarters(rq1: ModuleType, wins, expected) -> None:
    verdict = rq1.null_verdict(_cells([1e-6], [wins], [-5.0]), alpha=0.05, min_winning_quarters=3)
    assert verdict["beats_baseline"].tolist() == [expected]


def test_holm_adjusts_over_the_scored_cells(rq1: ModuleType) -> None:
    p_values = [0.01, 0.02, 0.03, 0.04]
    alone = rq1.null_verdict(_cells([0.01], [4], [-3.0]), alpha=0.05, min_winning_quarters=3)
    assert alone["beats_baseline"].tolist() == [True]
    together = rq1.null_verdict(
        _cells(p_values, [4] * 4, [-3.0] * 4), alpha=0.05, min_winning_quarters=3
    )
    np.testing.assert_allclose(together["adjusted_p"], [0.04, 0.06, 0.06, 0.06])
    assert together["beats_baseline"].tolist() == [True, False, False, False]
    with_unscored = rq1.null_verdict(
        _cells([0.01, np.nan, np.nan], [4, 0, 0], [-3.0, np.nan, np.nan]),
        alpha=0.05, min_winning_quarters=3,
    )  # fmt: skip
    assert with_unscored["adjusted_p"].iloc[0] == pytest.approx(0.01)
    assert with_unscored["beats_baseline"].tolist() == [True, False, False]


def test_an_adjusted_p_equal_to_alpha_is_not_significant(rq1: ModuleType) -> None:
    verdict = rq1.null_verdict(
        _cells([0.025, 0.5], [4, 4], [-3.0, -1.0]), alpha=0.05, min_winning_quarters=3
    )
    assert verdict["adjusted_p"].tolist() == pytest.approx([0.05, 0.5])
    assert verdict["beats_baseline"].tolist() == [False, False]


def test_a_significantly_worse_model_does_not_beat_the_baseline(rq1: ModuleType) -> None:
    verdict = rq1.null_verdict(_cells([1e-6], [3], [5.0]), alpha=0.05, min_winning_quarters=3)
    assert verdict["beats_baseline"].tolist() == [False]


def _null_cells(quarters_cache_folder: Path, tmp_path: Path, model: str, release: str):
    # the cells of null_tables after the release date of one model of the cache is changed
    folder = tmp_path / "cache"
    shutil.copytree(quarters_cache_folder, folder)
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == model, "release_date"] = release
    models.to_csv(folder / "models.csv", index=False)
    return null_tables(read_cache(folder), MODELS, SETTINGS, RULE)["cells"]


def test_a_model_scored_in_too_few_quarters_does_not_beat_the_baseline(
    rq1: ModuleType, quarters_cache_folder: Path, tmp_path: Path
) -> None:
    cells = _null_cells(quarters_cache_folder, tmp_path, "skilled", "2026-03-01")
    verdict = rq1.null_verdict(cells, alpha=0.05, min_winning_quarters=3)
    assert not verdict["beats_baseline"].any()


def test_holm_leaves_out_the_cells_of_a_model_released_after_the_scored_folds(
    rq1: ModuleType, quarters_cache_folder: Path, tmp_path: Path
) -> None:
    cells = _null_cells(quarters_cache_folder, tmp_path, "sampled", "2027-01-01")
    verdict = rq1.null_verdict(cells, alpha=0.05, min_winning_quarters=3)
    scored = verdict[verdict["model"] == "skilled"]
    np.testing.assert_allclose(scored["adjusted_p"], holm(scored["p_value"].to_numpy()))


def test_holm_runs_within_each_family(rq1: ModuleType) -> None:
    # the late family's cells do not raise the main family's m, and get their own verdict
    cells = _cells([0.01, 0.02, 0.03, 0.04], [4] * 4, [-3.0] * 4, ["main", "main", "late", "late"])
    verdict = rq1.null_verdict(cells, alpha=0.05, min_winning_quarters=3)
    np.testing.assert_allclose(verdict["adjusted_p"], [0.02, 0.02, 0.06, 0.06])
    assert verdict["beats_baseline"].tolist() == [True, True, False, False]


def test_a_model_released_inside_the_window_forms_the_late_family(
    rq1: ModuleType, quarters_cache_folder: Path, tmp_path: Path
) -> None:
    cells = _null_cells(quarters_cache_folder, tmp_path, "skilled", "2026-03-01")
    verdict = rq1.null_verdict(cells, alpha=0.05, min_winning_quarters=3)
    families = verdict.groupby("model")["family"].first()
    assert families.to_dict() == {"sampled": "main", "skilled": "late"}
    for _, family in verdict.groupby("family"):
        scored = family[family["status"] == "scored"]
        np.testing.assert_allclose(scored["adjusted_p"], holm(scored["p_value"].to_numpy()))
