import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS, hit_rate, mase, quantile_crps, rank_ic
from src.stats import diebold_mariano, paired_t


def test_folds_are_back_to_back_and_development_before_test():
    folds = DEVELOPMENT_FOLDS + TEST_FOLDS
    for before, after in itertools.pairwise(folds):
        assert before.evaluation_end == after.evaluation_start
    assert not any(fold.is_test for fold in DEVELOPMENT_FOLDS)
    assert all(fold.is_test for fold in TEST_FOLDS)
    assert TEST_FOLDS[0].evaluation_start == pd.Timestamp("2025-11-01", tz="UTC")


def test_crps_is_zero_for_a_perfect_forecast_and_grows_with_error():
    levels = np.array([0.1, 0.5, 0.9])
    target = np.array([1.0, 2.0])
    perfect = np.repeat(target[:, None], 3, axis=1)
    assert quantile_crps(perfect, levels, target) == pytest.approx([0.0, 0.0])
    assert (
        quantile_crps(perfect + 1, levels, target) > quantile_crps(perfect + 0.5, levels, target)
    ).all()


def test_mase_and_its_guard():
    assert mase(np.array([1.0, 2.0]), np.array([2.0, 2.0]), 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="naive_scale"):
        mase(np.array([1.0]), np.array([1.0]), 0.0)


@pytest.mark.parametrize(
    ("forecast", "realised"),
    [
        ([0.5, 0.5, 0.5], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]),
        ([1.0, np.nan, np.nan], [1.0, 2.0, 3.0]),
        ([], []),
    ],
)
def test_rank_ic_without_variation_or_pairs_is_nan_without_a_warning(
    forecast: list[float], realised: list[float]
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = rank_ic(pd.Series(forecast, dtype=float), pd.Series(realised, dtype=float))
    assert np.isnan(result)


def test_rank_ic_and_hit_rate():
    forecast = pd.Series([1.0, 2.0, 3.0, np.nan])
    assert rank_ic(forecast, pd.Series([10.0, 20.0, 30.0, 40.0])) == pytest.approx(1.0)
    assert hit_rate(
        np.array([1.0, -1.0, 0.0, 2.0]), np.array([1.0, 1.0, 1.0, 3.0])
    ) == pytest.approx(2 / 3)


def test_paired_t():
    result = paired_t(np.array([1.0, 2.0, 3.0]))
    assert result["mean"] == pytest.approx(2.0)
    assert result["wins"] == 3
    with pytest.raises(ValueError, match="2 pairs"):
        paired_t(np.array([1.0]))


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(0)
    target = rng.normal(size=2000)
    good = (target - (target + rng.normal(0, 0.5, 2000))) ** 2
    bad = (target - (target + rng.normal(0, 1.0, 2000))) ** 2
    result = diebold_mariano(bad, good, overlap=0)
    assert result["statistic"] > 0
    assert result["p_value"] < 0.01
    same = diebold_mariano(good, good + rng.normal(0, 1e-6, 2000), overlap=3)
    assert same["p_value"] > 0.01


def test_verdict_claims_only_significant_gaps():
    from src.stats import verdict  # noqa: PLC0415

    assert verdict(np.array([1.0, 1.1, 0.9, 1.0])) == "better"
    assert verdict(np.array([-1.0, -1.1, -0.9])) == "worse"
    assert verdict(np.array([0.1, -0.2, 0.15, -0.05])) == "no difference"
    assert verdict(np.array([1.0, 1.0])) == "too few runs"


def test_holm_and_bootstrap_interval():
    from src.stats import bootstrap_interval, holm  # noqa: PLC0415

    assert holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    values = np.random.default_rng(0).normal(1.0, 1.0, 2000)
    low, high = bootstrap_interval(values, level=0.95, n_resamples=500, seed=0, block=20)
    assert low < values.mean() < high
    assert low > 0.8
    assert high < 1.2
