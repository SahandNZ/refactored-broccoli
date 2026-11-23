import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import t

from src.stats import (
    MIN_BOOTSTRAP_BLOCKS,
    andrews_lags,
    bootstrap_difference,
    bootstrap_interval,
    bootstrap_mean_test,
    bootstrap_p_value,
    circular_block_indices,
    diebold_mariano,
    dm_statistic,
    has_enough_blocks,
    holm,
    newey_west_ols,
    newey_west_variance,
    politis_white_block,
)

series = arrays(float, st.integers(10, 60), elements=st.floats(-100, 100, allow_nan=False))
SETTINGS = {"level": 0.9, "n_resamples": 300, "seed": 3, "blocks": (5, 5)}
INTERVAL = {"level": 0.9, "n_resamples": 300, "seed": 3, "block": 5}


# --- bootstrap_difference -----------------------------------------------------------------


@given(first=series, second=series, shift=st.floats(-50, 50, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_a_shift_of_both_periods_cancels_out(first, second, shift):
    plain = bootstrap_difference(first, second, **SETTINGS)
    shifted = bootstrap_difference(first + shift, second + shift, **SETTINGS)
    for key in ("difference", "low", "high"):
        assert shifted[key] == pytest.approx(plain[key], abs=1e-7)


@given(first=series, second=series)
@settings(max_examples=50, deadline=None)
def test_the_difference_is_the_difference_of_the_means_inside_an_ordered_interval(first, second):
    result = bootstrap_difference(first, second, **SETTINGS)
    assert result["difference"] == pytest.approx(first.mean() - second.mean(), abs=1e-9)
    assert result["low"] <= result["high"]
    assert 0 <= result["p_value"] <= 1


def test_a_clear_difference_is_significant_and_no_difference_is_not():
    rng = np.random.default_rng(0)
    first = rng.normal(1.0, 1.0, 500)
    second = rng.normal(0.0, 1.0, 400)
    options = {"level": 0.95, "n_resamples": 1000, "seed": 0, "blocks": (10, 10)}
    clear = bootstrap_difference(first, second, **options)
    assert clear["p_value"] < 1e-6
    assert 0.8 < clear["low"] < 1.0 < clear["high"] < 1.2
    same = bootstrap_difference(second[:200], second[200:], **options)
    assert same["p_value"] > 0.05
    assert same["low"] < 0 < same["high"]


def test_the_block_bootstrap_widens_the_interval_of_autocorrelated_values():
    rng = np.random.default_rng(1)
    walk = np.repeat(rng.normal(0.0, 1.0, 50), 20)
    other = rng.normal(0.0, 1.0, 1000)
    options = {"level": 0.95, "n_resamples": 1000, "seed": 0}
    single = bootstrap_difference(walk, other, **options, blocks=(1, 1))
    blocks = bootstrap_difference(walk, other, **options, blocks=(40, 1))
    assert blocks["high"] - blocks["low"] > 2 * (single["high"] - single["low"])


def test_each_period_is_resampled_with_its_own_block():
    rng = np.random.default_rng(5)
    walk = np.repeat(rng.normal(0.0, 1.0, 50), 20)
    other = rng.normal(0.0, 1.0, 30)
    options = {"level": 0.95, "n_resamples": 1000, "seed": 0}
    first_long = bootstrap_difference(walk, other, **options, blocks=(40, 1))
    with pytest.raises(ValueError, match="need at least 2 blocks of 40 values, got 30 values"):
        bootstrap_difference(walk, other, **options, blocks=(40, 40))
    short = bootstrap_difference(walk, other, **options, blocks=(1, 1))
    assert first_long["high"] - first_long["low"] > short["high"] - short["low"]


def test_constant_periods_give_a_p_value_of_zero_or_one():
    ones, twos = np.ones(10), np.full(10, 2.0)
    assert bootstrap_difference(ones, ones, **SETTINGS)["p_value"] == 1.0
    different = bootstrap_difference(twos, ones, **SETTINGS)
    assert different["p_value"] == 0.0
    assert different["low"] == pytest.approx(1.0)
    assert different["high"] == pytest.approx(1.0)


def test_the_draws_are_reproducible_from_the_seed():
    rng = np.random.default_rng(2)
    first, second = rng.normal(size=100), rng.normal(size=80)
    assert bootstrap_difference(first, second, **SETTINGS) == bootstrap_difference(
        first, second, **SETTINGS
    )


@pytest.mark.parametrize(("length", "block"), [(10, 0), (10, 11), (4, 5)])
def test_a_block_outside_one_to_the_period_length_raises(length, block):
    values = np.arange(float(length))
    with pytest.raises(ValueError, match=r"block must be between 1 and|need at least 2 blocks"):
        bootstrap_difference(values, np.arange(20.0), **{**SETTINGS, "blocks": (block, 5)})
    with pytest.raises(ValueError, match="block must be between 1 and"):
        bootstrap_interval(values, **{**INTERVAL, "block": block})


def test_a_period_needs_two_blocks():
    with pytest.raises(ValueError, match="need at least 2 blocks of 5 values, got 5 values"):
        bootstrap_difference(np.arange(5.0), np.arange(5.0, 15.0), **SETTINGS)
    result = bootstrap_difference(np.arange(10.0), np.arange(10.0, 20.0), **SETTINGS)
    assert result["difference"] == pytest.approx(-10.0)
    assert result["low"] < -10.0 < result["high"]


# --- has_enough_blocks, circular_block_indices, bootstrap_mean_test -------------------------


def test_a_test_needs_five_blocks():
    assert MIN_BOOTSTRAP_BLOCKS == 5
    assert has_enough_blocks(50, 10)
    assert not has_enough_blocks(49, 10)
    assert has_enough_blocks(5, 1)


def test_circular_blocks_wrap_around_and_draw_every_value_equally():
    rng = np.random.default_rng(0)
    indices = circular_block_indices(10, rng, n_resamples=4000, block=4)
    assert indices.shape == (4000, 10)
    differences = np.diff(indices, axis=1) % 10
    assert set(np.unique(differences)) <= set(range(10))
    counts = np.bincount(indices.ravel(), minlength=10) / indices.size
    np.testing.assert_allclose(counts, 0.1, atol=0.005)
    with pytest.raises(ValueError, match="block must be between 1 and the 10 values, got 11"):
        circular_block_indices(10, rng, n_resamples=1, block=11)


def test_the_bootstrap_mean_test_keeps_its_size_with_overlapping_labels():
    # 24-bar overlapping sums of white noise, no edge, 12 blocks of 48: the one-sided rejection
    # rate at 5 % (measured 0.083 here; RQ4's size check gives 0.055-0.085 for 5-20 blocks)
    rejected = []
    for seed in range(300):
        noise = np.random.default_rng(seed).standard_normal(624)
        values = np.convolve(noise, np.ones(24), mode="valid")[:600]
        test = bootstrap_mean_test(
            values, null=0.0, level=0.95, n_resamples=200, seed=seed, block=48
        )
        rejected.append(test["p_greater"] < 0.05)
    assert 0.01 < np.mean(rejected) < 0.11


def test_the_mean_test_detects_a_clear_mean_and_gives_its_interval():
    values = np.random.default_rng(1).normal(0.5, 1.0, 400)
    test = bootstrap_mean_test(values, null=0.0, level=0.95, n_resamples=500, seed=0, block=10)
    assert test["mean"] == pytest.approx(values.mean())
    assert test["p_greater"] < 1e-4
    assert test["low"] < test["mean"] < test["high"]
    assert test["df"] == 400 // 10 - 1


# --- bootstrap_p_value --------------------------------------------------------------------

DRAWS = {"n_resamples": 300, "seed": 3, "block": 5}


@given(values=series, null=st.floats(-100, 100, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_the_p_value_is_a_probability_and_symmetric_around_the_null(values, null):
    p_value = bootstrap_p_value(values, null=null, **DRAWS)
    assert 0 <= p_value <= 1
    mirrored = bootstrap_p_value(2 * null - values, null=null, **DRAWS)
    assert mirrored == pytest.approx(p_value, abs=1e-9)


@given(values=series, shift=st.floats(0, 50, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_the_p_value_falls_as_the_mean_moves_away_from_the_null(values, shift):
    null = float(values.mean()) - 1.0
    near = bootstrap_p_value(values, null=null, **DRAWS)
    far = bootstrap_p_value(values + shift, null=null, **DRAWS)
    assert far <= near + 1e-12


def test_the_p_value_uses_the_draws_of_the_interval():
    rng = np.random.default_rng(4)
    values = rng.normal(0.55, 0.3, 400)
    low, high = bootstrap_interval(values, level=0.95, **DRAWS)
    assert low > 0.5
    assert bootstrap_p_value(values, null=0.5, **DRAWS) < 0.05
    assert low < 0.55 < high
    assert bootstrap_p_value(values, null=float(values.mean()), **DRAWS) == pytest.approx(1.0)


def test_constant_values_give_a_p_value_of_zero_or_one():
    assert bootstrap_p_value(np.full(10, 0.5), null=0.5, **DRAWS) == 1.0
    assert bootstrap_p_value(np.full(10, 0.6), null=0.5, **DRAWS) == 0.0


def test_the_p_value_refuses_a_block_longer_than_the_values():
    with pytest.raises(ValueError, match="block must be between 1 and the 4 values, got 5"):
        bootstrap_p_value(np.arange(4.0), null=0.0, **DRAWS)


# --- holm ---------------------------------------------------------------------------------


def test_holm_multiplies_the_sorted_p_values_and_keeps_their_order():
    adjusted = holm(np.array([0.04, 0.01, 0.03, 0.02]))
    np.testing.assert_allclose(adjusted, [0.06, 0.04, 0.06, 0.06])


def test_cells_without_a_test_stay_nan_and_do_not_count():
    with_nan = holm(np.array([0.01, np.nan, 0.02, np.nan]))
    assert np.isnan(with_nan[[1, 3]]).all()
    np.testing.assert_allclose(with_nan[[0, 2]], holm(np.array([0.01, 0.02])))
    np.testing.assert_allclose(with_nan[[0, 2]], [0.02, 0.02])
    assert np.isnan(holm(np.array([np.nan]))).all()
    assert holm(np.array([])).shape == (0,)


@given(arrays(float, st.integers(1, 30), elements=st.floats(0, 1)))
def test_holm_never_lowers_a_p_value_and_stays_at_most_one(p_values):
    adjusted = holm(p_values)
    assert (adjusted >= p_values - 1e-12).all()
    assert (adjusted <= 1.0).all()


# --- newey_west_ols ------------------------------------------------------------------------


def _design(n, rng):
    return np.column_stack([np.ones(n), rng.standard_normal((n, 2))])


@pytest.mark.parametrize("lags", [0, 1, 4])
def test_newey_west_matches_statsmodels_hac_with_its_small_sample_correction(lags):
    statsmodels = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(lags)
    design = _design(120, rng)
    noise = np.convolve(rng.standard_normal(123), np.ones(4) / 2, mode="valid")
    response = design @ np.array([0.1, 0.5, -0.2]) + noise
    fit = newey_west_ols(response, design, lags=lags)
    reference = statsmodels.OLS(response, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    np.testing.assert_allclose(fit.coefficients, reference.params)
    np.testing.assert_allclose(fit.t, reference.tvalues)
    assert fit.r2 == pytest.approx(reference.rsquared)


def test_newey_west_recovers_a_planted_slope_and_rejects_none():
    rng = np.random.default_rng(0)
    design = _design(300, rng)
    planted = newey_west_ols(design[:, 1] * 0.8 + rng.standard_normal(300), design, lags=4)
    assert planted.coefficients[1] == pytest.approx(0.8, abs=0.15)
    assert planted.t[1] > 8
    assert abs(planted.t[2]) < 3


def test_newey_west_refuses_too_few_rows_and_negative_lags():
    design = _design(3, np.random.default_rng(0))
    with pytest.raises(ValueError, match="more rows than columns"):
        newey_west_ols(np.zeros(3), design, lags=0)
    with pytest.raises(ValueError, match="lags >= 0"):
        newey_west_ols(np.zeros(10), _design(10, np.random.default_rng(0)), lags=-1)


# --- diebold_mariano, andrews_lags, newey_west_variance -----------------------------------


def _ar1(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    noise = rng.standard_normal(n)
    for index in range(1, n):
        values[index] = rho * values[index - 1] + noise[index]
    return values


def test_andrews_lags_follow_the_ar1_plug_in_rule() -> None:
    values = _ar1(0.5, 400, 0)
    centred = values - values.mean()
    rho = float(centred[1:] @ centred[:-1] / (centred @ centred))
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    assert andrews_lags(values) == int(1.1447 * (alpha * 400) ** (1 / 3))
    assert andrews_lags(_ar1(0.9, 400, 0)) > andrews_lags(values) > andrews_lags(_ar1(0.0, 400, 0))
    assert andrews_lags(np.ones(10)) == 0
    assert andrews_lags(np.array([1.0])) == 0


def test_the_newey_west_variance_is_the_bartlett_sum_of_autocovariances() -> None:
    centred = np.array([[1.0, -2.0, 3.0, -1.0, -1.0]])
    gamma = [float(centred[0, lag:] @ centred[0, : 5 - lag]) / 5 for lag in range(3)]
    expected = gamma[0] + 2 * (2 / 3 * gamma[1] + 1 / 3 * gamma[2])
    assert newey_west_variance(centred, lags=2)[0] == pytest.approx(expected)
    assert newey_west_variance(centred, lags=0)[0] == pytest.approx(gamma[0])


@pytest.mark.parametrize(("overlap", "rho"), [(0, 0.0), (3, 0.2), (23, 0.6)])
def test_diebold_mariano_matches_statsmodels_hac_with_the_hln_correction(overlap, rho) -> None:
    # known answer: the HAC standard error of the mean from statsmodels (Bartlett kernel, the
    # same lags, no small-sample factor), the HLN factor and Student's t with n - 1 df
    sm = pytest.importorskip("statsmodels.api")

    n = 600
    loss_b = 1.0 + np.abs(_ar1(rho, n, 1))
    loss_a = loss_b + 0.05 + 0.3 * _ar1(rho, n, 2)
    result = diebold_mariano(loss_a, loss_b, overlap=overlap)
    difference = loss_a - loss_b
    lags = max(overlap, andrews_lags(difference))
    assert result["lags"] == lags
    fit = sm.OLS(difference, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    h = overlap + 1
    factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = float(fit.params[0] / fit.bse[0]) * factor
    assert result["statistic"] == pytest.approx(statistic, rel=1e-9)
    assert result["p_less"] == pytest.approx(t.cdf(statistic, n - 1), rel=1e-9)
    assert result["p_value"] == pytest.approx(2 * t.sf(abs(statistic), n - 1), rel=1e-9)
    assert result["mean_difference"] == pytest.approx(difference.mean())


def test_the_one_sided_p_value_favours_the_smaller_loss() -> None:
    rng = np.random.default_rng(5)
    base = 1.0 + rng.random(300)
    better = base - 0.2 + 0.1 * rng.standard_normal(300)
    forward = diebold_mariano(better, base, overlap=0)
    backward = diebold_mariano(base, better, overlap=0)
    assert forward["statistic"] < 0
    assert forward["p_less"] < 0.01
    assert backward["p_less"] > 0.99
    assert forward["p_less"] + backward["p_less"] == pytest.approx(1.0)
    assert forward["p_value"] == pytest.approx(2 * min(forward["p_less"], 1 - forward["p_less"]))


def test_a_loss_difference_without_variance_has_no_test() -> None:
    # a constant difference must not be "detected" with an infinite statistic
    loss = np.linspace(1.0, 2.0, 50)
    for shift in (0.0, -0.1):
        result = diebold_mariano(loss + shift, loss, overlap=2)
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_less"])
        assert np.isnan(result["p_value"])
    assert np.isnan(dm_statistic(0.1, 0.0, n=50, overlap=0))
    assert np.isnan(dm_statistic(0.1, 1.0, n=3, overlap=2))


def test_diebold_mariano_refuses_a_negative_overlap_and_one_value() -> None:
    with pytest.raises(ValueError, match="overlap >= 0"):
        diebold_mariano(np.ones(5), np.zeros(5), overlap=-1)
    with pytest.raises(ValueError, match="at least 2 values"):
        diebold_mariano(np.ones(1), np.zeros(1), overlap=0)


def test_under_the_null_the_one_sided_test_rejects_at_about_alpha() -> None:
    # size check: overlapping h-step differences (moving sums of white noise) with no edge
    rejected = []
    for seed in range(400):
        noise = np.random.default_rng(seed).standard_normal(424)
        difference = np.convolve(noise, np.ones(24), mode="valid")[:400]
        rejected.append(diebold_mariano(difference, np.zeros(400), overlap=23)["p_less"] < 0.05)
    assert 0.02 < np.mean(rejected) < 0.1


def test_politis_white_blocks_grow_with_autocorrelation() -> None:
    white = politis_white_block(_ar1(0.0, 2000, 3))
    persistent = politis_white_block(_ar1(0.9, 2000, 3))
    assert 1 <= white < persistent
    assert politis_white_block(np.ones(100)) == 1
    # a constant float series has a tiny but non-zero std (about 1e-17); it is still constant
    assert politis_white_block(np.full(25, 0.1)) == 1
    assert politis_white_block(np.arange(5.0)) == 1
