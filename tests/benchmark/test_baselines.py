import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

import src.benchmark._baselines as baselines_module
from src.benchmark._baselines import (
    NORMAL_QUANTILES,
    climatology,
    ewma_normal,
    garch_normal,
    lag_features,
    linear,
    random_walk,
)
from src.benchmark._contexts import context_ends
from src.models import QUANTILE_LEVELS

START = pd.Timestamp("2025-01-01", tz="UTC")


def _closes(n: int, seed: int, *, start: str = "2024-12-01") -> pd.Series:
    close_times = pd.date_range(start, periods=n, freq="1h", tz="UTC") + pd.Timedelta(hours=1)
    steps = np.random.default_rng(seed).normal(0, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=close_times)


def _rows(times: pd.DatetimeIndex, asset: str, horizon: int, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"time": times, "asset": asset, "horizon": horizon, "label": labels, "sigma": 0.01}
    )


def test_normal_quantiles_match_scipy() -> None:
    np.testing.assert_allclose(NORMAL_QUANTILES, norm.ppf(np.asarray(QUANTILE_LEVELS)), atol=1e-12)


def test_random_walk_is_zero_centred_with_the_root_mean_square_spread() -> None:
    times = pd.date_range("2024-12-01", periods=4, freq="1h", tz="UTC")
    train = pd.concat(
        [
            _rows(times, "A", 1, np.array([1.0, 1.0, 3.0, 3.0])),
            _rows(times, "A", 4, np.array([2.0, -2.0, 2.0, -2.0])),
        ]
    )
    test = pd.DataFrame({"asset": ["A", "A", "B"], "horizon": [1, 4, 1]})
    quantiles = random_walk(train, test)
    np.testing.assert_allclose(quantiles[0], np.sqrt(5.0) * NORMAL_QUANTILES)
    np.testing.assert_allclose(quantiles[1], 2.0 * NORMAL_QUANTILES)
    assert np.isnan(quantiles[2]).all()


def test_climatology_gives_each_group_its_own_empirical_quantiles() -> None:
    times = pd.date_range("2024-12-01", periods=101, freq="1h", tz="UTC")
    labels = np.linspace(-1, 1, 101)
    train = pd.concat([_rows(times, "A", 1, labels), _rows(times, "A", 24, 10 * labels)])
    test = pd.DataFrame({"asset": ["A", "A", "Z"], "horizon": [24, 1, 24]})
    quantiles = climatology(train, test)
    levels = np.asarray(QUANTILE_LEVELS)
    np.testing.assert_allclose(quantiles[1], 2 * levels - 1, atol=1e-12)
    np.testing.assert_allclose(quantiles[0], 10 * (2 * levels - 1), atol=1e-12)
    assert np.isnan(quantiles[2]).all()


def test_ewma_normal_is_the_standard_normal_for_every_row() -> None:
    quantiles = ewma_normal(pd.DataFrame({"asset": ["A"] * 3}))
    assert quantiles.shape == (3, 9)
    np.testing.assert_allclose(quantiles, np.tile(NORMAL_QUANTILES, (3, 1)))


def test_lag_features_are_vol_scaled_returns_over_own_bars() -> None:
    # bars close at 01:00, 02:00 and 05:00 (closed in between): one own bar back from 05:30
    # is 02:00, three hours earlier, and lag 2 reaches the first close
    close_times = pd.DatetimeIndex(
        ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 05:00"], tz="UTC"
    )
    closes = {"A": pd.Series([100.0, 110.0, 121.0], index=close_times)}
    frame = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(
                ["2025-01-01 02:30", "2025-01-01 04:00", "2025-01-01 05:30"], tz="UTC"
            ),
            "asset": "A",
            "sigma": [0.1, 0.2, 0.5],
        }
    )
    features = lag_features(frame, closes, (1, 2))
    np.testing.assert_allclose(features[:2, 0], [np.log(1.1) / 0.1, np.log(1.1) / 0.2])
    assert np.isnan(features[:2, 1]).all()
    np.testing.assert_allclose(features[2], [np.log(1.1) / 0.5, np.log(1.21) / (0.5 * np.sqrt(2))])


def test_linear_recovers_an_exact_lag_relation_and_its_residual_quantiles_are_zero() -> None:
    closes = {"A": _closes(600, 0)}
    times = pd.date_range("2024-12-03", periods=400, freq="1h", tz="UTC")
    train = _rows(times, "A", 4, np.zeros(400))
    features = lag_features(train, closes, (1, 2))
    train["label"] = 0.5 * features[:, 0] - 0.2 * features[:, 1]
    test = _rows(times[-5:] + pd.Timedelta(hours=2), "A", 4, np.zeros(5))
    expected = lag_features(test, closes, (1, 2)) @ np.array([0.5, -0.2])
    quantiles = linear(train, test, closes, lags=(1, 2), ridge=1e-9)
    np.testing.assert_allclose(quantiles, np.repeat(expected[:, None], 9, axis=1), atol=1e-6)


def test_linear_leaves_groups_without_training_rows_empty() -> None:
    closes = {"A": _closes(600, 0), "B": _closes(600, 1)}
    times = pd.date_range("2024-12-03", periods=50, freq="1h", tz="UTC")
    train = _rows(times, "A", 1, np.random.default_rng(0).normal(size=50))
    test = pd.concat([_rows(times[:2], "A", 1, np.zeros(2)), _rows(times[:2], "B", 1, np.zeros(2))])
    quantiles = linear(train, test.reset_index(drop=True), closes, lags=(1,), ridge=1.0)
    assert np.isfinite(quantiles[:2]).all()
    assert np.isnan(quantiles[2:]).all()


def test_garch_normal_on_iid_returns_is_close_to_the_scaled_standard_normal() -> None:
    closes = {"A": _closes(24 * 70, 3)}
    times = pd.date_range(START, periods=48, freq="1h", tz="UTC")
    test = pd.concat([_rows(times, "A", horizon, np.zeros(48)) for horizon in (1, 24)])
    quantiles, not_fitted = garch_normal(
        test.reset_index(drop=True), closes, start=START, train_days=30
    )
    assert not_fitted == {}
    assert quantiles.shape == (96, 9)
    np.testing.assert_allclose(quantiles[:, 4], 0.0, atol=1e-12)
    tails = [0, 1, 2, 3, 5, 6, 7, 8]
    np.testing.assert_allclose(quantiles[:, tails] / NORMAL_QUANTILES[tails], 1.0, atol=0.2)


def _garch_closes(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    variance, returns = 1e-4, np.zeros(n)
    for step in range(n):
        returns[step] = np.sqrt(variance) * rng.standard_normal()
        variance = 1e-5 + 0.15 * returns[step] ** 2 + 0.75 * variance
    close_times = pd.date_range("2024-12-01 01:00", periods=n, freq="1h", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=close_times)


def test_a_garch_fit_that_does_not_converge_is_not_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = baselines_module.arch_model

    def one_iteration(*args, **kwargs):
        # the same model, but its fit stops after one optimiser iteration
        model = real(*args, **kwargs)
        fit = model.fit
        model.fit = lambda **options: fit(options={"maxiter": 1}, **options)
        return model

    monkeypatch.setattr(baselines_module, "arch_model", one_iteration)
    closes = {"BTCUSDT": _garch_closes(24 * 70, 5)}
    times = pd.date_range(START, periods=4, freq="1h", tz="UTC")
    test = _rows(times, "BTCUSDT", 1, np.zeros(4))
    quantiles, not_fitted = garch_normal(test, closes, start=START, train_days=30)
    assert list(not_fitted) == ["BTCUSDT"]
    assert not_fitted["BTCUSDT"] != 0
    assert np.isnan(quantiles).all()


def test_garch_normal_never_uses_returns_after_the_decision() -> None:
    closes = _garch_closes(24 * 70, 4)
    cut = START + pd.Timedelta(hours=30)
    later = closes.index > cut
    changed = closes.copy()
    changed[later] = changed[later] * np.exp(np.random.default_rng(9).normal(0, 0.05, later.sum()))
    times = pd.date_range(START, periods=60, freq="1h", tz="UTC")
    test = pd.concat([_rows(times, "A", horizon, np.zeros(60)) for horizon in (1, 4)])
    test = test.reset_index(drop=True)
    first, _ = garch_normal(test, {"A": closes}, start=START, train_days=30)
    second, _ = garch_normal(test, {"A": changed}, start=START, train_days=30)
    known = (test["time"] <= cut).to_numpy()
    np.testing.assert_allclose(first[known], second[known], rtol=1e-10)
    assert not np.allclose(first[~known], second[~known])


@settings(max_examples=50, deadline=None)
@given(
    gaps=st.lists(st.integers(1, 120), min_size=1, max_size=40),
    offsets=st.lists(st.integers(0, 3000), min_size=1, max_size=20),
    lookback=st.integers(1, 10),
)
def test_context_ends_point_at_the_last_close_by_the_decision(gaps, offsets, lookback) -> None:
    close_times = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(np.cumsum(gaps), "h")
    closes = pd.Series(np.arange(len(gaps), dtype=float), index=close_times)
    times = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(sorted(offsets), "h")
    ends = context_ends(closes, pd.DatetimeIndex(times), lookback=lookback)
    for time, end in zip(times, ends, strict=True):
        closed = close_times[close_times <= time]
        assert end == (len(closed) - 1 if len(closed) >= lookback else -1)


def test_context_ends_of_a_short_series_are_all_invalid() -> None:
    closes = _closes(3, 0)
    times = pd.DatetimeIndex([closes.index[-1]])
    assert context_ends(closes, times, lookback=4).tolist() == [-1]
