"""Tests of 11-closures-length (``lib/_length.py``): the context's length at the same encoding."""

from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SETTINGS = {"alpha": 0.05, "min_closure_days": 20, "bootstrap_resamples": 200, "bootstrap_seed": 0}


def _losses(length: ModuleType, *, days: int, gain: float) -> pd.DataFrame:
    # one model, two assets, one decision per asset and closure day; the benchmark input's
    # CRPS is `gain` lower than trading hours only's, plus noise; the baseline is left out
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for day in range(days):
        closure_day = pd.Timestamp("2025-11-03", tz="UTC") + pd.Timedelta(days=day)
        for asset in ("EURUSD", "AAPL"):
            base = 1.0 + 0.1 * rng.standard_normal()
            longer = base - gain + 0.02 * rng.standard_normal()
            for arm, crps in ((length.TRADING_HOURS, base), (length.BENCHMARK_INPUT, longer)):
                rows.append(
                    {
                        "model": "m",
                        "group": length.CLOSED,
                        "representation": arm,
                        "is_scored": True,
                        "closure_day": closure_day,
                        "time": closure_day,
                        "asset": asset,
                        "market": "forex" if asset == "EURUSD" else "stock",
                        "crps": crps,
                    }
                )
    return pd.DataFrame(rows)


def test_a_lower_crps_with_the_longer_context_is_significant(length: ModuleType):
    settings = length.read_length_settings(SETTINGS)
    tests = length.length_tests(_losses(length, days=300, gain=0.05), {"m": "on time"}, settings)
    row = tests.iloc[0]
    assert row["status"] == "scored"
    assert row["skill"] == pytest.approx(0.05, rel=0.2)
    assert row["skill_low"] > 0
    assert row["adjusted_p"] < 0.05


def test_too_few_closure_days_are_not_tested(length: ModuleType):
    settings = length.read_length_settings(SETTINGS)
    tests = length.length_tests(_losses(length, days=10, gain=0.05), {"m": "on time"}, settings)
    assert tests.iloc[0]["status"].startswith("not tested")
    assert np.isnan(tests.iloc[0]["adjusted_p"])


def test_a_model_without_both_inputs_is_left_out(length: ModuleType):
    settings = length.read_length_settings(SETTINGS)
    losses = _losses(length, days=30, gain=0.05)
    losses = losses[losses["representation"] == length.TRADING_HOURS]
    assert length.length_tests(losses, {"m": "on time"}, settings).empty


def test_the_skill_by_market_is_one_minus_the_ratio(length: ModuleType):
    losses = _losses(length, days=30, gain=0.1)
    by_market = length.length_by_market(losses).set_index("market")
    assert set(by_market.index) == {"forex", "stock"}
    for market, row in by_market.iterrows():
        part = losses[losses["market"] == market]
        trading = part.loc[part["representation"] == length.TRADING_HOURS, "crps"].mean()
        benchmark = part.loc[part["representation"] == length.BENCHMARK_INPUT, "crps"].mean()
        assert row["skill"] == pytest.approx(1 - benchmark / trading)
    assert by_market["decisions"].tolist() == [30, 30]


def test_holm_runs_within_each_family(length: ModuleType):
    settings = length.read_length_settings(SETTINGS)
    first = _losses(length, days=300, gain=0.05)
    second = first.assign(model="n")
    tests = length.length_tests(
        pd.concat([first, second]), {"m": "on time", "n": "late"}, settings
    ).set_index("model")
    # alone in its family, each model's adjusted p is its own p
    np.testing.assert_allclose(tests["adjusted_p"], tests["p_value"])


def test_inputs_on_other_closure_days_raise(length: ModuleType):
    settings = length.read_length_settings(SETTINGS)
    losses = _losses(length, days=30, gain=0.05)
    first_day = losses["closure_day"].min()
    is_dropped = (losses["representation"] == length.BENCHMARK_INPUT) & (
        losses["closure_day"] == first_day
    )
    with pytest.raises(ValueError, match="other closure days"):
        length.length_tests(losses[~is_dropped], {"m": "on time"}, settings)


def test_a_day_weighs_the_same_whatever_its_decisions(length: ModuleType):
    frame = pd.DataFrame(
        {
            "closure_day": ["a", "a", "a", "b"],
            "time": [1, 1, 2, 3],
            "crps": [1.0, 3.0, 4.0, 10.0],
        }
    )
    series = length.closure_day_series(frame)
    np.testing.assert_allclose(series.to_numpy(), [3.0, 10.0])
