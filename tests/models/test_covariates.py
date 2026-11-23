"""The past-covariate paths of the model wrappers, on fakes (no weights are loaded)."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models import QUANTILE_LEVELS, Forecast, Forecaster
from src.models import _new_models as new_models

N_SERIES, LOOKBACK, HORIZON = 3, 8, 2


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    context = 100 + rng.standard_normal((N_SERIES, LOOKBACK))
    context[:, -3:] = np.nan  # a closed market
    covariates = rng.standard_normal((N_SERIES, 1, LOOKBACK))
    return context, covariates


def _quantiles(n: int) -> np.ndarray:
    # increasing in the level, different per series: (n, horizon, 9)
    levels = np.asarray(QUANTILE_LEVELS)
    return np.arange(n)[:, None, None] + levels[None, None, :] + np.zeros((n, HORIZON, 1))


def test_a_model_without_covariates_refuses_them():
    class Plain(Forecaster):
        def predict(self, context: np.ndarray, horizon: int) -> Forecast:
            raise AssertionError("not called")

    context, covariates = _inputs()
    with pytest.raises(NotImplementedError, match="takes no covariates"):
        Plain("plain", device="cpu").predict_with_covariates(context, covariates, HORIZON)


# --- Chronos-2 ----------------------------------------------------------------------------


def test_chronos2_gets_each_series_with_its_past_covariates_as_one_group():
    chronos = pytest.importorskip("src.models._chronos")

    class FakeChronos2(chronos.Chronos2Pipeline):
        def __init__(self) -> None:
            self.inputs: list[dict[str, object]] = []

        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            self.inputs = list(inputs)
            values = torch.as_tensor(_quantiles(len(self.inputs)), dtype=torch.float32)
            return [q[None] for q in values], [q[None, :, 4] for q in values]

    model = object.__new__(chronos.ChronosForecaster)
    model.name, model.pipeline = "amazon/chronos-2", FakeChronos2()
    context, covariates = _inputs()
    forecast = model.predict_with_covariates(context, covariates, HORIZON)
    assert len(model.pipeline.inputs) == N_SERIES
    first = model.pipeline.inputs[0]
    np.testing.assert_array_equal(first["target"], context[0].astype(np.float32))
    np.testing.assert_array_equal(
        first["past_covariates"]["covariate_0"], covariates[0, 0].astype(np.float32)
    )
    assert forecast.quantiles.shape == (N_SERIES, HORIZON, 9)
    np.testing.assert_allclose(forecast.quantiles, _quantiles(N_SERIES), rtol=1e-6)


def test_chronos_bolt_takes_no_covariates():
    chronos = pytest.importorskip("src.models._chronos")
    model = object.__new__(chronos.ChronosForecaster)
    model.name, model.pipeline = "amazon/chronos-bolt-small", object()
    context, covariates = _inputs()
    with pytest.raises(NotImplementedError, match="only Chronos-2"):
        model.predict_with_covariates(context, covariates, HORIZON)


# --- TimesFM 3.0, TiRex-2 and t0 ----------------------------------------------------------


class FakeTimesFM3:
    def __init__(self) -> None:
        self.covariates: list[np.ndarray] | None = None

    def predict_batch(self, series, *, horizon, past_only_covariates, **options):
        self.covariates = past_only_covariates
        return [SimpleNamespace(quantiles=q) for q in _quantiles(len(series))]


def test_timesfm3_passes_past_only_covariates():
    model = object.__new__(new_models.TimesFM3Forecaster)
    model.model = FakeTimesFM3()
    context, covariates = _inputs()
    forecast = model.predict_with_covariates(context, covariates, HORIZON)
    assert len(model.model.covariates) == N_SERIES
    np.testing.assert_array_equal(model.model.covariates[1], covariates[1].astype(np.float32))
    np.testing.assert_allclose(forecast.quantiles, _quantiles(N_SERIES))
    model.predict(context, HORIZON)
    assert model.model.covariates is None


@dataclass
class FakeSeries:
    target: torch.Tensor
    past_covariates: torch.Tensor | None
    future_covariates: torch.Tensor | None


class FakeTiRex2:
    def __init__(self) -> None:
        self.series: list[FakeSeries] = []

    def forecast(self, series, *, prediction_length, output_type, batch_size):
        self.series = list(series)
        # each output is (n_targets = 1, 9 quantiles, horizon)
        return [q.T[None] for q in _quantiles(len(self.series))]


def test_tirex2_passes_past_covariates_next_to_the_target():
    model = object.__new__(new_models.TiRex2Forecaster)
    model.model, model.series_type, model.batch_size = FakeTiRex2(), FakeSeries, 4
    context, covariates = _inputs()
    forecast = model.predict_with_covariates(context, covariates, HORIZON)
    first = model.model.series[0]
    assert first.target.shape == (1, LOOKBACK)
    assert first.past_covariates is not None
    assert first.past_covariates.shape == (1, LOOKBACK)
    assert first.future_covariates is None
    np.testing.assert_allclose(forecast.quantiles, _quantiles(N_SERIES))
    model.predict(context, HORIZON)
    assert model.model.series[0].past_covariates is None


class FakeT0:
    def __init__(self) -> None:
        self.roles: list[list[int]] = []

    def predict(self, inputs, *, horizon, quantile_levels):
        if isinstance(inputs, torch.Tensor):
            return SimpleNamespace(quantiles=torch.as_tensor(_quantiles(len(inputs))))
        # a t0 TimeSeries: one forecast per target row (the covariates are HISTORICAL)
        roles = inputs.variate_type[:, 0].tolist()
        self.roles.append(roles)
        return SimpleNamespace(quantiles=torch.as_tensor(_quantiles(roles.count(0))))


def test_t0_reads_covariates_as_past_only_variates_and_keeps_the_targets_forecast():
    pytest.importorskip("t0")
    model = object.__new__(new_models.T0Forecaster)
    model.model, model.batch_size, model.device = FakeT0(), 2, "cpu"
    context, covariates = _inputs()
    forecast = model.predict_with_covariates(context, covariates, HORIZON)
    assert model.model.roles == [[0, 1, 0, 1], [0, 1]]
    expected = np.concatenate([_quantiles(2), _quantiles(1)])
    np.testing.assert_allclose(forecast.quantiles, expected)
    np.testing.assert_allclose(forecast.point, expected[:, :, 4])
