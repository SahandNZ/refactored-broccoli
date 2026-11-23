"""Chronos runs in bfloat16: its quantiles must not be rounded to the bfloat16 grid in price units.

At a price of 100,000 the bfloat16 grid is 512 apart, far wider than a one-hour quantile spread
of Bitcoin, so a cast to bfloat16 after the model has scaled its output back to prices would
destroy the forecast. The wrapper must hand back the pipeline's float32 quantiles unchanged.
"""

import numpy as np
import pytest
import torch

chronos = pytest.importorskip("src.models._chronos")

LEVELS = 9


class Float32Pipeline:
    """Stands in for a Chronos-Bolt pipeline: float32 quantiles off the bfloat16 grid."""

    def predict_quantiles(self, inputs, *, prediction_length, quantile_levels):
        last = inputs[:, -1].double()
        offsets = torch.linspace(-30.0, 30.0, len(quantile_levels), dtype=torch.float64)
        quantiles = last[:, None, None] + offsets + 0.123 * torch.arange(prediction_length)[:, None]
        return quantiles.float(), quantiles[..., len(quantile_levels) // 2].float()


def _wrapper() -> "chronos.ChronosForecaster":
    model = object.__new__(chronos.ChronosForecaster)
    model.pipeline = Float32Pipeline()
    return model


def test_quantiles_keep_float32_precision_at_large_prices() -> None:
    context = np.full((2, 16), 100_000.0)
    forecast = _wrapper().predict(context, 4)
    assert forecast.quantiles.dtype == np.float32
    expected = Float32Pipeline().predict_quantiles(
        torch.as_tensor(context, dtype=torch.float32),
        prediction_length=4,
        quantile_levels=list(range(LEVELS)),
    )[0]
    np.testing.assert_array_equal(forecast.quantiles, expected.numpy())
    rounded = torch.as_tensor(forecast.quantiles).bfloat16().float().numpy()
    # the bfloat16 grid would collapse the nine quantiles of a step into at most two values
    assert len(np.unique(rounded[0, 0])) <= 2
    assert len(np.unique(forecast.quantiles[0, 0])) == LEVELS


@pytest.mark.cuda
def test_bfloat16_weights_give_quantiles_close_to_float32_ones() -> None:
    # real Chronos-Bolt small weights (from .hf): bfloat16 against float32 on random walks at
    # a Bitcoin-like price; the quantiles of the log return differ by a small part of their
    # 10-90 spread (measured on 2026-11-17: median 0.6 %, largest 5 % at one step)
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    rng = np.random.default_rng(0)
    context = 100_000 * np.exp(np.cumsum(0.005 * rng.standard_normal((64, 512)), axis=1))
    returns = {}
    for dtype in (torch.bfloat16, torch.float32):
        pipeline = chronos.BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cuda", torch_dtype=dtype
        )
        quantiles, _ = pipeline.predict_quantiles(
            torch.as_tensor(context, dtype=torch.float32),
            prediction_length=24,
            quantile_levels=list(chronos.QUANTILE_LEVELS),
        )
        returns[dtype] = np.log(quantiles.float().cpu().numpy() / context[:, -1, None, None])
    spread = returns[torch.float32][..., -1] - returns[torch.float32][..., 0]
    error = np.abs(returns[torch.bfloat16] - returns[torch.float32]) / spread[..., None]
    assert np.median(error) < 0.02
    assert error.max() < 0.1
