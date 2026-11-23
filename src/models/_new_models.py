"""Wrappers of the 2026 foundation models: TimesFM 3.0, TiRex, TiRex-2, Toto 2.0 and t0.

Every wrapper returns the 9 deciles 0.1 .. 0.9 (`QUANTILE_LEVELS`) and the median as the point
forecast, so all models are scored on the same quantiles.
"""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.models._forecaster import QUANTILE_LEVELS, Forecast, Forecaster, check_covariates

if TYPE_CHECKING:
    from t0 import TimeSeries


def _batches(context: np.ndarray, size: int):
    for start in range(0, len(context), size):
        yield context[start : start + size]


class TimesFM3Forecaster(Forecaster):
    """TimesFM 3.0 (Google, weights 2026-08; TimesFM non-commercial weight license).

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Torch device.
    """

    def __init__(
        self,
        name: str = "google/timesfm-3.0-pytorch",
        *,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        super().__init__(name, device=device)
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        config = ModelConfig(checkpoint_path=name, per_core_batch_size=batch_size, device=device)
        self.model = TimesFM3Evaluator(config)

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        return self._forecast(context, None, horizon)

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        """See `Forecaster.predict_with_covariates`: TimesFM 3.0's past-only covariates."""
        check_covariates(context, covariates)
        return self._forecast(context, covariates, horizon)

    def _forecast(
        self, context: np.ndarray, covariates: np.ndarray | None, horizon: int
    ) -> Forecast:
        series = [row.astype(np.float32) for row in context]
        extra = None if covariates is None else [rows.astype(np.float32) for rows in covariates]
        outputs = list(
            self.model.predict_batch(
                series,
                horizon=horizon,
                past_only_covariates=extra,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        quantiles = np.stack([np.asarray(out.quantiles) for out in outputs])
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)


class TiRexForecaster(Forecaster):
    """TiRex 1.1 (NX-AI, 2025; xLSTM, NX-AI community license).

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Torch device.
    """

    def __init__(
        self, name: str = "NX-AI/TiRex", *, batch_size: int = 256, device: str = "cuda"
    ) -> None:
        super().__init__(name, device=device)
        from tirex import load_model

        self.model = load_model(name, device=device)
        self.batch_size = batch_size

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        parts = []
        for batch in _batches(context, self.batch_size):
            quantiles, _ = self.model.forecast(
                context=torch.as_tensor(batch, dtype=torch.float32), prediction_length=horizon
            )
            parts.append(np.asarray(quantiles.float().cpu()))
        quantiles = np.concatenate(parts)  # (n, horizon, 9)
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)


class TiRex2Forecaster(Forecaster):
    """TiRex-2 (NX-AI, 2026-06; Apache-2.0).

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Torch device.
    """

    def __init__(
        self, name: str = "NX-AI/TiRex-2", *, batch_size: int = 256, device: str = "cuda"
    ) -> None:
        super().__init__(name, device=device)
        from tirex2 import TimeseriesType, load_model

        self.model = load_model(name, device=device)
        self.series_type = TimeseriesType
        self.batch_size = batch_size

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        return self._forecast(context, [None] * len(context), horizon)

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        """See `Forecaster.predict_with_covariates`: TiRex-2's past covariates."""
        check_covariates(context, covariates)
        rows: list[torch.Tensor | None] = [
            torch.as_tensor(row, dtype=torch.float32) for row in covariates
        ]
        return self._forecast(context, rows, horizon)

    def _forecast(
        self, context: np.ndarray, covariates: list[torch.Tensor | None], horizon: int
    ) -> Forecast:
        series = [
            self.series_type(
                target=torch.as_tensor(row, dtype=torch.float32)[None, :],
                past_covariates=extra,
                future_covariates=None,
            )
            for row, extra in zip(context, covariates, strict=True)
        ]
        outputs = self.model.forecast(
            series, prediction_length=horizon, output_type="numpy", batch_size=self.batch_size
        )
        # each output is (n_targets = 1, 9 quantiles, horizon)
        quantiles = np.stack([np.asarray(out)[0].T for out in outputs])
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)


class Toto2Forecaster(Forecaster):
    """Toto 2.0 (Datadog, 2026-04; Apache-2.0), sizes 4m .. 2.5B.

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights, e.g. ``Datadog/Toto-2.0-313m``.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Torch device.
    """

    def __init__(
        self, name: str = "Datadog/Toto-2.0-22m", *, batch_size: int = 128, device: str = "cuda"
    ) -> None:
        super().__init__(name, device=device)
        from toto2 import Toto2Model

        self.model = Toto2Model.from_pretrained(name).to(device).eval()
        self.batch_size = batch_size

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        parts = []
        for batch in _batches(context, self.batch_size):
            target = torch.as_tensor(batch, dtype=torch.float32, device=self.device)[:, None, :]
            inputs = {
                "target": target,
                "target_mask": torch.ones_like(target, dtype=torch.bool),
                "series_ids": torch.zeros(len(batch), 1, dtype=torch.long, device=self.device),
            }
            with torch.no_grad():
                quantiles = self.model.forecast(
                    inputs, horizon=horizon, decode_block_size=768, has_missing_values=False
                )
            # (9, batch, n_variates = 1, horizon) -> (batch, horizon, 9)
            parts.append(np.moveaxis(np.asarray(quantiles.float().cpu())[:, :, 0, :], 0, -1))
        quantiles = np.concatenate(parts)
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)


class T0Forecaster(Forecaster):
    """t0 (The Forecasting Company, gamma 2026-10; Apache-2.0).

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Torch device.
    """

    def __init__(
        self,
        name: str = "theforecastingcompany/t0-gamma",
        *,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__(name, device=device)
        from t0 import T0Forecaster as Model

        self.model = Model.from_pretrained(name).to(device).eval()
        self.batch_size = batch_size

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        quantiles = self._quantiles(context, horizon)  # (n, horizon, 9)
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        """See `Forecaster.predict_with_covariates`.

        t0's native past-only covariates: each covariate is a further variate of the same
        series with the role ``VariateType.HISTORICAL`` (`historical_series`), read over the
        context and never forecast; the forecast is the target's.
        """
        check_covariates(context, covariates)
        variates = np.concatenate([context[:, None, :], covariates], axis=1)
        parts: list[np.ndarray] = []
        for batch in _batches(variates, self.batch_size):
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            series = historical_series(tensor, horizon)
            with torch.no_grad():
                out = self.model.predict(
                    series, horizon=horizon, quantile_levels=list(QUANTILE_LEVELS)
                )
            parts.append(np.asarray(out.quantiles.float().cpu()))  # one row per target
        quantiles = np.concatenate(parts)  # (n, horizon, 9)
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)

    def _quantiles(self, inputs: np.ndarray, horizon: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for batch in _batches(inputs, self.batch_size):
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out = self.model.predict(
                    tensor, horizon=horizon, quantile_levels=list(QUANTILE_LEVELS)
                )
            parts.append(np.asarray(out.quantiles.float().cpu()))
        return np.concatenate(parts)


def historical_series(variates: torch.Tensor, horizon: int) -> "TimeSeries[torch.Tensor]":
    """Build t0's input for series with past-only covariates.

    Row 0 of each series is the target, the other rows are covariates with the role
    ``VariateType.HISTORICAL``: t0 reads them over the context and does not forecast them.

    Parameters
    ----------
    variates : torch.Tensor
        Shape ``(n_series, 1 + n_covariates, lookback)``; NaN where a value is missing.
    horizon : int
        The steps to forecast.

    Returns
    -------
    t0.TimeSeries
        The rows of every series in one group, the forecast region marked after the context.
    """
    from t0 import TimeSeries, VariateType

    series: TimeSeries[torch.Tensor] = TimeSeries.from_array(variates, horizon=horizon)
    roles: torch.Tensor = series.variate_type.clone()
    is_covariate = torch.arange(len(roles), device=variates.device) % variates.shape[1] != 0
    roles[is_covariate] = VariateType.HISTORICAL
    return dataclasses.replace(series, variate_type=roles)


class Moirai2Forecaster(Forecaster):
    """Moirai 2.0 (Salesforce, 2025-08; CC-BY-NC-4.0), run in its own environment.

    uni2ts pins torch<2.5, so the model runs in ``.venv-moirai`` through
    ``scripts/moirai_worker.py``; contexts and forecasts pass through temporary .npz files.

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    batch_size : int, optional
        Series per forward pass.
    device : str, optional
        Unused (the worker picks CUDA when available); kept for the common interface.
    """

    def __init__(
        self,
        name: str = "Salesforce/moirai-2.0-R-small",
        *,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__(name, device=device)
        from src import LAB_DIR

        self.python = LAB_DIR / ".venv-moirai" / "bin" / "python"
        self.worker = LAB_DIR / "scripts" / "moirai_worker.py"
        self.batch_size = batch_size

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "in.npz", Path(folder) / "out.npz"
            np.savez(source, context=context)
            environment = dict(os.environ)
            subprocess.run(  # noqa: S603 (fixed interpreter and script, no shell)
                [str(self.python), str(self.worker), str(source), str(target), str(horizon),
                 self.name, str(self.batch_size)],
                check=True, env=environment,
            )  # fmt: skip
            quantiles = np.load(target)["quantiles"]
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)
