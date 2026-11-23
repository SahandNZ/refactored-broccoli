import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from src.models._forecaster import QUANTILE_LEVELS, Forecast, Forecaster, check_covariates


class ChronosForecaster(Forecaster):
    """Chronos-Bolt or Chronos-2.

    Ansari et al. 2024, "Chronos: Learning the Language of Time Series"; Amazon 2025, "Chronos-2".

    Parameters
    ----------
    name : str, optional
        ``amazon/chronos-bolt-{tiny,mini,small,base}`` or ``amazon/chronos-2``.
    device : str, optional
        Torch device.
    """

    def __init__(self, name: str = "amazon/chronos-bolt-small", *, device: str = "cuda") -> None:
        super().__init__(name, device=device)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            name, device_map=device, torch_dtype=torch.bfloat16
        )

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        inputs = torch.as_tensor(context, dtype=torch.float32)
        if isinstance(self.pipeline, Chronos2Pipeline):
            inputs = inputs.unsqueeze(1)  # (n_series, 1 variate, lookback)
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=list(QUANTILE_LEVELS)
        )
        # Chronos-2 returns one (variates, horizon, ...) tensor per series
        if isinstance(quantiles, list):
            quantiles = torch.stack([q[0] for q in quantiles])
            mean = torch.stack([m[0] for m in mean])
        return Forecast(point=mean.float().cpu().numpy(), quantiles=quantiles.float().cpu().numpy())

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        """See `Forecaster.predict_with_covariates`.

        Chronos-2 only: each series and its covariates form one group, so the model attends
        across them (group attention); Chronos-Bolt takes no covariates and raises.
        """
        if not isinstance(self.pipeline, Chronos2Pipeline):
            raise NotImplementedError(f"{self.name} takes no covariates (only Chronos-2 does)")
        check_covariates(context, covariates)
        inputs = [
            {
                "target": target.astype(np.float32),
                "past_covariates": {
                    f"covariate_{index}": row.astype(np.float32) for index, row in enumerate(rows)
                },
            }
            for target, rows in zip(context, covariates, strict=True)
        ]
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=list(QUANTILE_LEVELS)
        )
        # one (1 target, horizon, ...) tensor per series
        return Forecast(
            point=torch.stack([m[0] for m in mean]).float().cpu().numpy(),
            quantiles=torch.stack([q[0] for q in quantiles]).float().cpu().numpy(),
        )
