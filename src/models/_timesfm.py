import numpy as np
import timesfm
import torch

from src.models._forecaster import Forecast, Forecaster


class TF32Precision:
    """Run float32 matrix products in TF32 ("high") inside the block.

    TimesFM's reference code sets it; the setting is process-wide, so the previous precision is
    restored afterwards and a model built later in the same process keeps full float32.
    """

    def __enter__(self) -> None:
        self.previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")

    def __exit__(self, *exc: object) -> None:
        torch.set_float32_matmul_precision(self.previous)


class TimesFMForecaster(Forecaster):
    """TimesFM 2.5.

    Das et al. 2024, "A decoder-only foundation model for time-series forecasting".

    Parameters
    ----------
    name : str, optional
        Hugging Face repository of the weights.
    max_context, max_horizon : int, optional
        The longest context and horizon the compiled model accepts.
    device : str, optional
        Torch device (TimesFM picks CUDA itself when it is available).
    """

    def __init__(
        self,
        name: str = "google/timesfm-2.5-200m-pytorch",
        *,
        max_context: int = 1024,
        max_horizon: int = 128,
        device: str = "cuda",
    ) -> None:
        super().__init__(name, device=device)
        with TF32Precision():
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(name)
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=True,
                    per_core_batch_size=64,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                )
            )

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        with TF32Precision():
            point, quantiles = self.model.forecast(horizon=horizon, inputs=list(context))
        # the quantile head returns the mean first, then the deciles 0.1 .. 0.9
        return Forecast(point=np.asarray(point), quantiles=np.asarray(quantiles)[..., 1:])
