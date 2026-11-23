from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
"""The quantile levels every forecaster returns."""


@dataclass(frozen=True)
class Forecast:
    """Point and quantile forecasts of several series.

    Attributes
    ----------
    point : np.ndarray
        Point forecast (the model's mean or median), shape ``(n_series, horizon)``.
    quantiles : np.ndarray
        Quantile forecasts, shape ``(n_series, horizon, len(levels))``.
    levels : tuple of float
        The quantile levels of the last axis of `quantiles`.
    """

    point: np.ndarray
    quantiles: np.ndarray
    levels: tuple[float, ...] = QUANTILE_LEVELS

    def checked_quantiles(self, n_series: int, horizon: int) -> np.ndarray:
        """Return `quantiles` after checking their shape ``(n_series, horizon, 9)``.

        Raises
        ------
        ValueError
            If the shape is different (a wrapper returned too few series or steps).
        """
        expected = (n_series, horizon, len(QUANTILE_LEVELS))
        if self.quantiles.shape != expected:
            raise ValueError(
                f"forecast quantiles must have shape {expected}, got {self.quantiles.shape}"
            )
        return self.quantiles


def check_covariates(context: np.ndarray, covariates: np.ndarray) -> None:
    """Check that past covariates line up with their contexts (`predict_with_covariates`).

    Raises
    ------
    ValueError
        If `covariates` is not ``(n_series, n_covariates, lookback)`` for a `context` of shape
        ``(n_series, lookback)``.
    """
    is_aligned = covariates.ndim == 3 and covariates.shape[0] == context.shape[0]
    if not is_aligned or covariates.shape[2] != context.shape[1]:
        raise ValueError(
            "covariates must have shape (n_series, n_covariates, lookback) = "
            f"({context.shape[0]}, k, {context.shape[1]}), got {covariates.shape}"
        )


class Forecaster(ABC):
    """A pretrained time-series foundation model used zero-shot.

    Parameters
    ----------
    name : str
        Hugging Face repository of the weights.
    device : str, optional
        Torch device.
    """

    def __init__(self, name: str, *, device: str = "cuda") -> None:
        self.name = name
        self.device = device

    @abstractmethod
    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """Forecast the next `horizon` values of each series.

        Parameters
        ----------
        context : np.ndarray
            Past values, shape ``(n_series, lookback)``, oldest first.
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        Forecast
            Arrays of shape ``(n_series, horizon)`` and ``(n_series, horizon, 9)``.
        """

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        """Forecast the next `horizon` values of each series from its past and past covariates.

        Only models that read covariates override this; the others raise.

        Parameters
        ----------
        context : np.ndarray
            Past values, shape ``(n_series, lookback)``, oldest first; NaN where a value is
            missing (a closed market).
        covariates : np.ndarray
            Past-only covariates on the same time steps, shape
            ``(n_series, n_covariates, lookback)``.
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        Forecast
            Arrays of shape ``(n_series, horizon)`` and ``(n_series, horizon, 9)``.

        Raises
        ------
        NotImplementedError
            If the model takes no covariates.
        """
        raise NotImplementedError(f"{type(self).__name__} ({self.name}) takes no covariates")
