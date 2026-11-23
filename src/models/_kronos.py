import sys

import numpy as np
import pandas as pd

from src import LAB_DIR
from src.models._forecaster import QUANTILE_LEVELS, Forecast, Forecaster
from src.models._kronos_settings import KronosSettings

sys.path.insert(0, str(LAB_DIR / "vendor" / "Kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer

_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_MAX_CONTEXT = 512
"""The longest context Kronos-small and Kronos-base read."""


class KronosForecaster(Forecaster):
    """Kronos, a foundation model of K-line (candle) sequences.

    Shi et al. 2025, "Kronos: A Foundation Model for the Language of Financial Markets".

    Kronos reads whole candles and their timestamps and samples future candles. `predict` takes
    closes only (flat candles, zero volume, made-up timestamps ``settings.step`` apart) to
    share the interface; `predict_candles` gives it real candles. Quantiles come from
    ``settings.n_samples`` paths, sampled with ``settings.temperature`` and
    ``settings.top_p``. The future timestamps continue the last candle every
    ``settings.step``, the bar length, not the gap of the context's first two candles (which
    spans a closure when the context starts across one): the real times of the next bars of a
    market that closes are not known at the decision, so step ``k`` is read as the ``k``-th
    next bar.

    Parameters
    ----------
    name : str
        ``NeoQuasar/Kronos-{mini,small,base}``.
    settings : KronosSettings
        Tokenizer, sampling and bar length, from the study's configuration.
    device : str
        Torch device.
    batch_size : int, optional
        Rows (series x paths) per forward pass, a memory limit: larger batches overflow the
        12 GB GPU.
    """

    def __init__(
        self, name: str, *, settings: KronosSettings, device: str, batch_size: int = 128
    ) -> None:
        super().__init__(name, device=device)
        self.settings = settings
        self.batch_size = batch_size
        self.predictor = KronosPredictor(
            Kronos.from_pretrained(name),
            KronosTokenizer.from_pretrained(settings.tokenizer),
            device=device,
            max_context=_MAX_CONTEXT,
        )

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """See `Forecaster.predict`."""
        times = pd.date_range(
            "2026-01-01", periods=context.shape[1], freq=self.settings.step, tz="UTC"
        )
        candles = [
            pd.DataFrame(
                {"open": row, "high": row, "low": row, "close": row, "volume": 0.0, "amount": 0.0},
                index=times,
            )
            for row in context
        ]
        return self.predict_candles(candles, horizon)

    def predict_candles(self, candles: list[pd.DataFrame], horizon: int) -> Forecast:
        """Forecast the close from real candles.

        Parameters
        ----------
        candles : list of pd.DataFrame
            One frame per series with columns open, high, low, close, volume, amount and the
            bars' UTC times (a market that closes has gaps); all of the same length.
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        Forecast
            Close forecasts, the mean and the quantiles of the sampled paths.
        """
        closes = self.sample_closes(candles, horizon)
        quantiles = np.moveaxis(np.quantile(closes, QUANTILE_LEVELS, axis=1), 0, -1)
        return Forecast(point=closes.mean(axis=1), quantiles=quantiles)

    def sample_closes(self, candles: list[pd.DataFrame], horizon: int) -> np.ndarray:
        """Sample ``settings.n_samples`` future close paths per series from real candles.

        Parameters
        ----------
        candles : list of pd.DataFrame
            As in `predict_candles`.
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray
            Sampled closes, shape ``(n_series, n_samples, horizon)``.
        """
        step, n_samples = self.settings.step, self.settings.n_samples
        frames, past, future = [], [], []
        for frame in candles:
            future_times = pd.date_range(frame.index[-1] + step, periods=horizon, freq=step)
            # one batch row per sampled path
            for _ in range(n_samples):
                frames.append(frame[_PRICE_COLUMNS].reset_index(drop=True))
                past.append(pd.Series(frame.index))
                future.append(pd.Series(future_times))
        paths = []
        for first in range(0, len(frames), self.batch_size):
            last = first + self.batch_size
            paths += self.predictor.predict_batch(
                frames[first:last],
                past[first:last],
                future[first:last],
                pred_len=horizon,
                T=self.settings.temperature,
                top_p=self.settings.top_p,
                sample_count=1,
                verbose=False,
            )
        closes = np.stack([path["close"].to_numpy() for path in paths])
        return closes.reshape(len(candles), n_samples, horizon)
