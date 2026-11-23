import numpy as np
import pandas as pd
import pytest

from src.models import KronosSettings

kronos = pytest.importorskip("src.models._kronos")  # needs the pinned Kronos code in vendor/


class RecordingPredictor:
    """Stands in for KronosPredictor: records the future timestamps, returns flat paths."""

    def __init__(self) -> None:
        self.future: list[pd.Series] = []

    def predict_batch(self, frames, past, future, *, pred_len, **options):
        self.future += future
        self.sampling = (options["T"], options["top_p"])
        return [
            pd.DataFrame({"close": np.full(pred_len, frame["close"].iloc[-1])}) for frame in frames
        ]


def _forecaster(step: pd.Timedelta) -> "kronos.KronosForecaster":
    # the wrapper without loading weights: only the attributes predict_candles reads
    model = object.__new__(kronos.KronosForecaster)
    model.settings = KronosSettings("tokenizer", 2, 0.7, 0.8, step)
    model.batch_size = 8
    model.predictor = RecordingPredictor()
    return model


def test_future_timestamps_follow_the_bar_length_when_the_context_starts_across_a_closure():
    # a stock context: Friday 20:00 then the next bars on Monday from 14:00
    times = pd.DatetimeIndex(
        ["2025-03-07 20:00", "2025-03-10 14:00", "2025-03-10 15:00", "2025-03-10 16:00"], tz="UTC"
    )
    candle = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "amount": 1.0},
        index=times,
    )
    model = _forecaster(pd.Timedelta(hours=1))
    forecast = model.predict_candles([candle], 3)
    expected = pd.DatetimeIndex(
        ["2025-03-10 17:00", "2025-03-10 18:00", "2025-03-10 19:00"], tz="UTC"
    )
    assert len(model.predictor.future) == 2
    for future in model.predictor.future:
        assert pd.DatetimeIndex(future).equals(expected)
    assert forecast.quantiles.shape == (1, 3, 9)
    assert model.predictor.sampling == (0.7, 0.8)


def test_closes_only_get_made_up_timestamps_one_step_apart():
    model = _forecaster(pd.Timedelta(minutes=15))
    model.predict(np.ones((1, 4)), 2)
    future = pd.DatetimeIndex(model.predictor.future[0])
    assert (future[1] - future[0]) == pd.Timedelta(minutes=15)
