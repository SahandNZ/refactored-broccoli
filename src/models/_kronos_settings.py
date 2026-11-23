from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class KronosSettings:
    """How Kronos is read and sampled (`src.models._kronos.KronosForecaster`).

    Kept apart from the wrapper, which imports the vendored Kronos code, so a configuration
    can build it without loading the model.

    Attributes
    ----------
    tokenizer : str
        The Hugging Face name of the tokenizer that matches the model.
    n_samples : int
        Sampled paths per series; the quantiles come from them.
    temperature : float
        The sampling temperature (``T`` of `KronosPredictor.predict_batch`).
    top_p : float
        The nucleus-sampling threshold.
    step : pd.Timedelta
        The bar length of the candles, e.g. one hour for 1h bars.
    """

    tokenizer: str
    n_samples: int
    temperature: float
    top_p: float
    step: pd.Timedelta
