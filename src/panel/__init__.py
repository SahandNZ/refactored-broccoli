"""A panel model (later work): multi-asset, multi-timeframe inputs, labels, encoders, training."""

from src.panel._dataset import (
    PanelData,
    PanelSettings,
    forward_returns,
    hourly_closes,
    is_us_stock,
    load_panel_bars,
)
from src.panel._encoders import LSTMEncoder, RevIN, TransformerEncoder
from src.panel._experiment import (
    TimeframePlan,
    climatology,
    decision_frequency,
    evaluate_forecasts,
    run_fold,
    split_fold,
)
from src.panel._inputs import FEATURES, PanelWindows, bar_features
from src.panel._labels import ewma_volatility, vol_scaled_returns
from src.panel._losses import portfolio_return_loss
from src.panel._model import Mixing, PanelConfig, PanelModel
from src.panel._training import quantile_loss, train

__all__ = [
    "FEATURES",
    "LSTMEncoder",
    "Mixing",
    "PanelConfig",
    "PanelData",
    "PanelModel",
    "PanelSettings",
    "PanelWindows",
    "RevIN",
    "TimeframePlan",
    "TransformerEncoder",
    "bar_features",
    "climatology",
    "decision_frequency",
    "evaluate_forecasts",
    "ewma_volatility",
    "forward_returns",
    "hourly_closes",
    "is_us_stock",
    "load_panel_bars",
    "portfolio_return_loss",
    "quantile_loss",
    "run_fold",
    "split_fold",
    "train",
    "vol_scaled_returns",
]
