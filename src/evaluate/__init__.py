"""Evaluation: folds, forecast metrics, trading strategies after costs and ensembles."""

from src.evaluate._ensemble import zscore_ensemble
from src.evaluate._folds import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS, Fold
from src.evaluate._metrics import hit_rate, mase, panel_rank_ic, quantile_crps, rank_ic
from src.evaluate._strategy import (
    BINANCE_USDM_BPS,
    cost_per_side_bps,
    portfolio_returns,
    sharpe_ratio,
    strategy_returns,
)

__all__ = [
    "BINANCE_USDM_BPS",
    "DEVELOPMENT_FOLDS",
    "LEAKAGE_FOLDS",
    "TEST_FOLDS",
    "Fold",
    "cost_per_side_bps",
    "hit_rate",
    "mase",
    "panel_rank_ic",
    "portfolio_returns",
    "quantile_crps",
    "rank_ic",
    "sharpe_ratio",
    "strategy_returns",
    "zscore_ensemble",
]
