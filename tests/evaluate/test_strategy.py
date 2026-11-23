import numpy as np
import pandas as pd
import pytest
import torch

from src.enums import Market
from src.evaluate import (
    BINANCE_USDM_BPS,
    cost_per_side_bps,
    panel_rank_ic,
    portfolio_returns,
    sharpe_ratio,
    strategy_returns,
    zscore_ensemble,
)
from src.panel import portfolio_return_loss
from src.stats import deflated_sharpe_ratio, probability_of_backtest_overfitting


def test_costs_per_market():
    assert cost_per_side_bps(Market.CRYPTO, half_spread_bps=None, crypto_fee="taker") == 5.0
    assert cost_per_side_bps(Market.STOCK, half_spread_bps=1.2, crypto_fee="taker") == 1.2
    with pytest.raises(ValueError, match="half spread"):
        cost_per_side_bps(Market.FX, half_spread_bps=None, crypto_fee="maker")
    with pytest.raises(ValueError, match="maker or taker"):
        cost_per_side_bps(Market.CRYPTO, half_spread_bps=None, crypto_fee="vip")


def test_strategy_pays_costs_on_turnover():
    index = pd.RangeIndex(4)
    signal = pd.DataFrame({"A": [1.0, 1.0, -1.0, np.nan]}, index=index)
    realised = pd.DataFrame({"A": [0.01, 0.01, -0.01, 0.5]}, index=index)
    costs = pd.Series({"A": 10.0})
    strategy = strategy_returns(signal, realised, costs)
    assert strategy[("A", "gross")].tolist() == pytest.approx([0.01, 0.01, 0.01, 0.0])
    assert strategy[("A", "turnover")].tolist() == pytest.approx([1.0, 0.0, 2.0, 1.0])
    assert strategy[("A", "net")].iloc[2] == pytest.approx(0.01 - 2 * 10 / 1e4)
    assert strategy[("A", "net")].iloc[3] == pytest.approx(-10 / 1e4)
    assert portfolio_returns(strategy, kind="net").shape == (4,)


def test_binance_fees_are_the_regular_usdm_row():
    assert BINANCE_USDM_BPS["maker"] == pytest.approx(2.0)
    assert BINANCE_USDM_BPS["taker"] == pytest.approx(5.0)


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, 0.0, 0.01])
    assert sharpe_ratio(returns, periods_per_year=1) == pytest.approx(
        returns.mean() / returns.std()
    )
    assert sharpe_ratio(pd.Series([0.0, 0.0]), periods_per_year=252) == 0.0


def test_panel_rank_ic_and_ensemble():
    forecast = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [3.0, 1.0]})
    realised = pd.DataFrame({"A": [0.1, 0.3], "B": [0.2, 0.2], "C": [0.3, 0.1]})
    assert panel_rank_ic(forecast, realised, min_assets=3).tolist() == pytest.approx([1.0, 1.0])
    ensemble = zscore_ensemble([forecast, forecast * 100])
    assert ensemble.iloc[0].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="share"):
        zscore_ensemble([forecast, forecast.iloc[:1]])


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 1.0, 2000)
    few = deflated_sharpe_ratio(returns, trial_sharpes=rng.normal(0, 0.01, 2))
    many = deflated_sharpe_ratio(returns, trial_sharpes=rng.normal(0, 0.05, 1000))
    assert 0.0 <= many < few <= 1.0


def test_pbo_is_high_for_pure_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (800, 20))
    assert probability_of_backtest_overfitting(noise, n_blocks=8) > 0.2
    edge = noise.copy()
    edge[:, 0] += 0.5
    assert probability_of_backtest_overfitting(edge, n_blocks=8) < 0.05
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(noise, n_blocks=7)


def test_portfolio_return_loss_rewards_correct_ordering():
    target = torch.tensor([[1.0, 0.0, -1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    good = portfolio_return_loss(torch.tensor([[1.0, 0.0, -1.0]]), target, mask)
    bad = portfolio_return_loss(torch.tensor([[-1.0, 0.0, 1.0]]), target, mask)
    assert float(good) < 0 < float(bad)
    alone = portfolio_return_loss(
        torch.tensor([[1.0, 0.0, 0.0]]), target, torch.tensor([[True, False, False]])
    )
    assert float(alone) == 0.0
