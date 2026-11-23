import numpy as np
import pandas as pd
import pytest
import torch

from src.enums import Timeframe
from src.panel import (
    FEATURES,
    LSTMEncoder,
    Mixing,
    PanelConfig,
    PanelModel,
    PanelWindows,
    TransformerEncoder,
    bar_features,
    quantile_loss,
    train,
    vol_scaled_returns,
)


def _bars(timeframe: Timeframe, *, seed: int, weekdays_only: bool = False) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-05", "2026-02-02", freq=timeframe.pandas_freq, tz="UTC", inclusive="left"
    )
    if weekdays_only:
        index = index[index.day_of_week < 5]
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998, "close": close, "volume": 5.0},
        index=pd.DatetimeIndex(index, name="open_time"),
    )


def _panel(lookback: int = 8) -> PanelWindows:
    bars = {
        "BTC": {tf: _bars(tf, seed=1) for tf in (Timeframe.H1, Timeframe.H4)},
        "SPX": {tf: _bars(tf, seed=2, weekdays_only=True) for tf in (Timeframe.H1, Timeframe.H4)},
    }
    return PanelWindows.build(bars, lookback=lookback, max_age_hours=12)


def test_bar_features_use_mid_prices_and_start_with_zero_return():
    frame = _bars(Timeframe.H1, seed=0).rename(columns=lambda c: f"mid_{c}" if c != "volume" else c)
    features = bar_features(frame)
    assert features.shape == (len(frame), 4)
    assert features[0, 0] == 0.0
    assert np.isfinite(features).all()


def test_window_shapes_and_weekend_mask():
    panel = _panel()
    times = pd.DatetimeIndex(["2026-01-15 12:00", "2026-01-18 12:00"], tz="UTC")
    x, mask = panel.windows(times)
    assert x.shape == (2, 2, 2, 8, len(FEATURES))
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]
    assert (x[1, 1] == 0).all()


def test_windows_never_see_the_future():
    """Leakage test: changing bars that close after t leaves the windows at t unchanged."""
    cut = pd.Timestamp("2026-01-20 10:00", tz="UTC")
    original = {
        "BTC": {
            Timeframe.H1: _bars(Timeframe.H1, seed=1),
            Timeframe.H4: _bars(Timeframe.H4, seed=1),
        }
    }
    changed = {"BTC": {}}
    for timeframe, frame in original["BTC"].items():
        future = frame.index + timeframe.timedelta > cut
        altered = frame.copy()
        altered.loc[future, ["open", "high", "low", "close"]] *= 3.0
        changed["BTC"][timeframe] = altered
    times = pd.DatetimeIndex([cut - pd.Timedelta(hours=3), cut], tz="UTC")
    before, _ = PanelWindows.build(original, lookback=8, max_age_hours=12).windows(times)
    after, _ = PanelWindows.build(changed, lookback=8, max_age_hours=12).windows(times)
    np.testing.assert_array_equal(before, after)


def test_vol_scaled_labels_are_missing_when_the_market_is_closed():
    bars = _bars(Timeframe.H1, seed=3, weekdays_only=True)
    closes = bars["close"].set_axis(bars.index + pd.Timedelta(hours=1))
    times = pd.DatetimeIndex(["2026-01-14 12:00", "2026-01-10 12:00", "2026-01-05 03:00"], tz="UTC")
    labels = vol_scaled_returns(closes, times, horizon_hours=4, span_bars=24)
    assert np.isfinite(labels[0])
    assert np.isnan(labels[1])
    assert np.isnan(labels[2])


@pytest.mark.parametrize("mixing", list(Mixing))
def test_model_shapes_and_masked_assets_do_not_affect_others(mixing):
    torch.manual_seed(0)
    config = PanelConfig(
        d_model=16,
        n_timeframes=2,
        n_quantiles=3,
        mixing=mixing,
        n_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = PanelModel(LSTMEncoder(len(FEATURES), 16), config).eval()
    x = torch.randn(2, 3, 2, 8, len(FEATURES))
    mask = torch.tensor([[True, True, False], [True, True, True]])
    out = model(x, mask)
    assert out.shape == (2, 3, 3)
    noisy = x.clone()
    noisy[0, 2] = 100.0
    torch.testing.assert_close(model(noisy, mask)[0, :2], out[0, :2])


def test_model_rejects_bad_shapes_and_mixing():
    config = PanelConfig(16, 2, 3, Mixing.NONE, 2, 1, 0.0)
    model = PanelModel(TransformerEncoder(len(FEATURES), 16, 8, n_heads=2, num_layers=1), config)
    with pytest.raises(ValueError, match="batch, assets"):
        model(torch.randn(2, 3, 8, 5), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="Mixing"):
        PanelModel(LSTMEncoder(5, 16), PanelConfig(16, 2, 3, "both", 2, 1, 0.0))  # type: ignore[arg-type]


def test_quantile_loss_ignores_masked_and_missing():
    levels = torch.tensor([0.1, 0.5, 0.9])
    forecast = torch.zeros(1, 2, 3)
    target = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, True]])
    assert float(quantile_loss(forecast, target, levels, mask)) == pytest.approx(0.5)
    assert float(quantile_loss(forecast, target, levels, torch.tensor([[False, True]]))) == 0.0


def test_training_lowers_the_loss_and_restores_the_best_epoch():
    torch.manual_seed(0)
    x = torch.randn(64, 3, 1, 8, len(FEATURES))
    target = x[:, :, 0, -1, 0] * 2.0
    mask = torch.ones(64, 3, dtype=torch.bool)
    batches = [(x[i : i + 16], mask[i : i + 16], target[i : i + 16]) for i in range(0, 64, 16)]
    config = PanelConfig(16, 1, 3, Mixing.ASSET, 2, 1, 0.0)
    model = PanelModel(LSTMEncoder(len(FEATURES), 16), config)
    settings = {
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "max_epochs": 30,
        "patience": 30,
        "clip": 1.0,
    }
    history = train(
        model,
        batches,
        batches,
        levels=torch.tensor([0.1, 0.5, 0.9]),
        settings=settings,
        device="cpu",
    )
    assert history["validation_loss"][-1] < history["validation_loss"][0]
    assert (
        min(history["validation_loss"]) == history["validation_loss"][int(history["best_epoch"][0])]
    )
