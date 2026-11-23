import pandas as pd
import pytest

from src.data import _binance


def _raw_part(times: list[str], volumes: list[float], trades: list[int]) -> pd.DataFrame:
    """Binance kline rows (the columns of a monthly file) at `times`, price 1.5 when frozen."""
    open_time = pd.DatetimeIndex(times, tz="UTC").as_unit("ms").asi8
    rows = len(times)
    return pd.DataFrame({
        "open_time": open_time, "open": [1.5] * rows, "high": [1.6] * rows, "low": [1.4] * rows,
        "close": [1.5] * rows, "volume": volumes, "close_time": open_time + 899_999,
        "amount": [v * 1.5 for v in volumes], "trades": trades, "taker_buy_volume": [0.0] * rows,
        "taker_buy_amount": [0.0] * rows, "ignore": [0] * rows,
    })  # fmt: skip


def test_build_bars_masks_bars_without_a_trade(tmp_path):
    # a delisted coin: Binance keeps writing bars with the last price, volume 0 and 0 trades
    times = ["2024-06-11 22:00", "2024-06-11 22:15", "2024-06-11 22:30", "2024-06-11 22:45"]
    _raw_part(times, [5.0, 2.0, 0.0, 0.0], [3, 1, 0, 0]).to_parquet(tmp_path / "2024-06.parquet")
    bars = _binance.build_bars(tmp_path, start="2024-01-01", end="2025-01-01")
    assert list(bars.index) == list(pd.DatetimeIndex(times[:2], tz="UTC"))
    assert bars["volume"].tolist() == pytest.approx([5.0, 2.0])
    assert (bars["n_bars"] == 1).all()


def test_build_bars_keeps_a_bar_with_trades_or_volume_and_cuts_the_period(tmp_path):
    times = ["2023-12-31 23:45", "2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30"]
    part = _raw_part(times, [1.0, 0.0, 1.0, 1.0], [1, 2, 0, 1])
    part.to_parquet(tmp_path / "2024-01.parquet")
    bars = _binance.build_bars(tmp_path, start="2024-01-01", end="2024-01-01 00:30")
    assert list(bars.index) == list(pd.DatetimeIndex(times[1:3], tz="UTC"))
