"""The bootstrap of one book of 05-rq3-net (its ``lib``): `book_block`, `book_bootstrap`."""

import math
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.stats import bootstrap_mean_test, circular_block_indices, politis_white_block

type Series3 = tuple[np.ndarray, np.ndarray, np.ndarray]


def _frame(net: np.ndarray, gross: np.ndarray, turnover: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"net": net, "gross": gross, "turnover": turnover})


def _book(seed: int, n: int = 400) -> Series3:
    # every time is effective: the gross return is never exactly 0
    rng = np.random.default_rng(seed)
    turnover = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.5, 0.3, 0.2])
    gross = 0.0004 * turnover + 0.001 * rng.standard_normal(n)
    return gross - turnover * 2e-4, gross, turnover


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    values = np.zeros(n)
    for step in range(1, n):
        values[step] = phi * values[step - 1] + noise[step]
    return values


# --- is_effective ---------------------------------------------------------------------------


def test_a_time_counts_with_a_return_or_a_trade_above_horizon_one(rq3: ModuleType) -> None:
    book = pd.DataFrame(
        {"gross": [0.01, 0.0, 0.0, 0.0, 0.02], "turnover": [1.0, 0.0, 1.0, 0.0, 0.0]}
    )
    assert rq3.is_effective(book, horizon=4).tolist() == [True, False, True, False, True]
    assert rq3.is_effective(book, horizon=1).all()


# --- book_block and book_bootstrap -----------------------------------------------------------


def _run(
    rq3: ModuleType,
    book: pd.DataFrame,
    *,
    horizon: int,
    level: float = 0.95,
    n_resamples: int = 400,
    seed: int = 1,
) -> dict[str, float]:
    # the bootstrap of a book with its own block (`book_block`, no overlap) and a fresh stream
    block = rq3.book_block(book, horizon=horizon, overlap=0)
    rng = np.random.default_rng(seed)
    result = rq3.book_bootstrap(
        book, horizon=horizon, block=block, level=level, n_resamples=n_resamples, rng=rng
    )
    return result | {"block": block}


def test_the_net_test_is_the_shared_circular_block_t_test(rq3: ModuleType) -> None:
    # one series and one seed: the same numbers as src.stats.bootstrap_mean_test
    net, gross, turnover = _book(3)
    result = _run(rq3, _frame(net, gross, turnover), horizon=1, level=0.9, n_resamples=500, seed=7)
    block = max(politis_white_block(series) for series in (net, gross, turnover))
    shared = bootstrap_mean_test(net, null=0.0, level=0.9, n_resamples=500, seed=7, block=block)
    assert result["block"] == block
    assert result["p_value"] == pytest.approx(shared["p_greater"])
    assert result["net_low"] == pytest.approx(shared["low"])
    assert result["net_high"] == pytest.approx(shared["high"])


def test_the_break_even_interval_uses_the_same_circular_draws(rq3: ModuleType) -> None:
    net, gross, turnover = _book(3)
    result = _run(rq3, _frame(net, gross, turnover), horizon=1, level=0.9, n_resamples=500, seed=7)
    block = int(result["block"])
    rng = np.random.default_rng(7)
    indices = circular_block_indices(len(net), rng, n_resamples=500, block=block)
    ratios = gross[indices].mean(axis=1) / turnover[indices].mean(axis=1) * 1e4
    assert result["break_even_low"] == pytest.approx(np.quantile(ratios, 0.05))
    assert result["break_even_high"] == pytest.approx(np.quantile(ratios, 0.95))
    point = gross.mean() / turnover.mean() * 1e4
    assert result["break_even_low"] < point < result["break_even_high"]


def test_the_block_is_the_longest_of_the_three_series(rq3: ModuleType) -> None:
    # white-noise returns, but a turnover from a persistent sign path: it trades in runs
    rng = np.random.default_rng(6)
    sign = np.where(_ar1(0.97, 1500, seed=6) > 0, 1.0, -1.0)
    is_trading = np.repeat(rng.random(60) < 0.5, 25)
    turnover = np.abs(np.diff(sign, prepend=0.0)) + is_trading
    gross = 1e-3 * rng.standard_normal(1500)
    net = gross - turnover * 1e-4
    block = rq3.book_block(_frame(net, gross, turnover), horizon=1, overlap=0)
    assert block == max(politis_white_block(series) for series in (net, gross, turnover))
    assert block > politis_white_block(net)


def test_the_block_has_the_floor_of_rq1(rq3: ModuleType) -> None:
    # white noise has a Politis-White block of 1: the floor is 2, or 4 times the overlap
    noise = np.random.default_rng(8).standard_normal(400)
    book = _frame(noise, noise, np.ones(400))
    assert politis_white_block(noise) == 1
    assert rq3.book_block(book, horizon=1, overlap=0) == 2
    assert rq3.book_block(book, horizon=1, overlap=3) == 12


def test_the_zero_times_of_a_long_horizon_are_left_out(rq3: ModuleType) -> None:
    # an h = 24 book: an autocorrelated return every 24 times and zeros in between; the
    # block is the one of the effective series, the interval stays per book time
    effective = 1e-3 * _ar1(0.6, 600, seed=2) + 1e-4
    padded = np.zeros(600 * 24)
    padded[::24] = effective
    turnover = np.zeros(600 * 24)
    turnover[::24] = 1.0
    book = _frame(padded - turnover * 1e-5, padded, turnover)
    result = _run(rq3, book, horizon=24, n_resamples=300)
    block = max(
        politis_white_block(series) for series in (effective - 1e-5, effective, np.ones(600))
    )
    assert result["block"] == max(block, 2)
    assert block > 1
    assert politis_white_block(padded) < block
    assert result["net_low"] < book["net"].mean() < result["net_high"]
    # the same values without the zero times: p-value and break-even unchanged, the net
    # interval 24 times wider (per effective time instead of per book time)
    alone = _frame(effective - 1e-5, effective, np.ones(600))
    unpadded = _run(rq3, alone, horizon=24, n_resamples=300)
    assert result["p_value"] == pytest.approx(unpadded["p_value"])
    assert result["break_even_low"] == pytest.approx(unpadded["break_even_low"])
    assert result["break_even_high"] == pytest.approx(unpadded["break_even_high"])
    assert result["net_low"] * 24 == pytest.approx(unpadded["net_low"])


def test_the_p_value_is_one_sided(rq3: ModuleType) -> None:
    # with the same draws and block, the p-values of a series and of its negative add to 1
    net, gross, turnover = _book(4)
    block = rq3.book_block(_frame(net, gross, turnover), horizon=1, overlap=0)

    def p_value(values: np.ndarray) -> float:
        rng = np.random.default_rng(1)
        frame = _frame(values, gross, turnover)
        options = {"horizon": 1, "block": block, "level": 0.95, "n_resamples": 400}
        return rq3.book_bootstrap(frame, rng=rng, **options)["p_value"]

    assert p_value(net) + p_value(-net) == pytest.approx(1.0)
    assert p_value(np.abs(net) + 1e-4) < 0.001


def test_a_draw_without_turnover_leaves_the_break_even_interval_empty(
    rq3: ModuleType,
) -> None:
    # one trade in 40 times: some draws miss it
    turnover = np.zeros(40)
    turnover[0] = 1.0
    gross = np.linspace(-1e-3, 1e-3, 40)
    book = _frame(gross - turnover * 1e-4, gross, turnover)
    rng = np.random.default_rng(0)
    result = rq3.book_bootstrap(book, horizon=1, block=2, level=0.95, n_resamples=300, rng=rng)
    assert math.isnan(result["break_even_low"])
    assert math.isnan(result["break_even_high"])
    assert not math.isnan(result["net_low"])


def test_a_book_with_too_few_blocks_raises(rq3: ModuleType) -> None:
    net, gross, turnover = _book(9, n=20)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="20 effective times hold fewer than 5 blocks of 5"):
        rq3.book_bootstrap(
            _frame(net, gross, turnover), horizon=1, block=5, level=0.95, n_resamples=10, rng=rng
        )
