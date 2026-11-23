"""The interval dispersion check (`src.benchmark.interval_dispersion`)."""

import numpy as np
import pandas as pd
import pytest

from src.benchmark import interval_dispersion

TIMES = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"], utc=True)


def _paths(rows: int, spreads: list[float]) -> np.ndarray:
    # closes h steps ahead spread evenly over 100 +- spread[h - 1], the same for every row
    grid = np.linspace(-1.0, 1.0, 101)
    steps = np.stack([100.0 + spread * grid for spread in spreads], axis=1)
    return np.broadcast_to(steps[None], (rows, 101, len(spreads))).copy()


def _contexts(assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"time": TIMES[: len(assets)], "asset": assets, "last_close": 100.0, "sigma": 0.01}
    )


def test_coverage_and_width_follow_the_label_transform_at_each_step() -> None:
    # step 1 spreads 99..101 and step 4 spreads 98..102: the 10-90 % interval is about
    # [99.2, 100.8] at step 1 and [98.4, 101.6] at step 4; in label units
    # log(q / 100) / (sigma * sqrt(h)); one label inside and one outside per horizon
    paths = _paths(2, [1.0, 1.5, 1.8, 2.0])
    labels = pd.DataFrame(
        {
            "time": list(TIMES) * 2,
            "asset": ["A", "B"] * 2,
            "horizon": [1, 1, 4, 4],
            "label": [0.0, 5.0, 0.1, -3.0],
        }
    )
    table = interval_dispersion([paths], [_contexts(["A", "B"])], labels, (1, 4))
    width_1 = (np.log(100.8) - np.log(99.2)) / 0.01
    width_4 = (np.log(101.6) - np.log(98.4)) / (0.01 * 2)
    assert table["decisions"].tolist() == [2, 2]
    assert table["coverage"].to_numpy() == pytest.approx([0.5, 0.5])
    assert table["median_width"].to_numpy() == pytest.approx([width_1, width_4])


def test_two_windows_are_pooled_and_a_decision_without_a_label_is_left_out() -> None:
    first, second = _contexts(["A"]), _contexts(["A", "B"]).assign(asset=["B", "C"])
    labels = pd.DataFrame(
        {"time": [TIMES[0], TIMES[0]], "asset": ["A", "B"], "horizon": [1, 1], "label": [0.0, 9.0]}
    )
    table = interval_dispersion([_paths(1, [1.0]), _paths(2, [1.0])], [first, second], labels, (1,))
    assert table["decisions"].tolist() == [2]
    assert table["coverage"].to_numpy() == pytest.approx([0.5])


def test_paths_and_contexts_must_have_the_same_rows() -> None:
    contexts = _contexts(["A"])
    with pytest.raises(ValueError, match="2 path rows for 1 contexts"):
        interval_dispersion([np.ones((2, 3, 1))], [contexts], pd.DataFrame(), (1,))


def test_a_bound_from_a_zero_price_quantile_covers_nothing_and_has_no_width() -> None:
    # paths at 0 and 200: the 10 % quantile is 0, its label bound -inf; the label inside would
    # otherwise count as covered with an infinite width
    paths = np.concatenate([np.zeros((1, 50, 1)), np.full((1, 51, 1), 200.0)], axis=1)
    labels = pd.DataFrame({"time": TIMES[:1], "asset": ["A"], "horizon": [1], "label": [0.0]})
    with np.errstate(divide="ignore"):
        table = interval_dispersion([paths], [_contexts(["A"])], labels, (1,))
    assert table["coverage"].tolist() == [0.0]
    assert np.isnan(table["median_width"].iloc[0])
