import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.benchmark import own_bars_only


def test_the_closed_hours_are_dropped_and_the_own_bars_right_aligned() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    is_closed = np.array([[False, True, True, False, False], [False, False, False, False, False]])
    result = own_bars_only(values, is_closed)
    np.testing.assert_array_equal(result[0], [np.nan, np.nan, 1.0, 4.0, 5.0])
    np.testing.assert_array_equal(result[1], values[1])


@given(
    closed=arrays(bool, (4, 12)),
    values=arrays(float, (4, 12), elements=st.floats(-100, 100, allow_nan=False)),
)
def test_the_own_bars_keep_their_order_and_count(closed, values) -> None:
    result = own_bars_only(values, closed)
    for row in range(4):
        kept = result[row][~np.isnan(result[row])]
        np.testing.assert_array_equal(kept, values[row][~closed[row]])
        assert np.isnan(result[row][: closed[row].sum()]).all()
