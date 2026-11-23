import numpy as np

from src.benchmark import cell_stream


def test_a_cells_stream_depends_on_its_names_not_on_order() -> None:
    first = cell_stream(0, "power", "kronos-small", "crypto", 1).random(5)
    again = cell_stream(0, "power", "kronos-small", "crypto", 1).random(5)
    np.testing.assert_array_equal(first, again)
    other_cell = cell_stream(0, "power", "kronos-small", "forex", 1).random(5)
    other_seed = cell_stream(1, "power", "kronos-small", "crypto", 1).random(5)
    other_use = cell_stream(0, "skill", "kronos-small", "crypto", 1).random(5)
    for draws in (other_cell, other_seed, other_use):
        assert not np.allclose(first, draws)
