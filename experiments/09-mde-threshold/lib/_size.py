import numpy as np
import pandas as pd

from src.stats import bootstrap_difference, bootstrap_mean_test, politis_white_block

OVERLAP = 23
"""The overlap of 24-bar labels decided every bar: each shares its label with the next 23."""


def _overlapping(rng: np.random.Generator, length: int) -> np.ndarray:
    # 24-bar sums of white noise: a series with no edge whose neighbours share 23 terms
    noise = rng.standard_normal(length + OVERLAP)
    return np.convolve(noise, np.ones(OVERLAP + 1), mode="valid")[:length]


def bootstrap_size_table(
    *, lengths: tuple[int, ...], floors: tuple[int, ...], draws: int, resamples: int, seed: int
) -> pd.DataFrame:
    """Return the one-sided size of the shared bootstrap tests with overlapping 24-bar labels.

    For each series length and block floor, `draws` pairs of independent series without an
    edge (24-bar sums of white noise) are tested at a nominal 5 %, one-sided: the mean of the
    first against zero (`src.stats.bootstrap_mean_test`) and the difference of the two
    means (`src.stats.bootstrap_difference`). The block is the Politis-White block of the
    difference, at least ``floor x 23`` (the floor times the overlap) and at least 2, as in
    03-rq1-null. The rejection rate is the test's size; a correct test rejects about 5 %.

    Parameters
    ----------
    lengths : tuple of int
        Series lengths in decision times.
    floors : tuple of int
        Block floors in multiples of the overlap (e.g. 2 and 4).
    draws : int
        Pairs of series per length and floor.
    resamples : int
        Bootstrap draws per test.
    seed : int
        Seed of the series and the bootstraps (draw ``i`` uses ``seed + i``).

    Returns
    -------
    pd.DataFrame
        One row per length and floor: ``length``, ``floor``, ``median_blocks``,
        ``mean_test_size``, ``difference_test_size``, ``draws``, ``nominal`` (0.05).
    """
    rows: list[dict[str, float]] = []
    for length in lengths:
        for floor in floors:
            mean_rejected = difference_rejected = 0
            blocks: list[int] = []
            for draw in range(draws):
                rng = np.random.default_rng(seed + draw)
                first, second = _overlapping(rng, length), _overlapping(rng, length)
                block = max(politis_white_block(first - second), floor * OVERLAP, 2)
                blocks.append(length // block)
                options = {"level": 0.95, "n_resamples": resamples, "seed": seed + draw}
                mean = bootstrap_mean_test(first, null=0.0, block=block, **options)
                difference = bootstrap_difference(first, second, blocks=(block, block), **options)
                mean_rejected += mean["p_greater"] < 0.05
                difference_rejected += difference["p_greater"] < 0.05
            rows.append(
                {
                    "length": length,
                    "floor": floor,
                    "median_blocks": float(np.median(blocks)),
                    "mean_test_size": mean_rejected / draws,
                    "difference_test_size": difference_rejected / draws,
                    "draws": draws,
                    "nominal": 0.05,
                }
            )
    return pd.DataFrame(rows)
