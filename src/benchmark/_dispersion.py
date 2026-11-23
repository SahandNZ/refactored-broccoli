"""The dispersion of sampled forecast intervals: coverage and width of the 10-90 % interval."""

import numpy as np
import pandas as pd

from src.benchmark._build import label_quantiles

# the central interval whose coverage is measured (a calibrated forecast covers 80 %)
LEVELS = (0.1, 0.9)


def interval_dispersion(
    paths: list[np.ndarray],
    contexts: list[pd.DataFrame],
    labels: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Return the coverage and median width of the 10-90 % interval of sampled close paths.

    The interval of each context and horizon is the 10 % and 90 % quantile of the paths' closes
    h steps ahead, turned into the label's scale by the cache's own transform
    (`src.benchmark.label_quantiles`: ``log(q / last close) / (sigma * sqrt(h))``); it covers
    a decision when its label lies inside. Decisions without a label at a horizon are left
    out. A bound that is not finite (NaN or -inf, from a price quantile at or below zero) never
    covers its label, and its width is left out of the median.

    Parameters
    ----------
    paths : list of np.ndarray
        Per window, the sampled closes, shape ``(contexts, paths, max(horizons))``.
    contexts : list of pd.DataFrame
        Per window, the contexts in the order of `paths`: ``time``, ``asset``, ``last_close``
        and ``sigma`` (`src.benchmark.with_last_close` on the model's grid decisions).
    labels : pd.DataFrame
        The cache's labels: ``time``, ``asset``, ``horizon`` and ``label``.
    horizons : tuple of int
        The label horizons in own bars.

    Returns
    -------
    pd.DataFrame
        One row per horizon: ``horizon``, ``decisions``, ``coverage`` (share of labels inside
        the interval) and ``median_width`` (in label units; a calibrated normal forecast of a
        unit-variance label has about 2.56).

    Raises
    ------
    ValueError
        If a window's paths and contexts differ in length.
    """
    parts: list[pd.DataFrame] = []
    for window_paths, window in zip(paths, contexts, strict=True):
        if len(window_paths) != len(window):
            raise ValueError(f"{len(window_paths)} path rows for {len(window)} contexts")
        prices = np.moveaxis(np.quantile(window_paths, LEVELS, axis=1), 0, -1)
        bounds = label_quantiles(
            prices,
            window["last_close"].to_numpy(dtype=float),
            window["sigma"].to_numpy(dtype=float),
            horizons,
        )
        for index, horizon in enumerate(horizons):
            parts.append(
                window[["time", "asset"]].assign(
                    horizon=horizon, low=bounds[:, index, 0], high=bounds[:, index, 1]
                )
            )
    merged = pd.concat(parts, ignore_index=True).merge(
        labels[["time", "asset", "horizon", "label"]], on=["time", "asset", "horizon"]
    )
    # a bound that is not finite (NaN, or -inf from a price quantile of 0) covers nothing and
    # has no width
    finite = np.isfinite(merged["low"].to_numpy()) & np.isfinite(merged["high"].to_numpy())
    inside = (merged["label"] >= merged["low"]) & (merged["label"] <= merged["high"])
    merged["inside"] = inside & finite
    merged["width"] = (merged["high"] - merged["low"]).where(finite)
    return (
        merged.groupby("horizon")
        .agg(
            decisions=("label", "size"),
            coverage=("inside", "mean"),
            median_width=("width", "median"),
        )
        .reset_index()
    )
