import pandas as pd


def zscore_ensemble(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Average several models' forecasts after z-scoring each across assets at every time.

    Z-scoring puts models with different scales on one footing; the mean over models keeps the
    assets each model forecast (a model without a forecast for an asset is left out there).

    Parameters
    ----------
    predictions : list of pd.DataFrame
        One per model, each ``(n_times, n_assets)`` on the same index and columns.

    Returns
    -------
    pd.DataFrame
        The ensemble forecast, same shape.

    Raises
    ------
    ValueError
        If the list is empty or the frames do not share index and columns.
    """
    if not predictions:
        raise ValueError("need at least one model's predictions")
    first = predictions[0]
    for frame in predictions[1:]:
        if not (frame.index.equals(first.index) and frame.columns.equals(first.columns)):
            raise ValueError("predictions must share index and columns")
    scores = [
        frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)
        for frame in predictions
    ]
    return pd.concat(scores).groupby(level=0, sort=False).mean().loc[first.index]
