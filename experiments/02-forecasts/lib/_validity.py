from typing import TYPE_CHECKING

import pandas as pd

from src.benchmark import (
    QUANTILE_COLUMNS,
    ForecastCache,
    cache_settings,
    is_after_release,
    on_model_grid,
    release_date,
    valid_quantiles,
)

if TYPE_CHECKING:
    from src.experiment import StudyConfig


def validity_table(cache: ForecastCache, config: "StudyConfig") -> pd.DataFrame:
    """Count the labelled decisions with a valid forecast, per model and window (Q1).

    A decision (time, asset) on a model's grid (Kronos: every `kronos_every_hours`) counts
    only after the model's release (baselines: always; `is_after_release`), the decisions
    every study scores. It is valid when the model has a valid row (`valid_quantiles`: nine
    finite quantiles in increasing order, equal neighbours allowed; the rule every scoring
    study applies) for every labelled horizon of the decision. A baseline fit that did not
    converge and is listed in the run's ``not_fitted.csv`` (asset and window) is the plan's
    listed exception: its decisions leave the count and are counted in ``not_fitted``.

    Parameters
    ----------
    cache : ForecastCache
        A 02-forecasts run.
    config : StudyConfig
        The configuration of that run (its ``config.toml``).

    Returns
    -------
    pd.DataFrame
        ``model``, ``window``, ``decisions``, ``valid``, ``not_fitted``, ``share``: one row per
        model and window with a decision after the model's release, plus one row per model with
        window ``all`` (a model released after every window has only that row, with no decision
        and a NaN share). A window whose decisions are all listed in ``not_fitted.csv`` keeps its
        row with no decision left, its ``not_fitted`` count and a NaN share; it does not decide
        Q1 and is named among the listed exceptions.

    Raises
    ------
    FileNotFoundError
        If a model of the run has no forecasts file (it was not built yet).
    """
    settings = cache_settings(config)
    listed = _not_fitted(cache)
    parts: list[pd.DataFrame] = []
    for model in cache.models["model"].astype(str):
        path = cache.folder / "forecasts" / f"{model}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{model} is not built in {cache.folder}: build it with run.py --run-id "
                f"{cache.folder.name} --models {model}"
            )
        labels = on_model_grid(cache.labels, model, settings)
        released = labels[is_after_release(labels, release_date(cache, model))]
        keys = released[["window", "asset"]].apply(tuple, axis=1)
        is_listed = keys.isin(listed.get(model, set())).to_numpy()
        decisions = _valid_decisions(released[~is_listed], cache.forecasts(model))
        # every window with a decision after the release, also one whose decisions are all
        # listed (it keeps its not_fitted count, with no decision left to count)
        windows = pd.Index(released["window"].unique(), name="window")
        table = decisions.groupby("window", sort=False)["valid"].agg(["size", "sum"])
        table = table.reindex(windows, fill_value=0)
        skipped = released[is_listed].drop_duplicates(["time", "asset"]).groupby("window").size()
        table["not_fitted"] = skipped.reindex(windows, fill_value=0)
        table.loc["all"] = table.sum()
        table.columns = ["decisions", "valid", "not_fitted"]
        parts.append(table.reset_index().assign(model=model))
    result = pd.concat(parts, ignore_index=True)
    result["share"] = result["valid"] / result["decisions"].where(result["decisions"] > 0)
    return result[["model", "window", "decisions", "valid", "not_fitted", "share"]]


def _not_fitted(cache: ForecastCache) -> dict[str, set[tuple[str, str]]]:
    # the (window, asset) pairs of each model listed in not_fitted.csv of the run, if any (a
    # pair listed twice, after a rerun that followed a crash, counts once: they form a set)
    path = cache.folder / "not_fitted.csv"
    if not path.exists():
        return {}
    listed = pd.read_csv(path)
    return {
        str(model): set(zip(rows["window"].astype(str), rows["asset"].astype(str), strict=True))
        for model, rows in listed.groupby("model")
    }


def _valid_decisions(labels: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    # a decision is valid when it has one valid forecast row for each of its labelled horizons
    is_valid = valid_quantiles(forecasts[list(QUANTILE_COLUMNS)].to_numpy(dtype=float))
    found = forecasts[is_valid].groupby(["time", "asset"]).size()
    decisions = labels.groupby(["time", "asset", "window"]).size().reset_index(name="needed")
    keys = pd.MultiIndex.from_frame(decisions[["time", "asset"]])
    decisions["valid"] = found.reindex(keys).to_numpy() == decisions["needed"]
    return decisions
