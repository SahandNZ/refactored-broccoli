"""Study 07-rq5-off-hours, Q1: forecasts of the first bar after a weekend reopening.

Every model forecasts the close of the first bar after each reopening from the same 512
calendar hours in three arms (`MODEL_ARMS`): only the asset's own bars in them (event time,
as 08-closures' "trading hours only"), all hours with the closed ones missing (calendar
time), and the calendar hours with the weekend crypto basket as a past covariate. Two arms
need no model input beyond the event forecast: the event forecast moved to a beta-on-crypto
centre and widened (the correction), and beta times the crypto move plus the quantiles of
earlier residuals (model-free). Every forecast is in the benchmark's label units: the log
move from Friday's close over the EWMA volatility at that close. `reopening_forecasts`
writes them.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.benchmark import (
    QUANTILE_COLUMNS,
    calendar_windows,
    fold_names,
    label_quantiles,
    own_bars_only,
    valid_quantiles,
)
from src.evaluate import quantile_crps
from src.experiment import seed_everything
from src.models import QUANTILE_LEVELS, Forecaster, build_forecaster, require_revision

from ._settings import OffHoursSettings
from ._weekends import Weekends

logger = logging.getLogger(__name__)

MODEL_ARMS = ("event", "calendar", "covariate")
"""The arms a model forecasts itself, all from the same calendar hours."""

ARMS = (*MODEL_ARMS, "correction", "model-free")
"""Every forecast of a model and weekend."""

WEEKEND_COLUMNS = [
    "asset",
    "market",
    "fold",
    "week",
    "close_time",
    "reopen_time",
    "label_time",
    "friday_close",
    "sigma",
    "weekend_crypto",
    "move",
    "label",
]
"""The columns of a weekend in the forecasts table; ``label`` (``move / sigma``) is scored."""

LEVELS = np.asarray(QUANTILE_LEVELS)


# --- contexts -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReopeningContexts:
    """The model inputs of every weekend, all cut from the same calendar hours.

    Attributes
    ----------
    event : np.ndarray
        The asset's own closes inside the hours, in order, right-aligned and NaN-padded on
        the left, shape ``(n, lookback)``; only the last ``own_bars`` values are given to a
        model.
    own_bars : np.ndarray
        The number of own closes in each window, shape ``(n,)``; 0 without enough history.
    calendar : np.ndarray
        The asset's closes on the `lookback` hours up to ``reopen_time``, NaN while closed,
        shape ``(n, lookback)``; NaN rows without that many hours of history.
    covariates : np.ndarray
        The basket's mean log return since ``close_time`` on the same hours (the weekend
        crypto move at the last hour; NaN where an asset has no bar), shape
        ``(n, 1, lookback)``.
    """

    event: np.ndarray
    own_bars: np.ndarray
    calendar: np.ndarray
    covariates: np.ndarray


def reopening_contexts(weekends: pd.DataFrame, data: Weekends, lookback: int) -> ReopeningContexts:
    """Cut the model inputs of each weekend: nothing after the reopening is read.

    Every arm reads the `lookback` calendar hours whose close is at or before
    ``reopen_time`` (`calendar_windows`): the event arm only the own closes in them (the last
    is Friday's close), the calendar and covariate arms every hour.

    Parameters
    ----------
    weekends : pd.DataFrame
        Rows of `Weekends.table`.
    data : Weekends
        The closes of the closed-market assets and the crypto basket.
    lookback : int
        Calendar hours per context.

    Returns
    -------
    ReopeningContexts
        The contexts, in the row order of `weekends`.
    """
    n = len(weekends)
    event, calendar = np.full((n, lookback), np.nan), np.full((n, lookback), np.nan)
    covariates = np.full((n, 1, lookback), np.nan)
    own_bars = np.zeros(n, dtype=int)
    for asset in weekends["asset"].unique():
        rows = np.flatnonzero(weekends["asset"] == asset)
        part = weekends.iloc[rows]
        closes = data.closes[str(asset)]
        hours = calendar_windows(closes, pd.DatetimeIndex(part["reopen_time"]), lookback)
        valid = rows[hours.is_valid]
        event[valid] = own_bars_only(hours.values, hours.is_closed)[hours.is_valid]
        own_bars[valid] = (~hours.is_closed[hours.is_valid]).sum(axis=1)
        calendar[valid] = np.where(hours.is_closed, np.nan, hours.values)[hours.is_valid]
        crypto = data.crypto.reindex(hours.hours).to_numpy(dtype=float)
        crypto = crypto.reshape(len(rows), lookback, -1)
        anchor = data.crypto.reindex(pd.DatetimeIndex(part["close_time"])).to_numpy(dtype=float)
        covariates[rows, 0] = (crypto - anchor[:, None, :]).mean(axis=2)
    return ReopeningContexts(event, own_bars, calendar, covariates)


# --- the correction and the model-free arm ------------------------------------------------


def fit_correction(
    quantiles: np.ndarray, crypto: np.ndarray, labels: np.ndarray, width_factors: np.ndarray
) -> tuple[float, float]:
    """Fit the beta-on-crypto centre and the width factor of the correction.

    The corrected forecast of a weekend is ``beta * crypto + width * (q - median(q))``: the
    event-time quantiles ``q`` moved to a centre on the crypto move and widened. ``beta`` is
    the least-squares slope through the origin of the labels on the crypto moves; ``width``
    is the factor of `width_factors` with the lowest mean CRPS on the same weekends.

    Parameters
    ----------
    quantiles : np.ndarray
        Event-time label quantiles of the fitting weekends, shape ``(m, 9)``.
    crypto : np.ndarray
        Their weekend crypto moves in label units (over the asset's sigma), shape ``(m,)``,
        not all zero.
    labels : np.ndarray
        Their labels, shape ``(m,)``.
    width_factors : np.ndarray
        The candidate factors, shape ``(n_factors,)``.

    Returns
    -------
    tuple of (float, float)
        ``beta`` and ``width``.
    """
    beta = float(crypto @ labels / (crypto @ crypto))
    spread = quantiles - quantiles[:, 4:5]
    candidates = beta * crypto[None, :, None] + width_factors[:, None, None] * spread[None]
    targets = np.tile(labels, len(width_factors))
    crps = quantile_crps(candidates.reshape(-1, len(LEVELS)), LEVELS, targets)
    mean_crps = crps.reshape(len(width_factors), -1).mean(axis=1)
    return beta, float(width_factors[np.argmin(mean_crps)])


def _fit_rows(weekends: pd.DataFrame, is_usable: np.ndarray, row: int) -> np.ndarray:
    # the earlier weekends a fit of weekend `row` may use: the same asset, usable, and a label
    # known at the row's Friday close
    same_asset = (weekends["asset"] == weekends["asset"].iloc[row]).to_numpy()
    is_known = pd.DatetimeIndex(weekends["label_time"]) <= weekends["close_time"].iloc[row]
    return is_usable & same_asset & np.asarray(is_known)


def _scaled_crypto(weekends: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # the crypto move in label units (over the asset's sigma at the close) and the label
    crypto = weekends["weekend_crypto"].to_numpy(dtype=float) / weekends["sigma"].to_numpy()
    return crypto, weekends["label"].to_numpy(dtype=float)


def corrected_quantiles(
    event: pd.DataFrame, settings: OffHoursSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return the correction of every event-time forecast, fitted on earlier weekends only.

    The fit of a weekend uses the valid event-time forecasts of the same asset and model with
    a finite label and crypto move whose label was known at its Friday close (``label_time <=
    close_time``); everything is in label units, the crypto move over the asset's sigma.

    Parameters
    ----------
    event : pd.DataFrame
        The event-time rows of one model: `WEEKEND_COLUMNS` and ``QUANTILE_COLUMNS``.
    settings : OffHoursSettings
        The study's settings (``min_fit_weekends``, ``width_factors``).

    Returns
    -------
    quantiles : np.ndarray
        Label quantiles in the row order of `event`, shape ``(n, 9)``; NaN where a weekend
        has fewer than ``min_fit_weekends`` earlier weekends or no forecast of its own.
    widths : np.ndarray
        The fitted width factor of each weekend, shape ``(n,)``; NaN without a fit. A width
        at either end of ``width_factors`` means the grid may have bound the fit.
    """
    quantiles = event[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    crypto, labels = _scaled_crypto(event)
    is_valid = valid_quantiles(quantiles) & np.isfinite(labels) & np.isfinite(crypto)
    corrected = np.full(quantiles.shape, np.nan)
    widths = np.full(len(event), np.nan)
    for row in range(len(event)):
        fit = _fit_rows(event, is_valid, row)
        if fit.sum() < settings.min_fit_weekends:
            continue
        beta, width = fit_correction(
            quantiles[fit], crypto[fit], labels[fit], settings.width_factors
        )
        corrected[row] = beta * crypto[row] + width * (quantiles[row] - quantiles[row, 4])
        widths[row] = width
    return corrected, widths


def model_free_quantiles(weekends: pd.DataFrame, settings: OffHoursSettings) -> np.ndarray:
    """Return the model-free forecast of every weekend: beta times crypto plus past residuals.

    For each weekend, ``beta`` is the least-squares slope through the origin of the labels on
    the crypto moves (in label units) of the earlier weekends of the same asset whose label
    was known at its Friday close, and the quantiles are ``beta * crypto`` plus the
    `QUANTILE_LEVELS` quantiles of those weekends' residuals. The fit uses the weekends the
    correction may use, without needing a model forecast, so the arm is the same for every
    model.

    Parameters
    ----------
    weekends : pd.DataFrame
        `WEEKEND_COLUMNS` of the forecast weekends.
    settings : OffHoursSettings
        The study's settings (``min_fit_weekends``).

    Returns
    -------
    np.ndarray
        Label quantiles in the row order of `weekends`, shape ``(n, 9)``; NaN where a weekend
        has fewer than ``min_fit_weekends`` earlier weekends.
    """
    crypto, labels = _scaled_crypto(weekends)
    is_usable = np.isfinite(labels) & np.isfinite(crypto)
    quantiles = np.full((len(weekends), len(LEVELS)), np.nan)
    for row in range(len(weekends)):
        fit = _fit_rows(weekends, is_usable, row)
        if fit.sum() < settings.min_fit_weekends:
            continue
        beta = float(crypto[fit] @ labels[fit] / (crypto[fit] @ crypto[fit]))
        residuals = labels[fit] - beta * crypto[fit]
        quantiles[row] = beta * crypto[row] + np.quantile(residuals, LEVELS)
    return quantiles


# --- forecasts ----------------------------------------------------------------------------


def _arm_rows(
    keys: pd.DataFrame, model: str, arm: str, quantiles: np.ndarray, widths: np.ndarray
) -> pd.DataFrame:
    table = pd.DataFrame(quantiles, columns=list(QUANTILE_COLUMNS))
    return pd.concat([keys, table], axis=1).assign(model=model, arm=arm, width=widths)


def _model_arms(
    forecaster: Forecaster, contexts: ReopeningContexts, weekends: pd.DataFrame
) -> dict[str, np.ndarray]:
    # the three arms of one model as label quantiles, shape (n, 9) each: the event arm in one
    # call per context length (its padding is cut off, so the model never sees a NaN there),
    # the calendar and covariate arms in one call each; NaN where a weekend has no context
    n = len(weekends)
    prices = {arm: np.full((n, 1, len(LEVELS)), np.nan) for arm in MODEL_ARMS}
    for length in np.unique(contexts.own_bars[contexts.own_bars > 0]):
        rows = contexts.own_bars == length
        forecast = forecaster.predict(contexts.event[rows, -length:], 1)
        prices["event"][rows] = forecast.checked_quantiles(int(rows.sum()), 1)
    has_hours = contexts.own_bars > 0
    if not has_hours.any():
        raise ValueError(
            "no weekend has a full calendar context: check split_date against the data"
        )
    calendar = contexts.calendar[has_hours]
    forecast = forecaster.predict(calendar, 1)
    prices["calendar"][has_hours] = forecast.checked_quantiles(len(calendar), 1)
    forecast = forecaster.predict_with_covariates(calendar, contexts.covariates[has_hours], 1)
    prices["covariate"][has_hours] = forecast.checked_quantiles(len(calendar), 1)
    last_close = weekends["friday_close"].to_numpy(dtype=float)
    sigma = weekends["sigma"].to_numpy(dtype=float)
    return {
        arm: label_quantiles(values, last_close, sigma, (1,))[:, 0]
        for arm, values in prices.items()
    }


def reopening_forecasts(
    data: Weekends, assets: list[str], settings: OffHoursSettings
) -> pd.DataFrame:
    """Forecast the first bar after every reopening from ``split_date`` with every model.

    Each model's pinned revision is checked (`src.models.require_revision`) before it is
    built. Each model forecasts one step from its event-time, calendar and covariate inputs
    (`reopening_contexts`); the price quantiles become quantiles of the benchmark's label
    (`label_quantiles` with the EWMA sigma at Friday's close). The correction is fitted on the
    model's own event forecasts, the model-free arm once for all models. A weekend without a
    full context, or without a crypto move (the correction and the model-free arm need it),
    keeps NaN quantiles in that arm, so it is counted when the cells are scored.
    No weekend is cut at a model's release: the arms of a model are compared on the same
    weekends.

    Parameters
    ----------
    data : Weekends
        The weekends and prices of `read_weekends`.
    assets : list of str
        The assets to forecast (`AssetSets.forecast`).
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    pd.DataFrame
        `WEEKEND_COLUMNS`, ``QUANTILE_COLUMNS``, ``model``, ``arm`` (`ARMS`) and ``width``
        (the correction's fitted width factor, NaN in the other arms), one row per weekend,
        model and arm.
    """
    table = data.table
    is_chosen = table["asset"].isin(assets) & (table["reopen_time"] >= settings.split_date)
    weekends = table[is_chosen].reset_index(drop=True)
    weekends["fold"] = fold_names(weekends["reopen_time"], settings.scored_folds)
    contexts = reopening_contexts(weekends, data, settings.lookback)
    keys = weekends[WEEKEND_COLUMNS]
    model_free = model_free_quantiles(keys, settings)
    parts: list[pd.DataFrame] = []
    for model, (kind, repository) in settings.zero_shot.items():
        require_revision(repository, settings.revisions[model])
        seed_everything(settings.seed)
        forecaster = build_forecaster(kind, name=repository, device=settings.device)
        arms = _model_arms(forecaster, contexts, weekends)
        del forecaster
        torch.cuda.empty_cache()
        no_width = np.full(len(keys), np.nan)
        event = _arm_rows(keys, model, "event", arms["event"], no_width)
        arms["correction"], widths = corrected_quantiles(event, settings)
        arms["model-free"] = model_free
        parts += [
            _arm_rows(keys, model, arm, arms[arm], widths if arm == "correction" else no_width)
            for arm in ARMS
        ]
        logger.info("%s: %d weekends forecast", model, len(weekends))
    return pd.concat(parts, ignore_index=True)
