import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.enums import Timeframe
from src.evaluate import Fold, quantile_crps
from src.experiment import seed_everything
from src.panel._dataset import PanelData
from src.panel._encoders import LSTMEncoder
from src.panel._inputs import FEATURES
from src.panel._model import Mixing, PanelConfig, PanelModel
from src.panel._training import train

logger = logging.getLogger(__name__)

type Batches = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class TimeframePlan:
    """What study 02 trains on each fold.

    Attributes
    ----------
    timeframes : tuple of Timeframe
        Every timeframe of the panel, in the order of its windows.
    input_sets : dict of str to list of Timeframe
        The input sets compared (e.g. ``S1 = [1h]``).
    seeds : tuple of int
        Seeds of every set.
    options : dict
        Training and split settings of ``study.toml`` ``[options]``.
    """

    timeframes: tuple[Timeframe, ...]
    input_sets: dict[str, list[Timeframe]]
    seeds: tuple[int, ...]
    options: dict[str, float]


@dataclass(frozen=True, slots=True)
class FoldSplit:
    """Decision positions of one fold in a `PanelData`.

    Training, early-stopping validation and the fold's evaluation quarter (never used for
    training or stopping).
    """

    train: np.ndarray
    validation: np.ndarray
    evaluation: np.ndarray


def split_fold(times: pd.DatetimeIndex, fold: Fold, options: dict[str, float]) -> FoldSplit:
    """Cut hourly decision times into the three parts of a fold.

    A gap of one forecast horizon separates the parts, so no label reaches into the next one.
    Training covers ``train_days`` before the validation, sampled every
    ``train_every_hours``; validation the ``validation_days`` before the evaluation quarter.
    """
    gap = pd.Timedelta(hours=options["horizon_hours"])
    validation_start = fold.evaluation_start - pd.Timedelta(days=options["validation_days"])
    train_start = validation_start - pd.Timedelta(days=options["train_days"])
    position = np.arange(len(times))
    in_train = (times >= train_start) & (times < validation_start - gap)
    in_validation = (times >= validation_start) & (times < fold.evaluation_start - gap)
    in_evaluation = (times >= fold.evaluation_start) & (times < fold.evaluation_end)
    train_positions = position[in_train][:: int(options["train_every_hours"])]
    return FoldSplit(train_positions, position[in_validation], position[in_evaluation])


def _select(batches: Batches, indices: list[int]) -> Batches:
    return [(x[:, :, indices].contiguous(), mask, target) for x, mask, target in batches]


def _predict(model: torch.nn.Module, batches: Batches, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, mask, _ in batches:
            outputs.append(model(x.to(device), mask.to(device)).float().cpu().numpy())
    return np.concatenate(outputs)


def climatology(labels: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return each asset's training-label quantiles (the naive baseline), ``(n_assets, levels)``."""
    rows = []
    for column in labels.T:
        finite = column[np.isfinite(column)]
        rows.append(np.quantile(finite, levels) if len(finite) else np.zeros(len(levels)))
    return np.stack(rows)


def evaluate_forecasts(
    quantiles: np.ndarray, labels: np.ndarray, mask: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Return the mean CRPS per asset over its valid decisions, ``(n_assets,)`` (NaN if none)."""
    result = np.full(labels.shape[1], np.nan)
    for asset in range(labels.shape[1]):
        valid = mask[:, asset] & np.isfinite(labels[:, asset])
        if valid.any():
            scores = quantile_crps(quantiles[valid, asset], levels, labels[valid, asset])
            result[asset] = scores.mean()
    return result


def _train_one(
    plan: TimeframePlan,
    indices: list[int],
    seed: int,
    batches: tuple[Batches, Batches],
    device: str,
) -> tuple[torch.nn.Module, int]:
    options = plan.options
    levels = torch.tensor(options["levels"], dtype=torch.float32)
    seed_everything(seed)
    size = int(options["d_model"])
    config = PanelConfig(size, len(indices), len(levels), Mixing.BOTH, 4, 1, 0.1)
    model = PanelModel(LSTMEncoder(len(FEATURES), size), config)
    keys = ("learning_rate", "weight_decay", "max_epochs", "patience", "clip")
    settings = {key: float(options[key]) for key in keys}
    train_batches, validation_batches = batches
    history = train(
        model, _select(train_batches, indices), _select(validation_batches, indices),
        levels=levels, settings=settings, device=device,
    )  # fmt: skip
    return model, int(history["best_epoch"][0])


def run_fold(
    data: PanelData, fold: Fold, plan: TimeframePlan, *, device: str
) -> dict[str, pd.DataFrame]:
    """Train and score every input set and seed of study 02 on one development fold.

    Returns
    -------
    dict of str to pd.DataFrame
        ``crps``: fold, set, seed, asset, crps, naive_crps, best_epoch. ``masking``: fold,
        seed, masked timeframe, asset, crps with that timeframe zeroed and the full crps of the
        largest set. ``forecasts``: the largest set's median forecast per evaluation hour and
        asset, with a ``seed`` column.
    """
    options = plan.options
    levels = np.asarray(options["levels"], dtype=np.float32)
    split = split_fold(data.times, fold, options)
    size = int(options["batch_size"])
    validation_batches = data.batches(split.validation, batch_size=size, shuffle_seed=0)
    evaluation_batches = data.batches(split.evaluation, batch_size=size, shuffle_seed=None)
    mask = np.concatenate([batch[1].numpy() for batch in evaluation_batches])
    labels = data.labels[split.evaluation]
    naive = climatology(data.labels[split.train], levels)
    naive_forecast = np.broadcast_to(naive, (len(split.evaluation), *naive.shape))
    naive_crps = evaluate_forecasts(naive_forecast, labels, mask, levels)
    assets = data.windows.assets
    largest = max(plan.input_sets, key=lambda name: len(plan.input_sets[name]))
    crps_rows, masking_rows, forecasts = [], [], []
    for seed in plan.seeds:
        train_batches = data.batches(split.train, batch_size=size, shuffle_seed=seed)
        for name, chosen in plan.input_sets.items():
            indices = [plan.timeframes.index(timeframe) for timeframe in chosen]
            model, best_epoch = _train_one(
                plan, indices, seed, (train_batches, validation_batches), device
            )
            quantiles = _predict(model, _select(evaluation_batches, indices), device)
            crps = evaluate_forecasts(quantiles, labels, mask, levels)
            logger.info(
                "%s %s seed %d: CRPS %.4f (naive %.4f), best epoch %d",
                fold.name, name, seed, np.nanmean(crps), np.nanmean(naive_crps), best_epoch,
            )  # fmt: skip
            for asset, value in enumerate(crps):
                crps_rows.append({
                    "fold": fold.name, "set": name, "seed": seed, "asset": assets[asset],
                    "crps": value, "naive_crps": naive_crps[asset], "best_epoch": best_epoch,
                })  # fmt: skip
            if name != largest:
                continue
            median = pd.DataFrame(
                quantiles[:, :, len(levels) // 2],
                index=data.times[split.evaluation],
                columns=list(assets),
            )
            forecasts.append(median.where(mask).assign(seed=seed))
            for position, masked in enumerate(indices):
                blanked = [(x.clone(), m, t) for x, m, t in _select(evaluation_batches, indices)]
                for x, _, _ in blanked:
                    x[:, :, position] = 0.0
                masked_crps = evaluate_forecasts(
                    _predict(model, blanked, device), labels, mask, levels
                )
                for asset, value in enumerate(masked_crps):
                    masking_rows.append({
                        "fold": fold.name, "seed": seed, "masked": plan.timeframes[masked].value,
                        "asset": assets[asset], "crps": value, "full_crps": crps[asset],
                    })  # fmt: skip
    return {
        "crps": pd.DataFrame(crps_rows),
        "masking": pd.DataFrame(masking_rows),
        "forecasts": pd.concat(forecasts),
    }


def decision_frequency(
    forecasts: pd.DataFrame, hourly_returns: pd.DataFrame, costs_bps: pd.Series, *, every_hours: int
) -> pd.Series:
    """Return the hourly net returns of a book that trades only every `every_hours` hours.

    The book holds the sign of each asset's forecast, set at every `every_hours`-th decision
    and kept in between; it pays each asset's cost on every change of position. Its return is
    the equal-weight mean over assets.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Median forecasts per hour and asset (one seed).
    hourly_returns : pd.DataFrame
        Log return of each asset over the hour after each decision time, same shape.
    costs_bps : pd.Series
        Cost per side in bps per asset.
    every_hours : int
        Hours between trades (1, 4, 24).

    Returns
    -------
    pd.Series
        Net return of the book per hour.
    """
    decide = pd.Series(np.arange(len(forecasts)) % every_hours == 0, index=forecasts.index)
    target = np.sign(forecasts.fillna(0.0)).where(decide, np.nan, axis=0).ffill().fillna(0.0)
    turnover = target.diff().abs().fillna(target.abs())
    pnl = target * hourly_returns.fillna(0.0) - turnover * costs_bps[forecasts.columns] / 1e4
    return pnl.mean(axis=1)
