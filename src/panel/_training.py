import copy
import logging
from collections.abc import Iterable

import torch
from torch import nn

logger = logging.getLogger(__name__)

type Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""``(x, mask, target)``: windows, valid-asset mask and labels ``(batch, n_assets)``."""


def quantile_loss(
    forecast: torch.Tensor, target: torch.Tensor, levels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean pinball loss over valid assets and every quantile level.

    Parameters
    ----------
    forecast : torch.Tensor
        Quantile forecasts, shape ``(batch, n_assets, n_levels)``.
    target : torch.Tensor
        Labels, shape ``(batch, n_assets)``; NaN where unknown.
    levels : torch.Tensor
        Quantile levels in (0, 1), shape ``(n_levels,)``.
    mask : torch.Tensor
        Valid assets, shape ``(batch, n_assets)``.

    Returns
    -------
    torch.Tensor
        A scalar; zero (with a gradient) when nothing is valid.

    References
    ----------
    R. Koenker, G. Bassett, "Regression Quantiles", Econometrica 46(1), 1978.
    """
    valid = mask & torch.isfinite(target)
    error = torch.nan_to_num(target)[..., None] - forecast
    pinball = torch.maximum(levels * error, (levels - 1) * error).mean(dim=-1)
    if not valid.any():
        return forecast.sum() * 0
    return pinball[valid].mean()


def _to(batch: Batch, device: str) -> Batch:
    x, mask, target = batch
    return x.to(device), mask.to(device), target.to(device)


def _mean_loss(
    model: nn.Module, batches: Iterable[Batch], levels: torch.Tensor, device: str
) -> float:
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            x, mask, target = _to(batch, device)
            total += float(quantile_loss(model(x, mask), target, levels, mask)) * len(x)
            count += len(x)
    return total / max(count, 1)


def train(
    model: nn.Module,
    train_batches: Iterable[Batch],
    validation_batches: Iterable[Batch],
    *,
    levels: torch.Tensor,
    settings: dict[str, float],
    device: str,
) -> dict[str, list[float]]:
    """Train with AdamW and early stopping on the validation quantile loss.

    The weights of the best validation epoch are restored at the end.

    Parameters
    ----------
    model : nn.Module
        A `PanelModel` (or any model ``(x, mask) -> (batch, n_assets, n_levels)``).
    train_batches, validation_batches : Iterable of Batch
        Re-iterable batches (e.g. lists or a ``DataLoader``).
    levels : torch.Tensor
        Quantile levels, shape ``(n_levels,)``.
    settings : dict
        ``learning_rate``, ``weight_decay``, ``max_epochs``, ``patience`` and ``clip`` (from
        the study's configuration; there are no defaults). The caller seeds the run
        (`src.experiment.seed_everything`) before it builds the model.
    device : str
        Torch device.

    Returns
    -------
    dict of str to list of float
        ``train_loss`` and ``validation_loss`` per epoch, and ``best_epoch`` (one value).
    """
    model.to(device)
    levels = levels.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": [], "best_epoch": [0.0]}
    best, best_state, waited = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        total, count = 0.0, 0
        for batch in train_batches:
            x, mask, target = _to(batch, device)
            loss = quantile_loss(model(x, mask), target, levels, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["clip"])
            optimizer.step()
            total, count = total + float(loss.detach()) * len(x), count + len(x)
        validation = _mean_loss(model, validation_batches, levels, device)
        history["train_loss"].append(total / max(count, 1))
        history["validation_loss"].append(validation)
        logger.info(
            "epoch %d: train %.5f, validation %.5f", epoch, history["train_loss"][-1], validation
        )
        if validation < best:
            best, best_state, waited = validation, copy.deepcopy(model.state_dict()), 0
            history["best_epoch"] = [float(epoch)]
        else:
            waited += 1
            if waited >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return history
