import torch


def portfolio_return_loss(
    score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Negative return of a dollar-neutral book built from the scores (an optional loss).

    At each time the scores of the valid assets become weights: demeaned and divided by their
    absolute sum (long and short halves of 0.5). The labels are z-scored over the same assets and
    the loss is ``-mean over times of sum(weights * z)``: how well the book is ordered, whatever
    the labels' scale. Times with fewer than two valid assets are skipped. The default loss of
    The panel model stays the quantile loss.

    Parameters
    ----------
    score : torch.Tensor
        One score per asset (e.g. the median quantile), shape ``(batch, n_assets)``.
    target : torch.Tensor
        Labels, shape ``(batch, n_assets)``; NaN where unknown.
    mask : torch.Tensor
        Valid assets, shape ``(batch, n_assets)``.

    Returns
    -------
    torch.Tensor
        A scalar.

    References
    ----------
    Z. Zhang, S. Zohren, S. Roberts, "Deep Learning for Portfolio Optimization", Journal of
    Financial Data Science 2(4), 2020.
    """
    valid = mask & torch.isfinite(target) & torch.isfinite(score)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    score = torch.where(valid, score, 0)
    target = torch.where(valid, target, 0)
    centred = torch.where(valid, score - score.sum(dim=1, keepdim=True) / count, 0)
    weights = centred / centred.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean = target.sum(dim=1, keepdim=True) / count
    spread = (torch.where(valid, target - mean, 0) ** 2).sum(dim=1, keepdim=True) / count
    zscore = torch.where(valid, (target - mean) / spread.sqrt().clamp_min(1e-12), 0)
    book = (weights * zscore).sum(dim=1)
    used = valid.sum(dim=1) >= 2
    return -book[used].mean() if used.any() else score.sum() * 0
