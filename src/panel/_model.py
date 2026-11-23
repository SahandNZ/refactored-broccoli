from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn


class Mixing(Enum):
    """Which attention stages of the panel model run (the ablations of the paper)."""

    NONE = "none"
    TIMEFRAME = "timeframe"
    ASSET = "asset"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class PanelConfig:
    """Sizes and ablation switch of a `PanelModel`.

    Attributes
    ----------
    d_model : int
        Vector size (divisible by `n_heads`).
    n_timeframes : int
        Timeframes per asset.
    n_quantiles : int
        Quantile levels forecast per asset.
    mixing : Mixing
        Which attention stages run.
    n_heads, num_layers : int
        Heads and layers of each attention stage.
    dropout : float
        Dropout inside the attention stages.
    """

    d_model: int
    n_timeframes: int
    n_quantiles: int
    mixing: Mixing
    n_heads: int
    num_layers: int
    dropout: float


class PanelModel(nn.Module):
    """Forecast every asset from all assets and timeframes at once.

    1. A shared window encoder turns each (asset, timeframe) window into a vector.
    2. Each vector gets a learned embedding of its timeframe; self-attention over the
       timeframes of each asset mixes them, and their mean is the asset's vector.
    3. Self-attention over the assets (masked assets left out) lets every asset see the others.
    4. A linear head gives quantile forecasts per asset.

    ``config.mixing`` drops stage 2, 3 or both for ablations; with ``Mixing.NONE`` each asset is
    forecast from its own timeframes only (mean of their vectors).

    Parameters
    ----------
    encoder : nn.Module
        Maps windows ``(n, lookback, n_features)`` to vectors ``(n, config.d_model)``.
    config : PanelConfig
        Sizes and the ablation switch.

    References
    ----------
    A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017; J. Lee et al., "Set
    Transformer", ICML 2019 (attention over sets of assets).
    """

    def __init__(self, encoder: nn.Module, config: PanelConfig) -> None:
        super().__init__()
        if not isinstance(config.mixing, Mixing):
            raise TypeError(f"mixing must be a Mixing, got {config.mixing!r}")
        self.encoder = encoder
        self.config = config
        size = config.d_model
        self.timeframe_embedding = nn.Parameter(torch.zeros(config.n_timeframes, size))
        mixing = config.mixing
        uses_timeframes = mixing in (Mixing.TIMEFRAME, Mixing.BOTH)
        uses_assets = mixing in (Mixing.ASSET, Mixing.BOTH)
        self.timeframe_attention = self._stage(config) if uses_timeframes else None
        self.asset_attention = self._stage(config) if uses_assets else None
        self.head = nn.Linear(size, config.n_quantiles)

    @staticmethod
    def _stage(config: PanelConfig) -> nn.TransformerEncoder:
        size = config.d_model
        layer = nn.TransformerEncoderLayer(
            size, config.n_heads, 4 * size, config.dropout, batch_first=True
        )
        return nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forecast quantiles of every asset.

        Parameters
        ----------
        x : torch.Tensor
            Windows, shape ``(batch, n_assets, n_timeframes, lookback, n_features)``.
        mask : torch.Tensor
            True for assets with a valid window, shape ``(batch, n_assets)``.

        Returns
        -------
        torch.Tensor
            Quantile forecasts, shape ``(batch, n_assets, n_quantiles)``; rows of masked assets
            are not meaningful.

        Raises
        ------
        ValueError
            If `x` does not have 5 dimensions or does not match `mask`.
        """
        if x.dim() != 5 or x.shape[:2] != mask.shape:
            expected = "(batch, assets, timeframes, lookback, features) matching mask"
            raise ValueError(f"x must be {expected} {tuple(mask.shape)}, got {tuple(x.shape)}")
        batch, n_assets, n_timeframes, lookback, n_features = x.shape
        flat = x.reshape(-1, lookback, n_features)
        vectors = self.encoder(flat).reshape(batch * n_assets, n_timeframes, -1)
        vectors = vectors + self.timeframe_embedding
        if self.timeframe_attention is not None:
            vectors = self.timeframe_attention(vectors)
        assets = vectors.mean(dim=1).reshape(batch, n_assets, -1)
        if self.asset_attention is not None:
            assets = self.asset_attention(assets, src_key_padding_mask=~mask)
        return self.head(assets)
