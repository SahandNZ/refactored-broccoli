import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalisation: each window is scaled by its own mean and spread.

    Parameters
    ----------
    n_features : int
        Features per step.
    eps : float, optional
        Floor of the standard deviation.

    References
    ----------
    T. Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift", ICLR 2022.
    """

    def __init__(self, n_features: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise windows ``(batch, lookback, n_features)`` over the lookback axis."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std * self.weight + self.bias


class LSTMEncoder(nn.Module):
    """Encode a window with RevIN and an LSTM; the last hidden state is the window's vector.

    Parameters
    ----------
    n_features : int
        Features per step.
    d_model : int
        Size of the output vector.
    num_layers : int, optional
        Stacked LSTM layers.
    dropout : float, optional
        Dropout between layers.

    References
    ----------
    S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory", Neural Computation 9(8), 1997.
    """

    def __init__(
        self, n_features: int, d_model: int, *, num_layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = RevIN(n_features)
        self.lstm = nn.LSTM(n_features, d_model, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, lookback, n_features)`` to ``(batch, d_model)``."""
        _, (hidden, _) = self.lstm(self.norm(x))
        return hidden[-1]


class TransformerEncoder(nn.Module):
    """Encode a window with RevIN, a linear embedding, learned positions and self-attention.

    The mean over the steps is the window's vector.

    Parameters
    ----------
    n_features : int
        Features per step.
    d_model : int
        Size of the output vector (divisible by `n_heads`).
    lookback : int
        Steps per window.
    n_heads, num_layers : int, optional
        Attention heads and layers.
    dropout : float, optional
        Dropout inside the layers.

    References
    ----------
    A. Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    """

    def __init__(
        self, n_features: int, d_model: int, lookback: int, *, n_heads: int = 4,
        num_layers: int = 2, dropout: float = 0.1,
    ) -> None:  # fmt: skip
        super().__init__()
        self.norm = RevIN(n_features)
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, lookback, n_features)`` to ``(batch, d_model)``."""
        hidden = self.layers(self.embed(self.norm(x)) + self.position)
        return hidden.mean(dim=1)
