"""Run Moirai 2.0 in its own environment (``.venv-moirai``: uni2ts pins torch<2.5).

Reads contexts from an .npz file (``context``, shape (n, lookback)), writes the 9 deciles
0.1 .. 0.9 to another .npz (``quantiles``, shape (n, horizon, 9)). Called by
``src.models._new_models.Moirai2Forecaster``; it imports nothing from src.

Usage: .venv-moirai/bin/python scripts/moirai_worker.py IN.npz OUT.npz HORIZON [NAME] [BATCH]
"""

import sys

import numpy as np
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    """Load Moirai 2.0, forecast the contexts and save the deciles."""
    source, target, horizon = sys.argv[1], sys.argv[2], int(sys.argv[3])
    name = sys.argv[4] if len(sys.argv) > 4 else "Salesforce/moirai-2.0-R-small"
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    context = np.load(source)["context"].astype(np.float32)
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(name),
        prediction_length=horizon,
        context_length=context.shape[1],
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    levels = [round(float(q), 4) for q in model.module.quantile_levels]
    index = [levels.index(q) for q in LEVELS]
    parts = []
    for start in range(0, len(context), batch):
        rows = [row[:, None] for row in context[start : start + batch]]
        with torch.no_grad():
            out = np.asarray(model.predict(rows))  # (batch, num_quantiles, horizon)
        parts.append(np.moveaxis(out[:, index, :], 1, 2))
    np.savez(target, quantiles=np.concatenate(parts))


if __name__ == "__main__":
    main()
