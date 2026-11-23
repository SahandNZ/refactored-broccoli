"""Smoke-test the panel pipeline on real data: one development fold, one seed, few epochs.

Not a study: it only checks that the panel trains end to end and beats the climatology
baseline. Writes tests/smoke/out/smoke/<run-id>/ (log, crps.csv, summary.json).

Usage: .venv/bin/python tests/smoke/smoke_panel.py --symbols BTCUSDT,ETHUSDT,... [--epochs 3]
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from src import LAB_DIR
from src.data import load_data_config
from src.enums import Timeframe
from src.evaluate import DEVELOPMENT_FOLDS
from src.panel import PanelData, PanelSettings, TimeframePlan, load_panel_bars, run_fold

logger = logging.getLogger("panel_smoke")


def main() -> None:
    """Load the chosen symbols, train the panel model on fold d4 and compare with climatology."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fold", default="d4")
    arguments = parser.parse_args()
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H%M")
    folder = LAB_DIR / "tests" / "smoke" / "out" / "smoke" / run_id
    folder.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(folder / "run.log"), logging.StreamHandler()],
    )  # fmt: skip
    config = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=LAB_DIR / "data")
    market_of = {instrument.name: instrument.market for instrument in config.instruments}
    markets = {symbol: market_of[symbol] for symbol in arguments.symbols.split(",")}
    timeframes = (Timeframe.H1, Timeframe.H4, Timeframe.D1)
    fold = next(f for f in DEVELOPMENT_FOLDS if f.name == arguments.fold)
    started = time.monotonic()
    bars = load_panel_bars(LAB_DIR / "data", markets, timeframes)
    settings = PanelSettings(lookback=64, horizon_hours=4, span_bars=96, max_age_hours=96)
    start = fold.evaluation_start - pd.Timedelta(days=760)
    data = PanelData.build(
        bars, start=str(start.date()), end=str(fold.evaluation_end.date()), settings=settings
    )
    options = {
        "levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "train_days": 730, "validation_days": 30, "horizon_hours": 4, "train_every_hours": 4,
        "batch_size": 64, "d_model": 64, "learning_rate": 1e-3, "weight_decay": 0.01,
        "max_epochs": arguments.epochs, "patience": 3, "clip": 1.0,
    }  # fmt: skip
    plan = TimeframePlan(timeframes, {"S3": list(timeframes)}, (0,), options)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_fold(data, fold, plan, device=device)
    crps = result["crps"]
    crps.to_csv(folder / "crps.csv", index=False)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    summary = {
        "fold": fold.name, "assets": len(markets), "epochs": arguments.epochs, "device": device,
        "panel_crps": float(np.nanmean(crps["crps"])),
        "climatology_crps": float(np.nanmean(crps["naive_crps"])),
        "assets_better": int((crps["crps"] < crps["naive_crps"]).sum()),
        "minutes": round((time.monotonic() - started) / 60, 1), "peak_gpu_gb": round(peak, 2),
    }  # fmt: skip
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("%s", summary)


if __name__ == "__main__":
    main()
