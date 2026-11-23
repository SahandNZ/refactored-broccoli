"""Run 01-rq1-smoke: every zero-shot model on 24 real hourly contexts (Q1).

Usage: .venv/bin/python experiments/00-template/01-rq1-smoke/run.py [--run-id ID] [--dry-run]
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import LAB_DIR
from src.experiment import Placement, parse_invocation, start_run
from src.models import build_forecaster

logger = logging.getLogger("01-rq1-smoke")


def contexts(options: dict) -> np.ndarray:
    """Cut the configured number of recent contexts from each asset's 1h closes."""
    rows = []
    lookback, count = int(options["lookback"]), int(options["contexts_per_asset"])
    for entry in options["assets"]:
        market, symbol, column = entry.split(":")
        path = LAB_DIR / "data" / market / symbol / "ohlcv" / "1h.pq"
        closes = pd.read_parquet(path)[column].dropna().to_numpy()
        for end in range(len(closes) - count * 300, len(closes), 300):
            rows.append(closes[end - lookback : end])
    return np.stack(rows).astype(np.float64)


def check(kind: str, name: str, context: np.ndarray, options: dict) -> dict:
    """Load one model, forecast, and return its checks and timings."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    model = build_forecaster(kind, name=name)
    loaded = time.time() - start
    start = time.time()
    quantiles = model.predict(context, int(options["horizon"])).quantiles
    spent = time.time() - start
    shape = quantiles.shape == (len(context), int(options["horizon"]), 9)
    finite = bool(np.isfinite(quantiles).all())
    ordered = bool(np.all(np.diff(quantiles, axis=-1) >= -1e-6 * np.abs(quantiles[..., :1])))
    gap = float(np.max(np.abs(quantiles[:, 0, 4] / context[:, -1] - 1)))
    del model
    return {
        "model": f"{kind}:{name}", "shape_ok": shape, "finite": finite, "ordered": ordered,
        "max_median_gap": round(gap, 5), "median_ok": gap < float(options["max_median_gap"]),
        "passed": shape and finite and ordered and gap < float(options["max_median_gap"]),
        "load_seconds": round(loaded, 1), "predict_seconds": round(spent, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }  # fmt: skip


def main() -> None:
    """Check every model of the configuration and save one table row per model."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    if invocation.dry_run:
        logger.info("dry run: %d models", len(config.models))
        return
    run = start_run(
        invocation.study_dir, invocation.run_id, config, data_dir=LAB_DIR / "data",
        placement=Placement("local", None, None, 1),
    )  # fmt: skip
    try:
        context = contexts(config.options)
        rows = []
        for entry in config.models:
            kind, name = entry.split(":", 1)
            rows.append(check(kind, name, context, config.options))
            logger.info("%s", rows[-1])
        table = pd.DataFrame(rows)
    except KeyboardInterrupt:
        run.finish(None, status="stopped", reason="interrupted or over twice the estimated time")
        raise
    except Exception as error:
        run.finish(None, status="failed", reason=repr(error))
        raise
    table.to_csv(run.folder / "tables" / "checks.csv", index=False)
    run.finish(table[["model", "passed", "predict_seconds", "peak_gpu_gb"]])


if __name__ == "__main__":
    main()
