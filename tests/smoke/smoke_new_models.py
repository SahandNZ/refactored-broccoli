"""Smoke-test every zero-shot model on a few real contexts (local GPU).

For each model: load it, forecast 24 hourly steps from 512 hourly closes of BTCUSDT, EURUSD and
AAPL (8 contexts each), and check shapes, finiteness, ordered quantiles and that the median
stays near the last price. Prints load and forecast time and peak GPU memory.

Usage: .venv/bin/python tests/smoke/smoke_new_models.py [KIND:NAME ...]
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from src import LAB_DIR

OUT = LAB_DIR / "tests" / "smoke" / "out" / "smoke_new_models.jsonl"

os.environ.setdefault("HF_HOME", str(LAB_DIR / ".hf"))

from src.models import build_forecaster  # noqa: E402 (after HF_HOME is set)

MODELS = [
    ("chronos", "amazon/chronos-bolt-base"),
    ("chronos", "amazon/chronos-2"),
    ("timesfm", "google/timesfm-2.5-200m-pytorch"),
    ("timesfm3", "google/timesfm-3.0-pytorch"),
    ("tirex", "NX-AI/TiRex"),
    ("tirex2", "NX-AI/TiRex-2"),
    ("toto2", "Datadog/Toto-2.0-22m"),
    ("toto2", "Datadog/Toto-2.0-313m"),
    ("t0", "theforecastingcompany/t0-gamma"),
    ("moirai2", "Salesforce/moirai-2.0-R-small"),
]
ASSETS = [
    ("crypto", "BTCUSDT", "close"),
    ("forex", "EURUSD", "mid_close"),
    ("stock", "AAPL", "close"),
]


def contexts() -> np.ndarray:
    """Return 8 recent 512-bar hourly contexts per asset."""
    rows = []
    for market, symbol, column in ASSETS:
        closes = (
            pd.read_parquet(LAB_DIR / "data" / market / symbol / "ohlcv" / "1h.pq")[column]
            .dropna()
            .to_numpy()
        )
        for end in range(len(closes) - 8 * 300, len(closes), 300):
            rows.append(closes[end - 512 : end])
    return np.stack(rows).astype(np.float64)


def main() -> None:
    """Load and check each model in turn."""
    chosen = [tuple(arg.split(":", 1)) for arg in sys.argv[1:]] or MODELS
    ctx = contexts()
    for kind, name in chosen:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            start = time.time()
            kwargs = (
                {} if kind == "timesfm" and name.endswith("2.5-200m-pytorch") else {"name": name}
            )
            model = build_forecaster(kind, **kwargs)
            loaded = time.time() - start
            start = time.time()
            forecast = model.predict(ctx, 24)
            spent = time.time() - start
            q = forecast.quantiles
            ordered = bool(np.all(np.diff(q, axis=-1) >= -1e-6 * np.abs(q[..., :1])))
            median_gap = float(np.nanmax(np.abs(q[:, 0, 4] / ctx[:, -1] - 1)))
            status = (
                "ok"
                if q.shape == (len(ctx), 24, 9)
                and np.isfinite(q).all()
                and ordered
                and median_gap < 0.05
                else "CHECK"
            )
            peak = torch.cuda.max_memory_allocated() / 2**30
            OUT.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind,
                "name": name, "status": status, "shape": list(q.shape), "ordered": ordered,
                "max_median_gap": round(median_gap, 5), "load_s": round(loaded, 1),
                "predict_s": round(spent, 2), "peak_gb": round(peak, 2),
            }  # fmt: skip
            with OUT.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(
                f"{status:5} {kind:8} {name:36} shape {q.shape} ordered {ordered} "
                f"max |median/last-1| {median_gap:.4f} load {loaded:.0f}s "
                f"predict {spent:.1f}s peak {peak:.1f} GB",
                flush=True,
            )
            del model
        except Exception as error:  # noqa: BLE001 (report every model and keep going)
            print(
                f"FAIL  {kind:8} {name:36} {type(error).__name__}: {str(error)[:300]}", flush=True
            )


if __name__ == "__main__":
    main()
