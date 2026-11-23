"""Smoke-test every foundation model on 64 series of 15m closes.

Loads each model, forecasts, and logs shapes, runtime and peak GPU memory. A model that reads
past covariates (`src.models.takes_covariates`) also forecasts with one covariate and the
target's last ``CLOSED_STEPS`` steps set to NaN, like a market closed over a weekend (07, 08).

The 64 series are 32 windows each of BTC and ETH (lookback 512, horizon 9) ending at random
15m bars of 2026. Each model runs in its own process, one after another, so only one model is on
the GPU at a time and its peak memory is its own.

Usage: .venv/bin/python tests/smoke/smoke_models.py [model ...]
"""

import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from src import LAB_DIR
from src.data import read_bars
from src.enums import Market, Timeframe
from src.models import Forecaster, KronosSettings, build_forecaster, takes_covariates

logger = logging.getLogger("smoke_test")

KRONOS = {
    "settings": KronosSettings(
        tokenizer="NeoQuasar/Kronos-Tokenizer-base",
        n_samples=20,
        temperature=1.0,
        top_p=0.9,
        step=Timeframe.M15.timedelta,
    ),
    "device": "cuda",
}
MODELS = {
    "chronos-bolt-small": ("chronos", {"name": "amazon/chronos-bolt-small"}),
    "chronos-bolt-base": ("chronos", {"name": "amazon/chronos-bolt-base"}),
    "chronos-2": ("chronos", {"name": "amazon/chronos-2"}),
    "timesfm-2.5": ("timesfm", {}),
    "timesfm-3": ("timesfm3", {"name": "google/timesfm-3.0-pytorch"}),
    "tirex-2": ("tirex2", {"name": "NX-AI/TiRex-2"}),
    "t0-gamma": ("t0", {"name": "theforecastingcompany/t0-gamma"}),
    "kronos-small": (
        "kronos",
        {"name": "NeoQuasar/Kronos-small", "settings": KRONOS, "device": "cuda"},
    ),
    "kronos-base": (
        "kronos",
        {"name": "NeoQuasar/Kronos-base", "settings": KRONOS, "device": "cuda"},
    ),
}
LOOKBACK, HORIZON, WINDOWS_PER_COIN = 512, 9, 32
CLOSED_STEPS = 48
OUT = LAB_DIR / "tests" / "smoke" / "out" / "smoke_test.jsonl"


def _synthetic_bars(seed: int) -> pd.DataFrame:
    """Return a random-walk 15m candle series, for machines without the data (a cloud GPU)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=4 * 96 * 30, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(index))))
    spread = np.abs(rng.normal(0, 0.001, len(index))) * close
    volume = rng.uniform(1, 10, len(index))
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close,
         "volume": volume, "amount": volume * close},
        index=index,
    )  # fmt: skip


def windows() -> tuple[np.ndarray, np.ndarray, list]:
    """Return contexts ``(64, LOOKBACK)``, targets ``(64, HORIZON)`` and the candle windows.

    BTC and ETH 15m closes from ``data/``, or random walks when ``SMOKE_SYNTHETIC=1``.
    """
    rng = np.random.default_rng(0)
    synthetic = os.environ.get("SMOKE_SYNTHETIC") == "1"
    contexts, targets, candles = [], [], []
    for number, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        if synthetic:
            bars = _synthetic_bars(number)
        else:
            bars = read_bars(LAB_DIR / "data", Market.CRYPTO, symbol, Timeframe.M15)
            bars = bars.loc["2026-01-01":]
        ends = rng.choice(np.arange(LOOKBACK, len(bars) - HORIZON), WINDOWS_PER_COIN, replace=False)
        for end in ends:
            contexts.append(bars["close"].to_numpy()[end - LOOKBACK : end])
            targets.append(bars["close"].to_numpy()[end : end + HORIZON])
            candles.append(bars.iloc[end - LOOKBACK : end])
    return np.stack(contexts), np.stack(targets), candles


def _covariate_result(model: Forecaster, context: np.ndarray) -> dict:
    """Forecast with a NaN tail and one past covariate (the log return of the context)."""
    closed = context.copy()
    closed[:, -CLOSED_STEPS:] = np.nan
    covariates = np.log(context / context[:, :1])[:, None, :]
    started = time.perf_counter()
    forecast = model.predict_with_covariates(closed, covariates, HORIZON)
    return {
        "covariates_seconds_per_64_series": round(time.perf_counter() - started, 2),
        "covariates_quantile_shape": list(forecast.quantiles.shape),
        "covariates_finite": bool(np.isfinite(forecast.quantiles).all()),
    }


def run_one(label: str) -> dict:
    """Load one model, forecast twice (warm-up, timed) and return the measurements."""
    kind, kwargs = MODELS[label]
    context, target, candles = windows()
    started = time.perf_counter()
    model = build_forecaster(kind, **kwargs)
    load_seconds = time.perf_counter() - started
    model.predict(context[:2], HORIZON)  # warm-up (compilation, CUDA kernels)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    forecast = model.predict(context, HORIZON)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "model": label,
        "point_shape": list(forecast.point.shape),
        "quantile_shape": list(forecast.quantiles.shape),
        "load_seconds": round(load_seconds, 1),
        "seconds_per_64_series": round(seconds, 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "finite": bool(np.isfinite(forecast.point).all() and np.isfinite(forecast.quantiles).all()),
    }
    # sanity: error of the 9-bar-ahead point forecast against the last close (naive)
    naive = np.abs(target[:, -1] - context[:, -1]) / context[:, -1]
    error = np.abs(target[:, -1] - forecast.point[:, -1]) / context[:, -1]
    result["mae_bps_h9"] = round(float(error.mean() * 1e4), 1)
    result["naive_mae_bps_h9"] = round(float(naive.mean() * 1e4), 1)
    if takes_covariates(kind, str(kwargs.get("name", ""))):
        result |= _covariate_result(model, context)
    if kind == "kronos":
        started = time.perf_counter()
        ohlcv = model.predict_candles(candles, HORIZON)
        result["candles_seconds_per_64_series"] = round(time.perf_counter() - started, 2)
        error = np.abs(target[:, -1] - ohlcv.point[:, -1]) / context[:, -1]
        result["candles_mae_bps_h9"] = round(float(error.mean() * 1e4), 1)
    return result


def main() -> None:
    """Run the models given on the command line (default: all), each in a fresh process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    labels = sys.argv[1:] or list(MODELS)
    if len(labels) == 1:
        result = run_one(labels[0])
        logger.info("%s", result)
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("a") as file:
            file.write(json.dumps(result) + "\n")
        return
    for label in labels:
        done = subprocess.run([sys.executable, __file__, label], check=False)
        if done.returncode != 0:
            logger.error("%s failed with exit code %d", label, done.returncode)


if __name__ == "__main__":
    main()
