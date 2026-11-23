"""Check that the stored bars are resampled correctly (run by hand after a data build).

1. Against an independent source: for stocks that have both 15m and 1h Dukascopy parts
   (2318.HK, 7974.T), the 1h bars built from 15m must equal the 1h bars Dukascopy serves; for
   EURUSD, XAUUSD and USA500, the 1h bars built from our 15m must equal a fresh download of
   Dukascopy's own 1h candles for 2025 (bid side).
2. On every built instrument: each coarser file equals a fresh resampling of the base file;
   OHLC are consistent (low <= open, close <= high); every bar opens on its grid; volumes add
   up from the base to the daily bars; each daily close is the last base close of the day.

Writes ``tests/smoke/out/resampling_check.json`` and prints a summary.

Usage: .venv/bin/python tests/smoke/check_resampling.py
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import LAB_DIR
from src.data import load_data_config
from src.data._dukascopy import CLI, build_bars
from src.data._read import bars_file
from src.data._resample import resample_bars
from src.enums import Market, Timeframe

OUT = LAB_DIR / "tests" / "smoke" / "out" / "resampling_check.json"
DATA = LAB_DIR / "data"
PRICES = ["bid_open", "bid_high", "bid_low", "bid_close"]


def compare(ours: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> dict:
    """Match bars by open time and return the share present in both and the largest gap."""
    both = ours.index.intersection(reference.index)
    gaps = (ours.loc[both, columns] / reference.loc[both, columns] - 1).abs()
    return {
        "bars_ours": len(ours), "bars_reference": len(reference), "bars_both": len(both),
        "only_ours": len(ours.index.difference(reference.index)),
        "only_reference": len(reference.index.difference(ours.index)),
        "max_rel_gap": float(gaps.to_numpy().max()) if len(both) else None,
        "share_exact": float((gaps < 1e-9).all(axis=1).mean()) if len(both) else None,
    }  # fmt: skip


def native_1h(code: str, folder: Path) -> pd.DataFrame:
    """Download Dukascopy's own 1h bid candles for 2025 into `folder`."""
    command = [
        str(CLI), "-i", code, "-from", "2025-01-01", "-to", "2026-01-01", "-t", "h1", "-p", "bid",
        "-v", "-vu", "units", "-f", "csv", "-s", "-r", "3", "-bs", "5", "-bp", "1000",
        "--rate-limit-retries", "8", "--request-interval", "250", "-fn", code, "-dir", str(folder),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603 (fixed command)
    frame = pd.read_csv(folder / f"{code}.csv")
    frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    return frame.add_prefix("bid_")


def independent_checks() -> dict:
    """Section 1: our 1h bars against Dukascopy's own 1h bars."""
    results = {}
    for symbol in ("2318.HK", "7974.T"):
        raw = DATA / "stock" / symbol / "raw" / "dukascopy"
        from_15m = resample_bars(build_bars(raw), Timeframe.H1)
        native = build_bars(raw / "1h")
        results[f"{symbol} 15m->1h vs native 1h"] = compare(from_15m, native, PRICES)
    with tempfile.TemporaryDirectory() as folder:
        cases = (
            ("forex", "EURUSD", "eurusd"),
            ("commodity", "XAUUSD", "xauusd"),
            ("index", "USA500", "usa500idxusd"),
        )
        for market, symbol, code in cases:
            ours = pd.read_parquet(bars_file(DATA, Market(market), symbol, Timeframe.H1))
            ours = ours[(ours.index >= "2025-01-01") & (ours.index < "2026-01-01")]
            reference = native_1h(code, Path(folder))
            label = f"{symbol} our 1h vs Dukascopy 1h (2025, bid)"
            results[label] = compare(ours, reference, PRICES)
            time.sleep(2)
    return results


def _check_timeframe(
    base: pd.DataFrame, stored: pd.DataFrame, timeframe, base_tf, label: str
) -> list[str]:
    """Problems of one stored timeframe against its base bars."""
    problems = []
    prefix = "mid_" if "mid_close" in base else ""
    fresh = base if timeframe is base_tf else resample_bars(base, timeframe)
    same_close = np.allclose(stored[f"{prefix}close"], fresh[f"{prefix}close"], equal_nan=True)
    if not stored.index.equals(fresh.index) or not same_close:
        problems.append(f"{label}: stored file differs from a fresh resampling")
    first, high, low, last = (stored[f"{prefix}{f}"] for f in ("open", "high", "low", "close"))
    tolerance = 1e-12
    if ((low > first + tolerance) | (low > last + tolerance) | (high < first - tolerance)
            | (high < last - tolerance)).any():  # fmt: skip
        problems.append(f"{label}: OHLC out of order")
    index = stored.index
    fixed_grid = timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.H12, Timeframe.D1)
    if fixed_grid and ((index.hour * 60 + index.minute) % timeframe.minutes).any():
        problems.append(f"{label}: bars off the {timeframe.value} grid")
    if timeframe is Timeframe.W1 and (index.dayofweek != 0).any():
        problems.append(f"{label}: weekly bars not on Mondays")
    return problems


def _check_daily(base: pd.DataFrame, daily: pd.DataFrame, name: str) -> list[str]:
    """Volumes add up and each daily close is the day's last base close."""
    problems = []
    prefix = "mid_" if "mid_close" in base else ""
    volume = next((c for c in base.columns if c.endswith("volume")), None)
    if volume is not None and not np.isclose(base[volume].sum(), daily[volume].sum(), rtol=1e-9):
        problems.append(f"{name}: daily volume does not add up")
    last = base[f"{prefix}close"].groupby(base.index.floor("1D")).last()
    if not np.allclose(last.reindex(daily.index), daily[f"{prefix}close"], equal_nan=True):
        problems.append(f"{name}: daily close is not the day's last base close")
    return problems


def internal_checks() -> dict:
    """Section 2: every built instrument is consistent with its own base bars."""
    config = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=DATA)
    problems, checked = [], 0
    for instrument in config.instruments:
        base_path = bars_file(DATA, instrument.market, instrument.name, instrument.timeframe)
        if not base_path.exists():
            continue
        base = pd.read_parquet(base_path)
        checked += 1
        for timeframe in Timeframe:
            if timeframe.minutes < instrument.timeframe.minutes:
                continue
            path = bars_file(DATA, instrument.market, instrument.name, timeframe)
            label = f"{instrument.name} {timeframe.value}"
            problems += _check_timeframe(
                base, pd.read_parquet(path), timeframe, instrument.timeframe, label
            )
        daily = pd.read_parquet(bars_file(DATA, instrument.market, instrument.name, Timeframe.D1))
        problems += _check_daily(base, daily, instrument.name)
    return {"instruments_checked": checked, "problems": problems}


def main() -> None:
    """Run both sections and save the report."""
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    report["independent"] = independent_checks()
    report["internal"] = internal_checks()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["independent"], indent=1))
    print("instruments checked:", report["internal"]["instruments_checked"])
    print("problems:", len(report["internal"]["problems"]), report["internal"]["problems"][:20])


if __name__ == "__main__":
    main()
