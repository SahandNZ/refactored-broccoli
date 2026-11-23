import json
from pathlib import Path

import pandas as pd

from src.data._config import Instrument
from src.enums import Market
from src.utils import active_days, expected_days


def describe_bars(bars: pd.DataFrame, instrument: Instrument) -> dict[str, object]:
    """Summarise the quality of one instrument's 15m bars for ``data/manifest.json``.

    Parameters
    ----------
    bars : pd.DataFrame
        15m bars with a UTC index of open times.
    instrument : Instrument
        The instrument they belong to.

    Returns
    -------
    dict
        source, market, symbol, code, first_bar, last_bar, n_bars, day_coverage (share of
        expected trading days, Monday to Friday outside crypto, with at least one bar),
        max_gap_days (longest time between two bars), median_spread_bps (bid/ask sources) and
        median_daily_amount (quote volume per day, Binance).
    """
    first, last = bars.index[0], bars.index[-1]
    expected = expected_days(first, last, weekdays_only=instrument.market is not Market.CRYPTO)
    present = active_days(bars.index)
    gaps = bars.index.to_series().diff().dt.total_seconds() / 86400
    row: dict[str, object] = {
        "source": instrument.source.value,
        "market": instrument.market.value,
        "symbol": instrument.name,
        "code": instrument.code,
        "first_bar": first,
        "last_bar": last,
        "n_bars": len(bars),
        "day_coverage": round(len(present.intersection(expected)) / len(expected), 4),
        "max_gap_days": round(float(gaps.max()), 2),
        "median_spread_bps": None,
        "median_daily_amount": None,
    }
    if "spread" in bars.columns:
        row["median_spread_bps"] = round(
            float((bars["spread"] / bars["mid_close"]).median() * 1e4), 2
        )
    if "amount" in bars.columns:
        row["median_daily_amount"] = float(bars["amount"].resample("1D").sum().median())
    return row


def write_manifest(path: Path, manifest: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Write ``data/manifest.json``: one entry per instrument with its splits.

    Parameters
    ----------
    path : Path
        The file to write.
    manifest : pd.DataFrame
        One row per instrument (`describe_bars` and the quality checks).
    splits : pd.DataFrame
        Every split applied: symbol, date, ratio, source.
    """
    entries = []
    for row in manifest.to_dict(orient="records"):
        mine = splits[splits["symbol"] == row["symbol"]]
        row["splits"] = [
            {"date": str(date), "ratio": float(ratio), "source": source}
            for date, ratio, source in zip(mine["date"], mine["ratio"], mine["source"], strict=True)
        ]
        entry: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, list) or not pd.isna(value):
                entry[str(key)] = value
            else:
                entry[str(key)] = None
        entries.append(entry)
    text = json.dumps({"instruments": entries}, indent=1, default=str)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)
