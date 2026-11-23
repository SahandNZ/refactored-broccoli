"""Tests of 12-horizon-spans (``lib/_spans.py``): the calendar span of a horizon."""

from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.benchmark import read_cache


def test_a_market_with_a_bar_every_hour_spans_h_hours(
    spans: ModuleType, quarters_cache_folder: Path
):
    # the synthetic cache has a bar every hour and label_time = time + h
    table = spans.horizon_spans(read_cache(quarters_cache_folder), windows=["t1"])
    assert set(table["market"]) == {"crypto", "forex", "stock"}
    for horizon, rows in table.groupby("horizon"):
        assert (rows["span_median_hours"] == float(horizon)).all()


def test_a_closure_inside_the_label_lengthens_its_span(
    spans: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    labels = cache.labels.copy()
    is_stock = (labels["asset"] == "AAPL") & (labels["horizon"] == 24)
    labels.loc[is_stock, "label_time"] += pd.Timedelta(hours=48)
    table = spans.horizon_spans(replace(cache, labels=labels), windows=["t1"])
    row = table[(table["market"] == "stock") & (table["horizon"] == 24)].iloc[0]
    assert row["span_median_hours"] == pytest.approx(72.0)


def test_no_label_in_the_windows_raises(spans: ModuleType, quarters_cache_folder: Path):
    with pytest.raises(ValueError, match="no label"):
        spans.horizon_spans(read_cache(quarters_cache_folder), windows=["l9"])
