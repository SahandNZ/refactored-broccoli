"""rq4 analyze.py: the sentences of the answer built from small verdict tables."""

import runpy
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import SCORED

if TYPE_CHECKING:
    from collections.abc import Callable

ANALYZE = LAB_DIR / "experiments" / "06-rq4-leakage" / "analyze.py"

# analyze.py imports the study's lib as `lib`, as when it runs from its own folder
pytestmark = pytest.mark.usefixtures("study_lib_as_lib")


@pytest.fixture(name="analyze")
def _analyze() -> dict[str, Any]:
    return runpy.run_path(str(ANALYZE), run_name="analyze")


def _cell(**values: object) -> dict[str, object]:
    cell: dict[str, object] = {
        "test": "leakage", "model": "chronos-2", "market": "stock", "horizon": 24,
        "status": SCORED, "second_times": 1000, "did": 0.001, "low": -0.001, "high": 0.003,
        "adjusted_p": 1.0, "rejects": False, "unpaired": 0,
    }  # fmt: skip
    return cell | values


def test_the_size_note_counts_only_the_hourly_models_24_bar_cells(
    analyze: dict[str, Any],
) -> None:
    size_note: Callable[[pd.DataFrame, set[str]], str] = analyze["_size_note"]
    primary = pd.DataFrame([
        _cell(second_times=302),
        _cell(model="t0-gamma", second_times=2170),
        _cell(model="kronos-small", second_times=52),  # its own grid: labels do not overlap
        _cell(horizon=1, second_times=10),  # 1 bar: not overlapping
        _cell(model="t0-gamma", status="not scored: too few blocks", second_times=5),
    ])  # fmt: skip
    note = size_note(primary, {"chronos-2", "t0-gamma"})
    assert "302-2170 times after the release, the shortest below the 400" in note
    assert "below" not in size_note(primary[primary["model"] == "t0-gamma"], {"t0-gamma"})
    assert "this run" not in size_note(primary, set())


def test_a_check_lists_the_cells_it_rejects(analyze: dict[str, Any]) -> None:
    checks: Callable[[pd.DataFrame], str] = analyze["_checks"]
    verdict = pd.DataFrame([
        _cell(),
        _cell(test="hourly", horizon=1, did=0.036, adjusted_p=3e-34, rejects=True),
        _cell(test="hourly", model="t0-gamma", horizon=1),
        _cell(test="placebo", model="chronos-bolt-small"),
    ])  # fmt: skip
    text = checks(verdict)
    assert "Holm rejects 1 of 2 scored cells: chronos-2 stock 1h (did 0.0360" in text
    assert "Holm rejects 0 of 1 scored cells." in text


def test_the_left_out_note_names_fits_and_unpaired_cells(analyze: dict[str, Any]) -> None:
    left_out: Callable[[pd.DataFrame, pd.DataFrame], str] = analyze["_left_out"]
    fits = pd.DataFrame(
        {"model": ["garch-normal"], "window": ["l4"], "asset": ["NFLX"], "convergence_flag": [8]}
    )
    primary = pd.DataFrame([
        _cell(unpaired=153),
        _cell(model="t0-gamma", unpaired=40),
        _cell(model="t0-gamma", market="crypto", status="not scored: too few blocks", unpaired=9),
    ])  # fmt: skip
    text = left_out(fits, primary)
    assert "(02-forecasts not_fitted.csv): garch-normal l4 NFLX;" in text
    assert text.endswith("stock 24h (at most 153 decisions of one model)")
    empty = left_out(fits.iloc[:0], primary.assign(unpaired=0))
    assert "not_fitted.csv): none;" in empty
    assert empty.endswith("(scored cells of the test leakage): none")


def test_the_cells_not_scored_are_named(analyze: dict[str, Any]) -> None:
    unscored_cells: Callable[[pd.DataFrame], str] = analyze["_unscored_cells"]
    primary = pd.DataFrame([
        _cell(),
        _cell(model="t0-gamma", status="not scored: too few blocks"),
        _cell(model="kronos-small", market="index", horizon=4, status="not scored: too few"),
    ])  # fmt: skip
    text = unscored_cells(primary)
    assert text == " The cells not scored: t0-gamma stock 24h, kronos-small index 4h."
    assert unscored_cells(primary.iloc[:1]) == ""
