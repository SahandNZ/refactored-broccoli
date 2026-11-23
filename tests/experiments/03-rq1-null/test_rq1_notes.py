"""The notes of 03-rq1-null's answers (its lib's `size_note`, `null_note`, `whole_quarter_note`)."""

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import null_rule
from src.experiment import load_study
from src.stats import BLOCK_OVERLAP_FLOOR

RULE = null_rule(load_study(LAB_DIR / "experiments/03-rq1-null/study.toml"))
"""The rule of rq1 (min 3 quarters, X = 2 %, K = 80 %)."""


def _nine(paper: Path, runs: str, floors: tuple[int, ...]) -> Path:
    # 09-mde-threshold's results: a README naming `runs` and one size table per run
    results = paper / "09-mde-threshold" / "results"
    (results / "a" / "tables").mkdir(parents=True)
    (results / "README.md").write_text(f"# Results\n\nRuns: {runs}.\n")
    size = pd.DataFrame(
        {
            "length": [600, 2400][: len(floors)],
            "floor": list(floors),
            "mean_test_size": [0.07, 0.05][: len(floors)],
            "difference_test_size": [0.06, 0.065][: len(floors)],
            "draws": 300,
            "nominal": 0.05,
        }
    )
    size.to_csv(results / "a" / "tables" / "bootstrap_size.csv", index=False)
    return paper / "03-rq1-null"


def test_the_size_note_reads_the_one_named_09_run_at_the_floor(
    rq1: ModuleType, tmp_path: Path
) -> None:
    floors = (BLOCK_OVERLAP_FLOOR, BLOCK_OVERLAP_FLOOR)
    note = rq1.size_note(_nine(tmp_path / "ok", "a", floors))
    assert "09-mde-threshold a tables/bootstrap_size.csv" in note
    assert "the mean test 5.0%-7.0%, the difference test 6.0%-6.5% at a nominal 5%" in note
    assert "300 draws each (standard error about 1.3%)" in note
    with pytest.raises(ValueError, match="names 2 runs, expected one"):
        rq1.size_note(_nine(tmp_path / "two", "a, b", floors))
    with pytest.raises(ValueError, match="no size row at the block floor"):
        rq1.size_note(_nine(tmp_path / "floor", "a", (BLOCK_OVERLAP_FLOOR + 1,)))
    unnamed = _nine(tmp_path / "none", "a", floors)
    (unnamed.parent / "09-mde-threshold" / "results" / "README.md").write_text("# Results\n")
    with pytest.raises(ValueError, match="names no run"):
        rq1.size_note(unnamed)


def test_the_null_note_counts_the_cells_that_fire(rq1: ModuleType) -> None:
    # three cells: never fired, fired once (0.001 > level 0.0005), fired 3 % (60x the level)
    mde = pd.DataFrame(
        {
            "model": ["a", "b", "c"],
            "market": "crypto",
            "horizon": [1, 1, 24],
            "times": [100, 200, 52],
            "zero_rate": [0.0, 0.001, 0.03],
            "false_positive_level": 0.0005,
        }
    )
    verdict = pd.DataFrame(
        {"status": ["scored", "scored"], "p_value": [0.5, 0.04], "adjusted_p": [1.0, 0.12]}
    )
    note = rq1.null_note(mde, verdict, resamples=1000)
    assert "at most 3.0% of the resamples" in note
    assert "the largest is c crypto 24 bar(s), 0.030 with 52 decision points" in note
    assert "2 of 3 cells fire at least once, which at the resolution of 0.001 is above" in note
    assert "1 exceed ten times it" in note
    assert "smallest one-sided p of a scored cell is 0.04, 0.12 after Holm" in note


def _quarters(full: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Q2's cells at 1 bar (all reached at 1 %) and the verdict with their whole quarters
    keys = {"model": [f"m{i}" for i in range(len(full))], "market": "crypto", "horizon": 1}
    cells = pd.DataFrame({**keys, "is_reached": True, "mde_pct": 1.0})
    return cells, pd.DataFrame({**keys, "full_quarters": full})


def test_the_whole_quarter_note_names_the_late_models_and_their_share(rq1: ModuleType) -> None:
    cells, verdict = _quarters([4, 4, 2])
    note = rq1.whole_quarter_note(cells, verdict, RULE, share=1.0)
    assert "so m2 (late family" in note
    assert "the share would be 100% (2 of 2), with the same verdict" in note
    assert rq1.whole_quarter_note(*_quarters([4, 4]), RULE, share=1.0) == ""
    with pytest.raises(ValueError, match="1 power cells have no row in the verdict"):
        rq1.whole_quarter_note(cells, verdict.iloc[:2], RULE, share=1.0)
