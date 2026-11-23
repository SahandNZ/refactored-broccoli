"""The two-panel power figure (`src.figures.draw_power_summary`)."""

# pyright: reportUnknownMemberType=false
# (as in src.figures: matplotlib's stubs leave the canvas methods partly Unknown)

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.benchmark import threshold_shares
from src.figures import PowerLines, draw_power_summary, empirical_shares, outcome_counts

if TYPE_CHECKING:
    from matplotlib.figure import Figure

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def _cells() -> pd.DataFrame:
    # three horizons; one 1-bar cell and one 24-bar cell never reach an MDE; the last cell of
    # each horizon is untestable (too few quarters) and never reached
    values = [0.4, 1.0, 2.0, np.nan, np.nan, 1.5, 2.5, np.nan, 3.0, 1.9, np.nan, 7.0, np.nan]
    horizons = [1, 1, 1, 1, 1, 4, 4, 4, 24, 24, 24, 24, 24]
    testable = [True, True, True, True, False, True, True, False, True, True, True, True, False]
    return pd.DataFrame({
        "horizon": horizons, "mde_pct": values, "is_reached": ~np.isnan(values),
        "is_testable": testable, "model": ["m"] * len(values),
    })  # fmt: skip


def _shares(cells: pd.DataFrame) -> pd.DataFrame:
    return threshold_shares(cells[cells["is_testable"]], thresholds=GRID)


def _draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: object) -> "Figure":
    drawn: list[Figure] = []

    def keep(figure: "Figure", pdf_path: Path, grey_path: Path, **options: object) -> None:
        figure.canvas.draw()
        drawn.append(figure)

    monkeypatch.setattr("src.figures._power_summary.save_figure", keep)
    cells = _cells()
    arguments: dict[str, object] = {
        "lines": PowerLines(chosen_pct=2.0, required_share=0.8, x_max=10.0),
        "pdf_path": tmp_path / "p.pdf", "grey_path": tmp_path / "p.png",
    }  # fmt: skip
    shares = changes.pop("shares", _shares(cells))
    draw_power_summary(cells, shares, **(arguments | changes))  # type: ignore[arg-type]
    return drawn[0]


def test_the_empirical_shares_count_every_cell_and_include_the_threshold() -> None:
    cells = _cells()
    cells = cells[cells["is_testable"]]
    # 1 bar: 0.4, 1.0, 2.0 reached and one never reached: 3 of 4 at X = 2 (MDE <= X)
    assert empirical_shares(cells, 1, np.array([1.99, 2.0, 10.0])).tolist() == [0.5, 0.75, 0.75]
    with pytest.raises(ValueError, match="no cell of horizon 12"):
        empirical_shares(cells, 12, np.array([1.0]))


def test_every_cell_gets_one_outcome_and_the_untestable_ones_stay_in() -> None:
    counts = outcome_counts(_cells(), 2.0).set_index("horizon")
    assert counts.loc[1].to_dict() == {
        "cells": 5, "testable": 4, "resolved": 3, "reached": 0, "not_reached": 1,
        "untestable": 1,
    }  # fmt: skip
    assert counts.loc[24].to_dict() == {
        "cells": 5, "testable": 4, "resolved": 1, "reached": 2, "not_reached": 1,
        "untestable": 1,
    }  # fmt: skip
    outcomes = ["resolved", "reached", "not_reached", "untestable"]
    assert (counts[outcomes].sum(axis=1) == counts["cells"]).all()


def test_the_key_of_panel_b_prints_the_share_of_the_testable_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = _draw(tmp_path, monkeypatch)
    bars, curves = figure.axes[0], figure.axes[1]
    key = curves.child_axes[0]
    # 1 bar 3/4, 4 bars 1/2, 24 bars 1/4 of the testable cells at X = 2; not reached 1, 0, 1
    texts = {text.get_text() for text in key.texts}
    assert {"75.0% (3/4)", "50.0% (1/2)", "25.0% (1/4)", "target 80%"} <= texts
    assert {"1", "0"} <= texts
    names = {text.get_text() for text in bars.texts}
    assert {"1 bar", "4 bars", "24 bars", "n = 5", "n = 3"} <= names
    assert bars.get_xlim() == (0.0, 100.0)
    assert curves.get_ylim() == (0.0, 100.0)
    # both panels the same width, labels included
    boxes = [axis.get_tightbbox() for axis in (bars, curves)]
    widths = [float(box.width) for box in boxes if box is not None]
    assert widths[0] == pytest.approx(widths[1], rel=0.02)


def test_a_table_that_differs_from_the_cells_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shares = _shares(_cells())
    shares.loc[4, "share_1h"] = 0.8
    with pytest.raises(ValueError, match="differ from the table at 1 bars"):
        _draw(tmp_path, monkeypatch, shares=shares)
    with pytest.raises(ValueError, match="not a threshold"):
        _draw(tmp_path, monkeypatch, lines=PowerLines(chosen_pct=2.2, required_share=0.8, x_max=10))


def test_the_figure_is_written_as_pdf_svg_and_png(tmp_path: Path) -> None:
    cells = _cells()
    draw_power_summary(
        cells, _shares(cells),
        lines=PowerLines(chosen_pct=2.0, required_share=0.8, x_max=10.0),
        pdf_path=tmp_path / "p.pdf", grey_path=tmp_path / "p_grey.png",
        exports=[tmp_path / "p.svg", tmp_path / "p.png"],
    )  # fmt: skip
    for name in ["p.pdf", "p_grey.png", "p.svg", "p.png"]:
        assert (tmp_path / name).stat().st_size > 0
    assert b"/CreationDate" not in (tmp_path / "p.pdf").read_bytes()
