"""The figure of RQ4 (`src.figures.draw_leakage`)."""

# pyright: reportUnknownMemberType=false
# (as in src.figures: matplotlib's stubs leave the canvas methods partly Unknown)

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.figures import MARKET_ORDER, MODEL_FAMILY, draw_leakage

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# constrained layout warns when it cannot fit the rows ("axes sizes collapsed to zero")
pytestmark = pytest.mark.filterwarnings("error::UserWarning")

LABELS = {"kronos-small": "Kronos (S)", "chronos-2": "Chronos-2"}
MARKET_LABELS = {market: market.upper() for market in MARKET_ORDER}


def _table(markets: tuple[str, ...] = MARKET_ORDER) -> pd.DataFrame:
    # two models, one row per market; chronos-2's last market not scored (NaN marks), its
    # other cells leaky; kronos-small's intervals wide (cut at an x_limit of 0.05)
    rows: list[dict[str, object]] = []
    for model in ["chronos-2", "kronos-small"]:
        for index, market in enumerate(markets):
            unscored = model == "chronos-2" and index == len(markets) - 1
            did = np.nan if unscored else 0.01
            spread = 0.2 if model == "kronos-small" else 0.02
            rows.append({
                "model": model, "market": market, "horizon": 1, "did": did,
                "low": did - spread, "high": did + spread,
                "shows_leakage": model == "chronos-2" and not unscored,
            })  # fmt: skip
    return pd.DataFrame(rows)


def _draw(tmp_path: Path, table: pd.DataFrame, **changes: object) -> None:
    arguments: dict[str, object] = {
        "labels": LABELS,
        "market_labels": MARKET_LABELS,
        "x_limit": 0.05,
        "pdf_path": tmp_path / "rq4.pdf",
        "grey_path": tmp_path / "rq4_grey.png",
    }
    draw_leakage(table, **(arguments | changes))  # type: ignore[arg-type]


def test_the_leakage_figure_is_drawn_at_its_size_in_the_paper_style(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    before = dict(matplotlib.rcParams)
    _draw(tmp_path, _table())
    # 6.80 in wide, 0.75 + 0.2 in per model high
    assert page_size(tmp_path / "rq4.pdf") == pytest.approx((6.80, 1.15))
    assert Image.open(tmp_path / "rq4_grey.png").mode == "L"
    assert dict(matplotlib.rcParams) == before
    assert b"/CreationDate" not in (tmp_path / "rq4.pdf").read_bytes()


def test_the_leakage_figure_draws_one_panel_per_market_with_marks_fills_and_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[Figure] = []

    def keep(figure: "Figure", pdf_path: Path, grey_path: Path) -> None:
        drawn.append(figure)

    monkeypatch.setattr("src.figures._leakage.save_figure", keep)
    _draw(tmp_path, _table())
    axes = drawn[0].axes
    # one panel per market, titled with its label, in MARKET_ORDER
    assert [axis.get_title() for axis in axes] == [MARKET_LABELS[m] for m in MARKET_ORDER]
    # rows top to bottom in the order of the labels, with their values
    ticks = sorted(axes[0].get_yticklabels(), key=lambda text: text.get_position()[1])
    assert [text.get_text() for text in ticks] == ["Kronos (S)", "Chronos-2"]
    assert axes[0].get_xlim() == pytest.approx((-0.05, 0.05))
    marks = [line for axis in axes for line in axis.get_lines() if line.get_marker() == "o"]
    # one mark per scored cell: 9 (chronos-2's unscored cell has none); filled = leaky
    assert len(marks) == 9
    fills = {
        float(np.asarray(line.get_ydata())[0]) < 0.5: str(line.get_markerfacecolor())
        for line in marks
    }
    assert fills[True] == "white"  # kronos-small (row 0): open
    assert fills[False] != "white"  # chronos-2 (row 1): filled
    # kronos-small's five intervals run past the edge on both sides: two arrowheads each
    cuts = [
        line for axis in axes for line in axis.get_lines() if str(line.get_marker()) in {"<", ">"}
    ]
    assert len(cuts) == 10
    assert not drawn[0].legends  # the caption names the encodings


def test_a_market_outside_the_order_or_a_model_without_a_label_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="crypto"):
        _draw(tmp_path, _table(("crypto",)))
    with pytest.raises(ValueError, match="have no label"):
        _draw(tmp_path, _table(), labels={})
    with pytest.raises(ValueError, match="one horizon"):
        _draw(tmp_path, _table().assign(horizon=[1, 4] * 5))
    with pytest.raises(ValueError, match="x_limit must be positive"):
        _draw(tmp_path, _table(), x_limit=0.0)


@pytest.mark.parametrize("count", [2, 13])
def test_each_model_row_of_the_leakage_figure_is_a_fifth_of_an_inch_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    drawn: list[Figure] = []

    def keep(figure: "Figure", pdf_path: Path, grey_path: Path) -> None:
        figure.canvas.draw()
        drawn.append(figure)

    monkeypatch.setattr("src.figures._leakage.save_figure", keep)
    names = [f"m{index}" for index in range(count)]
    for name in names:
        monkeypatch.setitem(MODEL_FAMILY, name, ("Chronos-2", ""))
    table = pd.concat([_table().assign(model=name) for name in names], ignore_index=True)
    table = table.drop_duplicates(["model", "market"])
    _draw(tmp_path, table, labels={name: name for name in names})
    figure = drawn[0]
    axes_height = figure.axes[0].get_position().height * figure.get_figheight()
    assert axes_height / count == pytest.approx(0.2, abs=0.05)
