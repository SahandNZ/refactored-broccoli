"""The figure of RQ2 (`src.figures.draw_centre_scale`, `src.figures.read_centre_scale`)."""

# pyright: reportUnknownMemberType=false
# (as in src.figures: matplotlib's stubs leave the canvas methods partly Unknown)

import csv
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex
from PIL import Image

from src.figures import (
    ORIGINAL_LAYOUT,
    PAPER_LAYOUT,
    RQ2_MARKETS,
    RQ2_MODELS,
    draw_centre_scale,
    read_centre_scale,
)
from src.figures._centre_scale import BLUE, FIGURE_HEIGHT, FIGURE_WIDTH
from src.figures._style import SERIF_FONTS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

COLUMNS: tuple[str, ...] = (
    "model", "market", "horizon", "hit_rate", "hit_low", "hit_high", "reference_hit_rate",
    "is_centre_tested", "coverage_gap", "coverage_gap_low", "coverage_gap_high",
    "is_calibrated",
)  # fmt: skip
"""The columns of ``rq2.csv`` the figure reads."""


def _row(model: str, market: str) -> dict[str, str]:
    # EWMA-normal has no hit rate (its median is always zero); chronos-2 is tested and
    # calibrated, every other model is neither
    has_hit = model != "ewma-normal"
    is_marked = model == "chronos-2"
    return {
        "model": model, "market": market, "horizon": "1",
        "hit_rate": "0.51" if has_hit else "", "hit_low": "0.5" if has_hit else "",
        "hit_high": "0.52" if has_hit else "", "reference_hit_rate": "0.5" if has_hit else "",
        "is_centre_tested": str(is_marked), "coverage_gap": "0.01",
        "coverage_gap_low": "-0.01", "coverage_gap_high": "0.03",
        "is_calibrated": str(is_marked),
    }  # fmt: skip


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _full_rows() -> list[dict[str, str]]:
    return [_row(model, market) for model, _ in RQ2_MODELS for market, _ in RQ2_MARKETS]


def _drawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "Figure":
    # draw the full table and keep the figure open to look at it
    kept: list[Figure] = []
    monkeypatch.setattr(plt, "close", kept.append)
    data = read_centre_scale(_write(tmp_path / "rq2.csv", _full_rows()))
    draw_centre_scale(data, pdf_path=tmp_path / "rq2.pdf", png_path=tmp_path / "rq2.png")
    return kept[0]


def test_the_table_is_read_as_text_by_model_and_market(tmp_path: Path) -> None:
    data = read_centre_scale(_write(tmp_path / "rq2.csv", _full_rows()))
    assert len(data) == len(RQ2_MODELS) * len(RQ2_MARKETS)
    assert data["ewma-normal", "Crypto"]["hit_rate"] == ""
    assert data["chronos-2", "US stocks"]["is_calibrated"] == "True"


def test_a_table_not_at_one_bar_or_with_a_missing_cell_raises(tmp_path: Path) -> None:
    rows = _full_rows()
    other = [*rows[:-1], {**rows[-1], "horizon": "4"}]
    with pytest.raises(ValueError, match="one-bar"):
        read_centre_scale(_write(tmp_path / "horizon.csv", other))
    with pytest.raises(ValueError, match="one estimate per model and market"):
        read_centre_scale(_write(tmp_path / "missing.csv", rows[:-1]))


def test_the_figure_is_cropped_within_the_text_width_without_dates_and_settings_stay(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    before = dict(matplotlib.rcParams)
    data = read_centre_scale(_write(tmp_path / "rq2.csv", _full_rows()))
    pdf, png = tmp_path / "out" / "rq2.pdf", tmp_path / "out" / "rq2.png"
    draw_centre_scale(data, pdf_path=pdf, png_path=png)
    # cropped to the content: no wider than the paper's text width, PNG and PDF alike
    assert pytest.approx(7.0) == FIGURE_WIDTH
    width, height = page_size(pdf)
    assert 6.5 < width <= FIGURE_WIDTH
    assert 5.5 < height <= FIGURE_HEIGHT
    assert b"CreationDate" not in pdf.read_bytes()
    with Image.open(png) as image:
        assert image.size[0] == pytest.approx(320 * width, abs=2)
        assert image.size[1] == pytest.approx(320 * height, abs=2)
    assert dict(matplotlib.rcParams) == before


def test_one_panel_per_market_in_each_block_with_the_model_labels_on_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = _drawn(tmp_path, monkeypatch)
    axes = figure.axes
    titles = ["Crypto", "FX", "Commodities", "Indices", "Stocks"]
    assert [axis.get_title() for axis in axes] == titles * 2
    labels = [label.get_text() for label in axes[0].get_yticklabels()]
    assert labels == [label for _, label in RQ2_MODELS]
    assert all(label.get_text() == "" for label in axes[1].get_yticklabels())
    texts = [text.get_text() for text in figure.texts]
    assert "(a) Sign hit rate of the forecast median" in texts
    assert "(b) 10\u201390% interval coverage gap" in texts
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "Climatology (a)", "Untested (a)", "Coverage criterion unmet (b)", "No estimate",
    ]  # fmt: skip


def test_marks_are_filled_only_when_tested_or_calibrated_and_a_dash_marks_no_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = _drawn(tmp_path, monkeypatch)
    hit_axis, gap_axis = figure.axes[0], figure.axes[len(RQ2_MARKETS)]
    # panel (a): EWMA-normal has a dash, every other model a mark and climatology's tick
    assert [text.get_text() for text in hit_axis.texts] == ["—"]
    marks = [line for line in hit_axis.get_lines() if line.get_marker() == "o"]
    assert len(marks) == len(RQ2_MODELS) - 1
    fills = [to_hex(str(line.get_markerfacecolor())) for line in marks]
    assert fills.count(BLUE.lower()) == 1
    ticks = [line for line in hit_axis.get_lines() if line.get_marker() == "|"]
    assert len(ticks) == len(RQ2_MODELS) - 1
    # panel (b): every model has a mark, only chronos-2's is filled; x ticks carry a sign
    gap_marks = [line for line in gap_axis.get_lines() if line.get_marker() == "o"]
    assert len(gap_marks) == len(RQ2_MODELS)
    gap_fills = [to_hex(str(line.get_markerfacecolor())) for line in gap_marks]
    assert gap_fills.count(BLUE.lower()) == 1
    figure.canvas.draw()
    assert [label.get_text() for label in gap_axis.get_xticklabels()] == [
        "\u221220", "\u221210", "0", "+10",
    ]  # fmt: skip


def test_a_study_table_with_other_models_and_markets_is_drawn_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = (("ewma-normal", "EWMA-normal"), ("centre-only", "centre-only"))
    markets = (("Crypto", "o"), ("US stocks", "v"))
    rows = [_row(model, market) for model, _ in models for market, _ in markets]
    path = _write(tmp_path / "rq2.csv", rows)
    with pytest.raises(ValueError, match="one estimate per model and market"):
        read_centre_scale(path)
    kept: list[Figure] = []
    monkeypatch.setattr(plt, "close", kept.append)
    data = read_centre_scale(path, models=models, markets=markets)
    pdf, png = tmp_path / "rq2.pdf", tmp_path / "rq2.png"
    draw_centre_scale(data, pdf_path=pdf, png_path=png, models=models, markets=markets)
    assert [axis.get_title() for axis in kept[0].axes] == ["Crypto", "Stocks"] * 2
    labels = [label.get_text() for label in kept[0].axes[0].get_yticklabels()]
    assert labels == ["EWMA-normal", "centre-only"]
    assert pdf.exists()
    assert png.exists()


def _embedded_fonts(pdf: Path) -> set[str]:
    # the base names of the fonts a PDF embeds (subset prefix dropped)
    names = re.findall(rb"/FontName /([A-Za-z+-]+)", pdf.read_bytes())
    return {name.split(b"+")[-1].decode() for name in names}


def test_the_original_layout_is_the_owner_script_number_for_number() -> None:
    # plot_rq2.py v3 (2026-11-19): figsize, left/right/gap, block positions, x label, legend
    assert ORIGINAL_LAYOUT.size == (8.35, 7.3)
    assert (ORIGINAL_LAYOUT.left, ORIGINAL_LAYOUT.right, ORIGINAL_LAYOUT.gap) == (
        0.166, 0.975, 0.023,
    )  # fmt: skip
    assert ORIGINAL_LAYOUT.blocks == ((0.938, 0.565), (0.500, 0.127))
    assert ORIGINAL_LAYOUT.title_x == 0.034
    assert ORIGINAL_LAYOUT.axes_height == 0.309
    assert ORIGINAL_LAYOUT.x_label_drop == 0.039
    assert ORIGINAL_LAYOUT.legend_y == 0.032
    assert ORIGINAL_LAYOUT.market_titles == {}
    assert SERIF_FONTS == ["TeX Gyre Termes", "Liberation Serif"]


def test_both_layouts_use_the_owner_serif_and_title_the_stock_market_each_its_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = read_centre_scale(_write(tmp_path / "rq2.csv", _full_rows()))
    kept: list[Figure] = []
    monkeypatch.setattr(plt, "close", kept.append)
    for name, layout in (("paper", PAPER_LAYOUT), ("original", ORIGINAL_LAYOUT)):
        pdf = tmp_path / f"{name}.pdf"
        draw_centre_scale(data, pdf_path=pdf, png_path=tmp_path / f"{name}.png", layout=layout)
        fonts = _embedded_fonts(pdf)
        assert fonts
        assert all(font.startswith("TeXGyreTermes") for font in fonts)
    paper, original = kept
    assert paper.axes[-1].get_title() == "Stocks"
    assert original.axes[-1].get_title() == "US stocks"
    assert tuple(original.get_size_inches()) == pytest.approx((8.35, 7.3))
    assert all(axis.title.get_fontfamily() == ["serif"] for axis in original.axes)
