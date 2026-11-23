# pyright: reportUnknownMemberType=false
# (matplotlib's own stubs type the **kwargs of every Axes method as Unknown; see
# src/figures/_style.py)

import contextlib
import re
import subprocess
import sys
import tomllib
import zlib
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

from src.figures import (
    DOUBLE_WIDTH,
    FAMILY_STYLE,
    HORIZON_LINESTYLE,
    MARKET_ORDER,
    MIN_FONT_SIZE,
    MODEL_FAMILY,
    SINGLE_WIDTH,
    check_figure,
    model_style,
    plain_tick,
    save_figure,
    shade_rows,
    top_legend,
    use_paper_style,
)

CONFIG = Path("experiments/02-forecasts/study.toml")


def _pdf_text(path: Path) -> bytes:
    # the raw PDF plus every stream that zlib can open
    data = path.read_bytes()
    parts = [data]
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        with contextlib.suppress(zlib.error):
            parts.append(zlib.decompress(match.group(1)))
    return b"\n".join(parts)


def _small_figure():
    figure, axes = plt.subplots(figsize=(SINGLE_WIDTH, 2.0))
    style = model_style("chronos-2")
    axes.plot([0, 1, 2], [1.0, 0.9, 1.1], color=str(style["color"]), marker=str(style["marker"]))
    axes.set_xlabel("Planted gain (%)")
    axes.set_ylabel("Share of cells")
    return figure


SETTINGS_CHECK = """
import matplotlib
before = dict(matplotlib.rcParams)
import src.figures
print(sorted(key for key in before if matplotlib.rcParams[key] != before[key]))
"""

LAZY_CHECK = """
import sys
import src.figures
print(sorted(name for name in sys.modules if name.startswith(("matplotlib", "PIL"))))
"""


def _run_python(code: str) -> str:
    # a fresh interpreter, because the other tests of this process already changed settings
    command = [sys.executable, "-c", code]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def test_import_changes_no_setting():
    # importing the package must not change any matplotlib setting (python.md rule 7)
    assert _run_python(SETTINGS_CHECK) == "[]"


def test_import_loads_neither_matplotlib_nor_pil():
    # matplotlib and PIL are imported lazily, inside the functions (python.md rule 12)
    assert _run_python(LAZY_CHECK) == "[]"


def _font_settings() -> list[str]:
    # every setting that gives a font size
    ends = ("labelsize", "titlesize", "fontsize")
    return [key for key in matplotlib.rcParams if key == "font.size" or key.endswith(ends)]


def test_paper_style_sets_9_pt_in_every_font_setting():
    keys = _font_settings()
    assert len(keys) >= 9
    use_paper_style()
    for key in keys:
        assert matplotlib.rcParams[key] == MIN_FONT_SIZE, key


def test_every_kind_of_text_is_9_pt_in_the_paper_style():
    use_paper_style()
    figure, axes = plt.subplots(figsize=(SINGLE_WIDTH, 2.0))
    lines = axes.plot([0, 1], [0, 1], label="line")
    image = axes.imshow([[0.0, 1.0]])
    axes.set_title("title")
    axes.set_xlabel("x label")
    axes.set_ylabel("y label")
    axes.legend(handles=lines, title="legend title")
    axes.annotate("note", (0.5, 0.5))
    axes.text(0.2, 0.2, "text")
    figure.colorbar(image, label="colour bar")
    figure.suptitle("figure title")
    figure.supxlabel("figure x label")
    figure.supylabel("figure y label")
    figure.canvas.draw()
    texts = [text for text in figure.findobj(Text) if text.get_text()]
    assert len(texts) > 12
    for text in texts:
        assert text.get_fontsize() == pytest.approx(MIN_FONT_SIZE), text.get_text()
    plt.close(figure)


def test_every_model_family_has_its_own_colour_and_marker_pair():
    models = [name for name in FAMILY_STYLE if name != "Baseline"]
    markers = [FAMILY_STYLE[name]["marker"] for name in models]
    pairs = {(FAMILY_STYLE[name]["color"], FAMILY_STYLE[name]["marker"]) for name in models}
    assert len(set(markers)) == len(markers)
    assert len(pairs) == len(models)


def test_every_model_of_the_benchmark_config_has_a_style():
    names = tomllib.loads(CONFIG.read_text())["models"]["names"]
    for name in names:
        style = model_style(name)
        assert set(style) == {"color", "marker", "linestyle", "markersize"}
    assert set(MODEL_FAMILY) == set(names)


def test_sizes_of_one_family_differ_but_share_colour_and_marker():
    small, base = model_style("kronos-small"), model_style("kronos-base")
    assert small["color"] == base["color"]
    assert small["marker"] == base["marker"]
    assert small["linestyle"] != base["linestyle"]


def test_an_unknown_model_is_refused():
    with pytest.raises(KeyError, match="no family"):
        model_style("not-a-model")


def test_constants_match_the_paper():
    assert SINGLE_WIDTH < 3.31
    assert DOUBLE_WIDTH < 7.0
    assert MARKET_ORDER == ("Crypto", "FX", "Commodities", "Indices", "US stocks")
    assert set(HORIZON_LINESTYLE) == {1, 4, 24}


def test_saved_pdf_has_no_type3_font_and_no_date(tmp_path: Path):
    use_paper_style()
    pdf_path, grey_path = tmp_path / "fig.pdf", tmp_path / "build" / "fig_grey.png"
    save_figure(_small_figure(), pdf_path, grey_path)
    text = _pdf_text(pdf_path)
    assert re.search(rb"/Subtype\s*/Type3", text) is None
    assert re.search(rb"/FontFile[23]?\s", text) is not None
    assert re.search(rb"/CreationDate", text) is None
    assert Image.open(grey_path).mode == "L"


def test_rerun_gives_the_same_pdf(tmp_path: Path):
    use_paper_style()
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    save_figure(_small_figure(), first, tmp_path / "a.png")
    save_figure(_small_figure(), second, tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()


def test_small_text_is_refused():
    use_paper_style()
    figure = _small_figure()
    figure.axes[0].set_title("too small", fontsize=8)
    with pytest.raises(ValueError, match="text below"):
        check_figure(figure)
    plt.close(figure)


def test_small_text_passes_only_with_a_lower_limit_and_still_has_a_floor():
    use_paper_style()
    figure = _small_figure()
    figure.axes[0].set_title("six points", fontsize=6)
    check_figure(figure, min_font_size=6.0)  # the owner's exception (charts.md 2a.14)
    with pytest.raises(ValueError, match=r"text below 7\.0 pt"):
        check_figure(figure, min_font_size=7.0)
    plt.close(figure)


@pytest.mark.parametrize("width", [0.1, 0.3, 0.4, 0.49])
def test_thin_line_is_refused(width: float):
    use_paper_style()
    figure = _small_figure()
    figure.axes[0].plot([0, 1], [0, 1], linewidth=width)
    with pytest.raises(ValueError, match="line below"):
        check_figure(figure)
    plt.close(figure)


def test_a_line_of_0_5_pt_and_a_hidden_thin_line_pass():
    use_paper_style()
    figure = _small_figure()
    figure.axes[0].plot([0, 1], [0, 1], linewidth=0.5)
    figure.axes[0].plot([0, 1], [1, 0], linewidth=0.1, visible=False)
    check_figure(figure)
    plt.close(figure)


def test_paper_style_passes_its_own_check():
    use_paper_style()
    figure = _small_figure()
    check_figure(figure)
    plt.close(figure)


def test_a_five_panel_figure_keeps_its_size_in_the_pdf(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
):
    use_paper_style()
    figure, axes = plt.subplots(1, 5, figsize=(DOUBLE_WIDTH, 2.3), sharex=True, sharey=True)
    for panel, market in zip(axes, MARKET_ORDER, strict=True):
        panel.plot([0, 1], [0, 1])
        panel.set_title(market)
    figure.supxlabel("CRPS relative to the best baseline (lower is better)")
    pdf_path = tmp_path / "five.pdf"
    save_figure(figure, pdf_path, tmp_path / "five.png")
    assert page_size(pdf_path) == pytest.approx((6.80, 2.30))


def test_every_font_in_the_saved_pdf_is_at_least_9_pt(tmp_path: Path):
    use_paper_style()
    pdf_path = tmp_path / "fig.pdf"
    save_figure(_small_figure(), pdf_path, tmp_path / "fig.png")
    sizes = [float(size) for size in re.findall(rb"/F\d+ ([\d.]+) Tf", _pdf_text(pdf_path))]
    assert sizes
    assert min(sizes) >= 9


def _keys(labels: list[str]) -> list[Line2D]:
    return [Line2D([], [], marker="o", linestyle="none", label=label) for label in labels]


@pytest.mark.parametrize(("second", "expected_rows"), [(["b1"], 1), (["a long key " * 6, "b2"], 2)])
def test_the_top_legend_is_one_row_when_it_fits_else_one_row_per_group(
    second: list[str], expected_rows: int
) -> None:
    # entries read left to right: a legend that fits is one row; one that does not keeps each
    # group on its own row (padded with empty entries), never a column-wise mix of the groups
    with matplotlib.rc_context():
        use_paper_style()
        figure, _ = plt.subplots(figsize=(SINGLE_WIDTH, 2.0))
        top_legend(figure, [_keys(["a1", "a2"]), _keys(second)])
        figure.draw_without_rendering()
        (legend,) = figure.legends
        texts = [text for text in legend.get_texts() if text.get_text()]
        rows: dict[float, list[str]] = {}
        for text in texts:
            rows.setdefault(round(text.get_window_extent().y0, 1), []).append(text.get_text())
        assert len(rows) == expected_rows
        if expected_rows == 2:
            top, bottom = (rows[y] for y in sorted(rows, reverse=True))
            assert (top, bottom) == (["a1", "a2"], second)
        plt.close(figure)


def test_every_second_row_is_shaded_behind_the_marks() -> None:
    with matplotlib.rc_context():
        use_paper_style()
        figure, axis = plt.subplots()
        axis.set_ylim(-0.5, 4.5)
        shade_rows(axis, 5)
        bands = list(axis.patches)
        assert len(bands) == 2
        assert all(band.get_zorder() < 0 for band in bands)
        assert axis.get_ylim() == (-0.5, 4.5)
        plt.close(figure)


@pytest.mark.parametrize(
    ("value", "label"),
    [(0.0, "0"), (-0.0, "0"), (0.02, "0.02"), (-0.025, "\N{MINUS SIGN}0.025"), (1.0, "1")],
)
def test_plain_ticks_have_no_trailing_zeros_and_a_true_minus(value: float, label: str) -> None:
    assert plain_tick(value) == label
