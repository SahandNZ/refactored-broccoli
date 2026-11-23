"""The shared figure style of the papers: sizes, fonts, one colour and marker per model, saving.

Every figure of a paper is drawn with this style, so that all figures look alike and pass the
AAAI-27 limits (text at least 9 pt, lines at least 0.5 pt, no Type 3 fonts). Importing this
module changes nothing; a figure script calls `use_paper_style` once, draws, and ends with
`save_figure`.
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of Figure methods as Unknown, which
# no code here can fix; nothing else is silenced)

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.transforms import Bbox
    from matplotlib.typing import RcKeyType

SINGLE_WIDTH = 3.15
"""Width in inches of a one-column figure; included at natural size (one column is 3.31 in)."""

DOUBLE_WIDTH = 6.80
"""Width in inches of a two-column figure (figure*); the text block is 7.0 in."""

MIN_FONT_SIZE = 9.0
"""Smallest text size in points the AAAI-27 kit allows inside a figure."""

MIN_LINE_WIDTH = 0.5
"""Thinnest line in points the AAAI-27 kit allows."""

FAMILY_STYLE = {
    "Chronos-Bolt": {"color": "#56B4E9", "marker": "o"},
    "Kronos": {"color": "#009E73", "marker": "v"},
    "TimesFM 2.5": {"color": "#D55E00", "marker": "D"},
    "Chronos-2": {"color": "#0072B2", "marker": "s"},
    "TimesFM 3": {"color": "#E69F00", "marker": "^"},
    "TiRex 1.1": {"color": "#CC79A7", "marker": "P"},
    "TiRex-2": {"color": "#CC79A7", "marker": "p"},
    "Toto 2": {"color": "#56B4E9", "marker": "h"},
    "t0": {"color": "#B8A900", "marker": "X"},
    "Moirai 2": {"color": "#009E73", "marker": "*"},
    "Baseline": {"color": "#555555", "marker": "o"},
}
"""One colour and one marker per model family, the same in every figure.

Okabe-Ito has seven colours besides black, so ten families share some colours; every family has
its own marker, so no two families look alike, also in greyscale. Baselines are grey (the best
baseline is drawn black where it must stand out); t0 uses a darkened yellow because pure yellow
vanishes on white.
"""

SIZE_STYLE = {
    "": {"linestyle": "-", "markersize": 4.5},
    "small": {"linestyle": "--", "markersize": 3.5},
    "base": {"linestyle": "-", "markersize": 5.0},
}
"""Sizes of one family share colour and marker and differ by line style and marker size.

Filled against open markers is kept for significance (charts.md), so sizes never use the fill.
"""

# waiting for the owner's final model list: a model added to or dropped from 02-forecasts
# changes MODEL_FAMILY (and FAMILY_STYLE for a new family) in the same commit
MODEL_FAMILY = {
    "random-walk": ("Baseline", ""),
    "climatology": ("Baseline", ""),
    "ewma-normal": ("Baseline", ""),
    "garch-normal": ("Baseline", ""),
    "linear": ("Baseline", ""),
    "chronos-bolt-small": ("Chronos-Bolt", "small"),
    "chronos-bolt-base": ("Chronos-Bolt", "base"),
    "chronos-2": ("Chronos-2", ""),
    "timesfm-2.5": ("TimesFM 2.5", ""),
    "timesfm-3": ("TimesFM 3", ""),
    "tirex-1.1": ("TiRex 1.1", ""),
    "tirex-2": ("TiRex-2", ""),
    "toto-2-22m": ("Toto 2", "small"),
    "toto-2-313m": ("Toto 2", "base"),
    "t0-gamma": ("t0", ""),
    "moirai-2-small": ("Moirai 2", "small"),
    "kronos-small": ("Kronos", "small"),
    "kronos-base": ("Kronos", "base"),
}
"""The family and size of every model id of the benchmark (02-forecasts `[models] names`)."""


def model_style(model: str) -> dict[str, str | float]:
    """Return the colour, marker, line style and marker size of a model id.

    Parameters
    ----------
    model : str
        A model id of `MODEL_FAMILY`, e.g. ``"kronos-small"``.

    Returns
    -------
    dict of str to str or float
        ``color``, ``marker``, ``linestyle`` and ``markersize``.

    Raises
    ------
    KeyError
        If the model has no family in `MODEL_FAMILY`.
    """
    if model not in MODEL_FAMILY:
        raise KeyError(f"model {model!r} has no family in MODEL_FAMILY; add it there first")
    family, size = MODEL_FAMILY[model]
    return {**FAMILY_STYLE[family], **SIZE_STYLE[size]}


MARKET_ORDER = ("Crypto", "FX", "Commodities", "Indices", "US stocks")
"""The order of the markets in every panel row and table, as in tables/markets.tex."""

HORIZON_LINESTYLE = {1: "-", 4: "--", 24: ":"}
"""The line style of each horizon in bars; horizons never get colours."""

REFERENCE_LINE: dict[str, Any] = {
    "color": "#555555",
    "linewidth": 0.6,
    "linestyle": "--",
    "zorder": 0,
}
"""The dashed line at the no-effect value (1 for ratios, 0 for differences, 0.5 for hit rates).

Keyword arguments of matplotlib (``axis.axhline(1, **REFERENCE_LINE)``), typed `Any` as in
matplotlib's own stubs, so they unpack into any line method.
"""

MARK_COLOUR = "#1A1A1A"
"""The near-black of the marks of a forest plot whose rows are named models (charts.md 2a.3)."""

INTERVAL_COLOUR = "#8C8C8C"
"""The grey of the interval lines behind those marks: lighter than the marks (charts.md 2a.5)."""

ROW_BAND_COLOUR = "#EFEFEF"
"""The light grey of every second row of a forest plot (`shade_rows`); greyscale-safe, no text
is ever placed on it."""


def plain_tick(value: float, _position: float | None = None) -> str:
    """Return a tick label without trailing zeros and with a true minus sign (0.02, 0, -0.02).

    Parameters
    ----------
    value : float
        The tick value.
    _position : float or None
        The tick position (matplotlib's `FuncFormatter` passes it; unused).

    Returns
    -------
    str
        The label.
    """
    text = f"{value:.6g}"
    if text in {"-0", "0"}:
        return "0"
    return text.replace("-", "\N{MINUS SIGN}")


def shade_rows(axis: "Axes", count: int) -> None:
    """Shade every second row of a plot with one row per model, so each row is easy to follow.

    Rows are at y = 0, 1, ..., `count` - 1 (the first, on top, stays white); the bands lie
    behind everything and change no limit.

    Parameters
    ----------
    axis : Axes
        The axes whose rows to shade.
    count : int
        The number of rows.
    """
    for row in range(1, count, 2):
        axis.axhspan(row - 0.5, row + 0.5, color=ROW_BAND_COLOUR, linewidth=0, zorder=-1)


def top_legend(figure: "Figure", rows: "Sequence[Sequence[Artist]]", **options: Any) -> None:
    """Put one shared legend above the panels, in one row if it fits, else row by row.

    The entries read left to right in the order given. When one row would be wider than the
    figure, each list of `rows` becomes its own legend row (matplotlib fills a legend column by
    column, so shorter rows are padded with empty entries and the handles reordered).

    Parameters
    ----------
    figure : Figure
        The figure; drawn once to measure the legend.
    rows : sequence of sequence of Artist
        The legend handles, in the rows to use when one row does not fit.
    **options : Any
        Further keyword arguments of ``figure.legend`` (spacing).
    """
    from matplotlib.lines import Line2D

    handles = [handle for row in rows for handle in row]
    legend = figure.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), **options
    )
    figure.draw_without_rendering()
    if legend.get_window_extent().width <= figure.bbox.width:
        return
    legend.remove()
    columns = max(len(row) for row in rows)
    padded = [
        [*row, *(Line2D([], [], linestyle="none", label="") for _ in range(columns - len(row)))]
        for row in rows
    ]
    ordered = [row[column] for column in range(columns) for row in padded]
    figure.legend(handles=ordered, loc="outside upper center", ncols=columns, **options)


PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": MIN_FONT_SIZE,
    "axes.labelsize": MIN_FONT_SIZE,
    "axes.titlesize": MIN_FONT_SIZE,
    "xtick.labelsize": MIN_FONT_SIZE,
    "ytick.labelsize": MIN_FONT_SIZE,
    "legend.fontsize": MIN_FONT_SIZE,
    "legend.title_fontsize": MIN_FONT_SIZE,
    "figure.labelsize": MIN_FONT_SIZE,
    "figure.titlesize": MIN_FONT_SIZE,
    "lines.linewidth": 1.25,
    "lines.markersize": 4.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.5,
    "errorbar.capsize": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.constrained_layout.use": True,
    "savefig.bbox": "standard",
    "savefig.dpi": 300,
    "image.cmap": "cividis",
}
"""The matplotlib settings of a paper figure: TrueType fonts (never Type 3), 9 pt text."""

TERMES_FOLDER = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
"""TeX Gyre Termes, the Times-like font of the paper, as installed with TinyTeX."""


SERIF_FONTS = ["TeX Gyre Termes", "Liberation Serif"]
"""The serif of every figure's text: TeX Gyre Termes, the paper's Times clone and the font of
Figure 1 (the owner's choice, 2026-11-19), else Liberation Serif."""


def serif_rc() -> "dict[RcKeyType, Any]":
    """Register TeX Gyre Termes and return the matplotlib settings of the figures' serif.

    Termes is registered when TinyTeX has it; without it matplotlib falls back to Liberation
    Serif (`SERIF_FONTS`).

    Returns
    -------
    dict of RcKeyType to Any
        ``font.family``, ``font.serif`` and ``mathtext.fontset``, to put in an rc context.
    """
    from matplotlib import font_manager

    for font_file in sorted(TERMES_FOLDER.glob("texgyretermes-*.otf")):
        font_manager.fontManager.addfont(str(font_file))
    return {"font.family": "serif", "font.serif": SERIF_FONTS, "mathtext.fontset": "stix"}


def use_paper_style() -> None:
    """Set the paper style for the figures drawn after this call.

    Registers TeX Gyre Termes when TinyTeX has it (otherwise matplotlib falls back to
    STIXGeneral) and sets `PAPER_RC`.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for font_file in sorted(TERMES_FOLDER.glob("texgyretermes-*.otf")):
        font_manager.fontManager.addfont(str(font_file))
    plt.style.use(PAPER_RC)


def check_figure(figure: "Figure", *, min_font_size: float = MIN_FONT_SIZE) -> None:
    """Check that every text is at least 9 pt and every visible line at least 0.5 pt.

    Parameters
    ----------
    figure : Figure
        The figure to check.
    min_font_size : float
        The smallest text size allowed, in points: `MIN_FONT_SIZE` (the AAAI-27 limit) unless
        the owner approved an exception for one figure (charts.md 2a.14).

    Raises
    ------
    ValueError
        If a text is smaller than `min_font_size` or a visible line thinner than
        `MIN_LINE_WIDTH`.
    """
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    for text in figure.findobj(Text):
        if text.get_text() and float(text.get_fontsize()) < min_font_size:
            raise ValueError(f"text below {min_font_size} pt: {text.get_text()!r}")
    for line in figure.findobj(Line2D):
        width = line.get_linewidth()
        if line.get_visible() and 0 < width < MIN_LINE_WIDTH:
            raise ValueError(f"line below {MIN_LINE_WIDTH} pt: {width} pt")


def save_figure(
    figure: "Figure",
    pdf_path: Path,
    grey_path: Path,
    *,
    min_font_size: float = MIN_FONT_SIZE,
    exports: "Sequence[Path]" = (),
    crop: "float | Bbox | None" = None,
) -> None:
    """Check the figure, write it as a PDF without dates, and write a greyscale preview.

    The PDF has no creation date and no creator, so a rerun gives the same file. The greyscale
    PNG is for a person to check that every series can be told apart without colour.

    Parameters
    ----------
    figure : Figure
        The figure to save; it is closed afterwards.
    pdf_path : Path
        The PDF to write, e.g. ``paper/figures/fig_rq1_power.pdf``.
    grey_path : Path
        The greyscale PNG to write, e.g. ``paper/build/fig_rq1_power_grey.png``
        (not in git).
    min_font_size : float
        The smallest text size allowed (`check_figure`).
    exports : sequence of Path
        Further copies to write, in the format of their suffix (``.svg``, ``.png`` at 300
        dpi), with the same options as the PDF.
    crop : float, Bbox or None
        Crop every file to its drawn content (``bbox_inches="tight"``) with this padding in
        inches, so the figure is no wider than it must be; a Bbox (in inches) crops every file
        to that box; None keeps the drawn size.

    Raises
    ------
    ValueError
        If the figure breaks a limit of `check_figure`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    from PIL import Image

    check_figure(figure, min_font_size=min_font_size)
    cropping: dict[str, Any] = {}
    if isinstance(crop, Bbox):
        cropping = {"bbox_inches": crop}
    elif crop is not None:
        cropping = {"bbox_inches": "tight", "pad_inches": crop}
    figure.savefig(pdf_path, metadata={"CreationDate": None, "Creator": None}, **cropping)
    for path in exports:
        path.parent.mkdir(parents=True, exist_ok=True)
        # no date in a vector copy, so a rerun gives the same file
        metadata = {"Date": None, "Creator": None} if path.suffix == ".svg" else None
        figure.savefig(path, dpi=300, metadata=metadata, **cropping)
    grey_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(grey_path, dpi=150, **cropping)
    Image.open(grey_path).convert("L").save(grey_path)
    plt.close(figure)
