"""The figures of the lab: the shared paper style and one drawing function per chart kind.

The style (`use_paper_style`, `check_figure`, `save_figure` and the constants of sizes, colours,
markers and line styles) is in `src.figures._style`. Every drawing function takes a results
table and the paths to write, draws in the paper style, and saves through `save_figure` (a PDF
without dates and a greyscale PNG preview); it leaves matplotlib's settings as they were.
Importing this package changes nothing.
"""

from src.figures._centre_scale import (
    ORIGINAL_LAYOUT,
    PAPER_LAYOUT,
    RQ2_MARKETS,
    RQ2_MODELS,
    draw_centre_scale,
    read_centre_scale,
)
from src.figures._correlation import draw_correlation
from src.figures._leakage import draw_leakage
from src.figures._mde import draw_mde, draw_threshold_shares
from src.figures._power_summary import (
    PowerLines,
    draw_power_summary,
    empirical_shares,
    outcome_counts,
)
from src.figures._style import (
    DOUBLE_WIDTH,
    FAMILY_STYLE,
    HORIZON_LINESTYLE,
    INTERVAL_COLOUR,
    MARK_COLOUR,
    MARKET_ORDER,
    MIN_FONT_SIZE,
    MIN_LINE_WIDTH,
    MODEL_FAMILY,
    PAPER_RC,
    REFERENCE_LINE,
    SERIF_FONTS,
    SINGLE_WIDTH,
    SIZE_STYLE,
    TERMES_FOLDER,
    check_figure,
    model_style,
    plain_tick,
    save_figure,
    serif_rc,
    shade_rows,
    top_legend,
    use_paper_style,
)
from src.figures._timing import draw_timing

__all__ = [
    "DOUBLE_WIDTH",
    "FAMILY_STYLE",
    "HORIZON_LINESTYLE",
    "INTERVAL_COLOUR",
    "MARKET_ORDER",
    "MARK_COLOUR",
    "MIN_FONT_SIZE",
    "MIN_LINE_WIDTH",
    "MODEL_FAMILY",
    "ORIGINAL_LAYOUT",
    "PAPER_LAYOUT",
    "PAPER_RC",
    "REFERENCE_LINE",
    "RQ2_MARKETS",
    "RQ2_MODELS",
    "SERIF_FONTS",
    "SINGLE_WIDTH",
    "SIZE_STYLE",
    "TERMES_FOLDER",
    "PowerLines",
    "check_figure",
    "draw_centre_scale",
    "draw_correlation",
    "draw_leakage",
    "draw_mde",
    "draw_power_summary",
    "draw_threshold_shares",
    "draw_timing",
    "empirical_shares",
    "model_style",
    "outcome_counts",
    "plain_tick",
    "read_centre_scale",
    "save_figure",
    "serif_rc",
    "shade_rows",
    "top_legend",
    "use_paper_style",
]
