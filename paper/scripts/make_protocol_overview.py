"""Draw Figure 1 of the benchmark paper: one picture of the whole protocol.

A diagram, not drawn from a results table: every word in it comes from
paper/context/fig_protocol_overview.toml, whose facts are taken from
paper/sections/protocol.tex. The top row is the pipeline (data, forecasts, scores and
tests, trading) as numbered cards with tags naming the research questions each step answers; the
bottom row is a time axis with the asset selection window, the three periods and the day the
protocol is locked (an optional marker, not used in the paper).

The figure is an HTML page filled from the TOML file and printed by headless Chromium to a
vector PDF, 6.80 in wide. A script inside the page measures the laid-out text; the run stops
when a text does not fit in its card or in the figure, overlaps another text, or is below
9 pt, so text never overflows unnoticed. Chromium embeds an OpenType font with CFF outlines as
a Type 3 font, which the AAAI kit forbids, so the script first converts TeX Gyre Termes to
TrueType outlines (embedded as CIDFontType2), and it blanks the creation and change dates that
Chromium writes into the PDF, so a rerun gives the same file.

Writes paper/figures/fig_protocol_overview.pdf, a 600 dpi colour preview and a
greyscale preview in paper/scripts/out/, and the page and fonts it printed in
paper/build/.

Usage (from the lab root): .venv/bin/python paper/scripts/make_protocol_overview.py
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose, as in src/figures: fontTools leaves the glyph objects of a font and their
# draw methods unannotated, which no code here can fix; nothing else is silenced)

import html
import json
import logging
import math
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path
from string import Template
from typing import Any

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._g_l_y_f import table__g_l_y_f
from PIL import Image

from src import LAB_DIR
from src.figures import DOUBLE_WIDTH, MIN_FONT_SIZE, TERMES_FOLDER

logger = logging.getLogger(__name__)

PAPER_DIR = LAB_DIR / "paper"
CONTENT_PATH = PAPER_DIR / "context" / "fig_protocol_overview.toml"
PDF_PATH = PAPER_DIR / "figures" / "fig_protocol_overview.pdf"
PREVIEW_PATH = PAPER_DIR / "scripts" / "out" / "fig_protocol_overview.png"
GREY_PREVIEW_PATH = PAPER_DIR / "scripts" / "out" / "fig_protocol_overview_grey.png"
PAGE_PATH = PAPER_DIR / "build" / "fig_protocol_overview.html"
FONT_FOLDER = PAPER_DIR / "build" / "fonts"

CHROMIUM_PATTERN = (
    ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/"
    "chrome-headless-shell"
)
"""Where Playwright installs headless Chromium, relative to the home folder."""

CHROMIUM_OPTIONS = ("--headless", "--no-sandbox", "--disable-gpu", "--font-render-hinting=none")
"""Options of every Chromium run. Without hinting, text has the same width at every scale, so
the fit check, the PDF and the 600 dpi preview get the same line breaks (with hinting the
check's text was about 7 % wider and broke lines the PDF did not)."""

PREVIEW_DPI = 600

TRUETYPE_MAXP = {
    "maxZones": 1,
    "maxTwilightPoints": 0,
    "maxStorage": 0,
    "maxFunctionDefs": 0,
    "maxInstructionDefs": 0,
    "maxStackElements": 0,
    "maxSizeOfInstructions": 0,
}
"""Fields of the maxp table that a TrueType font has and a CFF font lacks (no hinting code)."""

PAGE_SLACK = 0.02
"""Inches of the page height kept free under the figure's content.

Chromium rounds the page size to a grid of 1/300 in, sometimes down by up to 1.2 pt; content
filling the whole page would then spill onto a second page. The width is rounded too (6.80 in
becomes 6.807 in); `finish_pdf` sets it back to exactly `DOUBLE_WIDTH`.
"""

STAGE_COLOURS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
"""Okabe-Ito accent of each card, left to right: blue, bluish green, orange, reddish purple.

Colour is only an accent (top band, badge ring, tag outline, bullet dots); all text is black on
white, as the AAAI kit asks, so the figure reads the same in greyscale.
"""

PERIOD_COLOUR = "#0072B2"
"""Colour of the development and test bars; the pre-release bar is grey hatching."""

GREY = "#555555"
INK = "#1A1A1A"

AXIS_MARGIN = 1.0
"""Space in points between the page edge and each end of the time axis (room for the end ticks)."""

# vertical layout of the time axis in points, from the top of its strip
SELECTION_Y = 5.5
BAR_TOP = 12.5
BAR_HEIGHT = 7.0
AXIS_Y = 21.5
TICK_LENGTH = 2.5
YEAR_Y = 28.5
TIMELINE_HEIGHT = 33.0

CSS = """
@font-face { font-family: "Termes"; font-weight: normal; src: url("fonts/$regular_font"); }
@font-face { font-family: "Termes"; font-weight: bold; src: url("fonts/$bold_font"); }
@page { size: ${width}in ${height}in; margin: 0; }
html, body { margin: 0; padding: 0; }
body {
  width: ${width}in; height: ${body_height}in; overflow: hidden; position: relative;
  background: white; color: $ink; font-family: "Termes"; font-size: ${font_size}pt;
  line-height: 1.1;
}
.pipeline { display: flex; align-items: stretch; margin-top: 1pt; }
.stage { flex: 1 1 auto; display: flex; flex-direction: column; }
.tags {
  height: 11pt; margin-bottom: 2pt; display: flex; justify-content: flex-end;
  align-items: flex-end; gap: 3pt;
}
.tag {
  border: 0.75pt solid var(--accent); border-radius: 6pt; padding: 0 3.5pt;
  line-height: 10.5pt; white-space: nowrap;
}
.card {
  flex: 1; border: 0.6pt solid #A8A8A8; border-top: 2pt solid var(--accent);
  border-radius: 4pt; padding: 2pt 3pt 2.3pt 3pt;
}
.title { display: flex; align-items: center; gap: 3.5pt; margin-bottom: 1.5pt; }
.badge {
  flex: none; width: 10pt; height: 10pt; border: 0.8pt solid var(--accent); border-radius: 50%;
  display: flex; align-items: center; justify-content: center; font-weight: bold; line-height: 1;
}
.title-text { font-weight: bold; white-space: nowrap; }
ul { list-style: none; margin: 0; padding: 0; }
li { position: relative; padding-left: 4pt; margin-top: 1pt; white-space: nowrap; }
li::before {
  /* the font's bullet glyph, not a CSS circle: a 2.6 pt circle is snapped to device pixels
     differently on each line and prints as a dot on some lines and an oval on others */
  content: "\\2022"; position: absolute; left: 0; top: 0; color: var(--accent);
}
.arrow {
  flex: none; width: 10pt; padding-top: 14pt; display: flex; align-items: center;
  justify-content: center;
}
.timeline { position: relative; height: ${timeline_height}pt; margin-top: 4.5pt; }
.timeline svg { position: absolute; left: 0; top: 0; }
.placed { position: absolute; white-space: nowrap; line-height: 1.1; }
.centred { transform: translate(-50%, -50%); }
.right-of { transform: translate(0, -50%); }
.on-line { background: white; padding: 0 3pt; }
.year { color: #333333; }
.key {
  display: flex; justify-content: space-between; margin-top: 2pt; padding: 0 ${axis_margin}pt;
}
.key-item { flex: none; display: flex; gap: 4pt; align-items: flex-start; }
.key-item svg { flex: none; margin-top: 2.5pt; }
.key-line { white-space: nowrap; }
.key-line:first-child { font-weight: bold; }
"""

FIT_CHECK_SCRIPT = """
function textBox(element) {
  const range = document.createRange();
  range.selectNodeContents(element);
  const box = range.getBoundingClientRect();
  return {left: box.left, right: box.right, top: box.top + 2, bottom: box.bottom - 2};
}
function isInside(inner, outer) {
  return inner.left >= outer.left - 0.5 && inner.right <= outer.right + 0.5
    && inner.top >= outer.top - 0.5 && inner.bottom <= outer.bottom + 0.5;
}
function isOverlapping(first, second) {
  return first.left < second.right && second.left < first.right
    && first.top < second.bottom && second.top < first.bottom;
}
function checkFit() {
  const problems = [];
  const texts = Array.from(document.querySelectorAll(".text"));
  for (const element of texts) {
    const frame = element.closest(".frame").getBoundingClientRect();
    if (!isInside(textBox(element), frame)) {
      problems.push("does not fit in its box: " + element.textContent);
    }
    if (parseFloat(getComputedStyle(element).fontSize) < MIN_PIXELS) {
      problems.push("smaller than the smallest allowed text: " + element.textContent);
    }
  }
  for (const placed of document.querySelectorAll(".placed")) {
    for (const other of texts) {
      if (other !== placed && isOverlapping(textBox(placed), textBox(other))) {
        problems.push("overlaps: " + placed.textContent + " / " + other.textContent);
      }
    }
  }
  document.getElementById("fit-report").textContent = JSON.stringify(problems);
}
document.fonts.ready.then(checkFit);
"""
"""Runs in the page: lists every text that leaves its box, overlaps placed text, or is too small.

A text's box is its nearest element of class ``frame`` (a card, a stage column or the page).
Distances are in CSS pixels (0.75 pt). The box of a text is the box of its glyphs less 2 px at
the top and the bottom: the ascent and descent of the font reach about 1.5 pt past the line the
text sits on, where no letter is drawn.
The list is written into the hidden element ``fit-report``, which `check_fit` reads back.
"""

PAGE = Template("""<!doctype html>
<html><head><meta charset="utf-8">
<style>$css</style>
</head>
<body class="frame">
<div class="pipeline">$pipeline</div>
<div class="timeline">$timeline</div>
<div class="key">$key</div>
<pre id="fit-report" hidden>pending</pre>
<script>const MIN_PIXELS = $min_pixels;$script</script>
</body></html>
""")

ARROW_SVG = (
    '<svg width="9pt" height="7pt" viewBox="0 0 9 7">'
    f'<line x1="0.5" y1="3.5" x2="5.5" y2="3.5" stroke="{GREY}" stroke-width="1"/>'
    f'<polygon points="4.5,0.8 9,3.5 4.5,6.2" fill="{GREY}"/></svg>'
)

LOCK_SVG = (
    '<path d="M -2 0 V -2.2 A 2 2 0 0 1 2 -2.2 V 0" fill="none" stroke="#333333" '
    'stroke-width="0.9"/>'
    '<rect x="-3.4" y="0" width="6.8" height="5" rx="0.8" fill="#333333"/>'
)
"""A small padlock drawn around (0, 0), the top of its body; moved into place with translate."""


def printed(text: str) -> str:
    """Return a text of the TOML file as HTML, with ``<`` and ``&`` escaped.

    ``--`` becomes an en dash, and every hyphen a non-breaking hyphen, so a line never breaks
    inside ``USD-M`` or ``2020-01``.
    """
    return html.escape(text.replace("--", "\N{EN DASH}").replace("-", "\N{NON-BREAKING HYPHEN}"))


def stage_html(stage: dict[str, Any], number: int, colour: str) -> str:
    """Return one stage of the pipeline: its question tags above a numbered card of bullets."""
    tags = "".join(f'<span class="tag text">{printed(tag)}</span>' for tag in stage["questions"])
    bullets = "".join(f'<li class="text">{printed(bullet)}</li>' for bullet in stage["bullets"])
    # each card starts at the width of its widest line (bullets do not wrap) and the cards
    # share what is left of the row equally
    return (
        f'<div class="stage frame" style="--accent: {colour}">'
        f'<div class="tags">{tags}</div>'
        '<div class="card frame">'
        f'<div class="title"><span class="badge text">{number}</span>'
        f'<span class="title-text text">{printed(stage["title"])}</span></div>'
        f"<ul>{bullets}</ul></div></div>"
    )


def pipeline_html(stages: list[dict[str, Any]]) -> str:
    """Return the cards of all stages, left to right, with an arrow between two cards."""
    if len(stages) > len(STAGE_COLOURS):
        raise ValueError(f"at most {len(STAGE_COLOURS)} stages fit, got {len(stages)}")
    parts: list[str] = []
    for index, stage in enumerate(stages):
        if index > 0:
            parts.append(f'<div class="arrow">{ARROW_SVG}</div>')
        parts.append(stage_html(stage, index + 1, STAGE_COLOURS[index]))
    return "".join(parts)


def bar_svg(style: str, left: float, top: float, width: float, height: float, name: str) -> str:
    """Return the SVG of one period bar in its style; coordinates in points.

    ``hatched`` is grey diagonal hatching in a grey outline, ``outlined`` a light bar with a
    coloured outline and ``solid`` a full coloured bar (the strongest). ``name`` must be unique
    in the page: it names the clipping area of the hatching.
    """
    outline = f'x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}"'
    if style == "outlined":
        return f'<rect {outline} fill="#DCEAF5" stroke="{PERIOD_COLOUR}" stroke-width="0.9"/>'
    if style == "solid":
        return (
            f'<rect {outline} fill="{PERIOD_COLOUR}" stroke="{PERIOD_COLOUR}" stroke-width="0.9"/>'
        )
    if style != "hatched":
        raise ValueError(f'style must be "hatched", "outlined" or "solid", got {style!r}')
    # diagonal lines every 2.6 pt, cut to the bar by a clip path; the first line starts one bar
    # height left of the bar, so the top-left corner is hatched too
    lines: list[str] = []
    for step in range(math.ceil((width + height) / 2.6) + 1):
        start = left - height + step * 2.6
        lines.append(
            f'<line x1="{start:.2f}" y1="{top + height:.2f}" x2="{start + height:.2f}" '
            f'y2="{top:.2f}"/>'
        )
    return (
        f'<clipPath id="{name}"><rect {outline}/></clipPath>'
        f'<g clip-path="url(#{name})" stroke="#777777" stroke-width="0.6">{"".join(lines)}</g>'
        f'<rect {outline} fill="none" stroke="{GREY}" stroke-width="0.6"/>'
    )


def x_of(day_number: int, timeline: dict[str, Any]) -> float:
    """Return the x position in points of the start of a day (a `date.toordinal` number)."""
    first = timeline["start"].toordinal()
    after_last = timeline["end"].toordinal() + 1
    if not first <= day_number <= after_last:
        shown = date.fromordinal(day_number)
        raise ValueError(f"{shown} is outside the time axis {timeline['start']}..{timeline['end']}")
    share = (day_number - first) / (after_last - first)
    return AXIS_MARGIN + share * (DOUBLE_WIDTH * 72 - 2 * AXIS_MARGIN)


def span_of(item: dict[str, Any], timeline: dict[str, Any]) -> tuple[float, float]:
    """Return the left and right x in points of an item with a first and last day."""
    if item["start"] > item["end"]:
        raise ValueError(f"start {item['start']} is after end {item['end']}")
    left = x_of(item["start"].toordinal(), timeline)
    return left, x_of(item["end"].toordinal() + 1, timeline)


def axis_svg(timeline: dict[str, Any]) -> tuple[str, str]:
    """Return the SVG of the axis line and its year ticks, and the HTML of the year labels."""
    start, end = timeline["start"], timeline["end"]
    after_end = end.toordinal() + 1
    left, right = x_of(start.toordinal(), timeline), x_of(after_end, timeline)
    shapes = [f'<line x1="{left}" y1="{AXIS_Y}" x2="{right}" y2="{AXIS_Y}"/>']
    labels: list[str] = []
    for year in range(start.year, end.year + 1):
        year_left = x_of(max(date(year, 1, 1), start).toordinal(), timeline)
        year_right = x_of(min(date(year + 1, 1, 1).toordinal(), after_end), timeline)
        for tick in (year_left, year_right):
            shapes.append(
                f'<line x1="{tick}" y1="{AXIS_Y}" x2="{tick}" y2="{AXIS_Y + TICK_LENGTH}"/>'
            )
        centre = (year_left + year_right) / 2
        labels.append(
            f'<span class="placed centred year text" style="left: {centre:.2f}pt; '
            f'top: {YEAR_Y}pt">{year}</span>'
        )
    svg = f'<g stroke="#333333" stroke-width="0.6">{"".join(shapes)}</g>'
    return svg, "".join(labels)


def selection_svg(timeline: dict[str, Any]) -> tuple[str, str]:
    """Return the SVG of the selection window (a line with end ticks) and its label on the line."""
    selection = timeline["selection"]
    left, right = span_of(selection, timeline)
    svg = (
        f'<g stroke="{GREY}" stroke-width="0.75">'
        f'<line x1="{left}" y1="{SELECTION_Y}" x2="{right}" y2="{SELECTION_Y}"/>'
        f'<line x1="{left}" y1="{SELECTION_Y - 3}" x2="{left}" y2="{SELECTION_Y + 3}"/>'
        f'<line x1="{right}" y1="{SELECTION_Y - 3}" x2="{right}" y2="{SELECTION_Y + 3}"/></g>'
    )
    label = (
        f'<span class="placed centred on-line text" style="left: {(left + right) / 2:.2f}pt; '
        f'top: {SELECTION_Y}pt">{printed(selection["label"])}</span>'
    )
    return svg, label


def lock_svg(timeline: dict[str, Any]) -> tuple[str, str]:
    """Return the SVG of the lock (a padlock over a vertical rule) and its label to the right.

    Empty strings when the timeline has no ``lock`` table.
    """
    lock = timeline.get("lock")
    if lock is None:
        return "", ""
    x = x_of(lock["date"].toordinal(), timeline)
    svg = (
        f'<line x1="{x:.2f}" y1="{SELECTION_Y + 5}" x2="{x:.2f}" y2="{AXIS_Y + TICK_LENGTH}" '
        'stroke="#333333" stroke-width="1"/>'
        f'<g transform="translate({x:.2f} {SELECTION_Y - 1.5})">{LOCK_SVG}</g>'
    )
    label = (
        f'<span class="placed right-of text" style="left: {x + 6:.2f}pt; '
        f'top: {SELECTION_Y}pt">{printed(lock["label"])}</span>'
    )
    return svg, label


def timeline_html(timeline: dict[str, Any], periods: list[dict[str, Any]]) -> str:
    """Return the time axis: selection window, period bars, lock and year labels."""
    bars: list[str] = []
    for index, period in enumerate(periods):
        left, right = span_of(period, timeline)
        name = f"bar-clip-{index}"
        bars.append(bar_svg(period["style"], left, BAR_TOP, right - left, BAR_HEIGHT, name))
    axis, years = axis_svg(timeline)
    selection, selection_label = selection_svg(timeline)
    lock, lock_label = lock_svg(timeline)
    width = DOUBLE_WIDTH * 72
    svg = (
        f'<svg width="{width}pt" height="{TIMELINE_HEIGHT}pt" '
        f'viewBox="0 0 {width} {TIMELINE_HEIGHT}">'
        f"{''.join(bars)}{axis}{selection}{lock}</svg>"
    )
    return svg + selection_label + lock_label + years


def key_html(periods: list[dict[str, Any]]) -> str:
    """Return the key under the axis: per period a swatch in its bar style and its lines."""
    items: list[str] = []
    for index, period in enumerate(periods):
        swatch = bar_svg(period["style"], 0.5, 0.5, 19, 6.5, f"key-clip-{index}")
        lines = "".join(
            f'<div class="key-line text">{printed(line)}</div>' for line in period["lines"]
        )
        items.append(
            '<div class="key-item">'
            f'<svg width="20pt" height="7.5pt" viewBox="0 0 20 7.5">{swatch}</svg>'
            f"<div>{lines}</div></div>"
        )
    return "".join(items)


def page_html(content: dict[str, Any], regular_font: Path, bold_font: Path) -> str:
    """Return the whole figure as one HTML page, sized to the printed figure."""
    height = content["figure"]["height"]
    css = Template(CSS).substitute(
        regular_font=regular_font.name,
        bold_font=bold_font.name,
        width=DOUBLE_WIDTH,
        height=height,
        body_height=height - PAGE_SLACK,
        ink=INK,
        font_size=MIN_FONT_SIZE,
        timeline_height=TIMELINE_HEIGHT,
        axis_margin=AXIS_MARGIN,
    )
    return PAGE.substitute(
        css=css,
        pipeline=pipeline_html(content["stages"]),
        timeline=timeline_html(content["timeline"], content["periods"]),
        key=key_html(content["periods"]),
        min_pixels=MIN_FONT_SIZE * 96 / 72,
        script=FIT_CHECK_SCRIPT,
    )


def write_truetype_font(source: Path, target: Path) -> None:
    """Write an OpenType font with CFF (cubic) outlines as a TrueType (quadratic) font.

    The recipe of fontTools' ``otf2ttf`` snippet: every glyph is redrawn with quadratic curves
    (error at most one font unit), the CFF table is replaced by ``glyf`` and ``loca``, and
    ``maxp`` gets its TrueType version (Termes' ``post`` table already has format 3, which
    TrueType allows). The time stamp of the source is kept, so
    the same source always gives the same file.
    """
    font = TTFont(source, recalcTimestamp=False)
    glyph_set = font.getGlyphSet()
    glyf = table__g_l_y_f()
    glyf.glyphOrder = font.getGlyphOrder()
    glyf.glyphs = {}
    for name in glyf.glyphOrder:
        pen = TTGlyphPen(None)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    del font["CFF "]
    # fontTools makes the fields of a table at run time from the table's format, so the new
    # fields are set by name; save() computes the glyph counts of maxp
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for field, value in TRUETYPE_MAXP.items():
        setattr(maxp, field, value)
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(target)


def find_chromium() -> Path:
    """Return the newest headless Chromium that Playwright installed for this user.

    Raises
    ------
    FileNotFoundError
        If none is installed.
    """
    candidates = sorted(Path.home().glob(CHROMIUM_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"no headless Chromium in ~/{CHROMIUM_PATTERN}; install it with "
            "`uvx playwright install chromium`"
        )
    return candidates[-1]


def run_chromium(chromium: Path, *options: str) -> str:
    """Run headless Chromium on the figure page with the given options; return what it prints."""
    command = [str(chromium), *CHROMIUM_OPTIONS, *options, PAGE_PATH.as_uri()]
    result = subprocess.run(  # noqa: S603 (a fixed program, arguments built here)
        command, check=True, capture_output=True, text=True
    )
    return result.stdout


def check_fit(chromium: Path) -> None:
    """Lay out the page and stop if any text leaves its box, overlaps or is too small.

    Raises
    ------
    RuntimeError
        If the check inside the page did not run.
    ValueError
        Naming every text that does not fit, with what to change.
    """
    dom = run_chromium(chromium, "--virtual-time-budget=10000", "--dump-dom")
    match = re.search(r'<pre id="fit-report"[^>]*>(.*?)</pre>', dom, re.DOTALL)
    if match is None or match.group(1) == "pending":
        raise RuntimeError("the fit check inside the page did not run")
    problems = json.loads(html.unescape(match.group(1)))
    if problems:
        listed = "\n  ".join(problems)
        raise ValueError(
            f"text does not fit in the figure:\n  {listed}\n"
            f"shorten the text or raise [figure] height in {CONTENT_PATH}"
        )


def overwrite(data: bytes, match: re.Match[bytes], text: bytes) -> bytes:
    """Return the data with a match replaced by a text padded with spaces to the same length.

    The length stays the same, so every byte offset in the PDF's cross-reference table stays
    valid.
    """
    if len(text) > len(match.group()):
        raise ValueError(f"{text!r} is longer than {match.group()!r}")
    return data[: match.start()] + text.ljust(len(match.group())) + data[match.end() :]


def finish_pdf(pdf_path: Path) -> None:
    """Remove the dates Chromium writes, set the width to 6.80 in, and check there is one page.

    Without dates a rerun gives the same file. Chromium rounds the page width to 6.807 in; the
    content is 6.80 in wide and starts at the left edge, so setting the width of the page box
    cuts only empty space.

    Raises
    ------
    ValueError
        If the PDF has more than one page.
    """
    data = pdf_path.read_bytes()
    pages = len(re.findall(rb"/Type /Page\b", data))
    if pages != 1:
        raise ValueError(f"the figure must be one page, got {pages}; raise [figure] height")
    for match in re.finditer(rb"/(CreationDate|ModDate) \(D:[^)]*\)", data):
        data = overwrite(data, match, b"")
    box = re.search(rb"/MediaBox \[0 0 [\d.]+ ([\d.]+)\]", data)
    if box is None:
        raise ValueError(f"no page box found in {pdf_path}")
    width = f"{DOUBLE_WIDTH * 72:.1f}".encode()
    data = overwrite(data, box, b"/MediaBox [0 0 " + width + b" " + box.group(1) + b"]")
    pdf_path.write_bytes(data)


def write_previews(chromium: Path, height: float) -> None:
    """Write the colour preview at 600 dpi and its greyscale copy."""
    window = f"--window-size={math.ceil(DOUBLE_WIDTH * 96)},{math.ceil(height * 96)}"
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    scale = f"--force-device-scale-factor={PREVIEW_DPI / 96}"
    run_chromium(chromium, window, scale, "--hide-scrollbars", f"--screenshot={PREVIEW_PATH}")
    Image.open(PREVIEW_PATH).convert("L").save(GREY_PREVIEW_PATH)


def main() -> None:
    """Fill the page from the TOML file, check that the text fits, and write PDF and previews."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    with CONTENT_PATH.open("rb") as file:
        content = tomllib.load(file)
    FONT_FOLDER.mkdir(parents=True, exist_ok=True)
    regular_font = FONT_FOLDER / "texgyretermes-regular.ttf"
    bold_font = FONT_FOLDER / "texgyretermes-bold.ttf"
    write_truetype_font(TERMES_FOLDER / "texgyretermes-regular.otf", regular_font)
    write_truetype_font(TERMES_FOLDER / "texgyretermes-bold.otf", bold_font)
    PAGE_PATH.write_text(page_html(content, regular_font, bold_font), encoding="utf-8")
    chromium = find_chromium()
    check_fit(chromium)
    run_chromium(chromium, "--no-pdf-header-footer", f"--print-to-pdf={PDF_PATH}")
    finish_pdf(PDF_PATH)
    write_previews(chromium, content["figure"]["height"])
    logger.info("wrote %s, %s and %s", PDF_PATH, PREVIEW_PATH, GREY_PREVIEW_PATH)


if __name__ == "__main__":
    main()
