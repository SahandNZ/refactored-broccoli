"""The paper's RQ4 figure (``paper/scripts/make_fig_rq4.py``) from the committed verdict."""

# pyright: reportUnknownMemberType=false
# (as in src.figures: matplotlib's stubs leave the container and collection methods partly Unknown)

import csv
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pytest
from PIL import Image

from src import LAB_DIR

if TYPE_CHECKING:
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

SCRIPT = LAB_DIR / "paper" / "scripts" / "make_fig_rq4.py"
VERDICT = LAB_DIR / "experiments/06-rq4-leakage/results/test-2026-11-18/tables/verdict.csv"


@pytest.fixture(name="figure_script", scope="module")
def _figure_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fig_rq4", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    # draw() sets its rcParams globally, as the owner's script does
    with matplotlib.rc_context():
        yield


def _source_rows() -> list[dict[str, str]]:
    with VERDICT.open(newline="", encoding="utf-8") as stream:
        return [
            row for row in csv.DictReader(stream)
            if row["test"] == "leakage" and row["horizon"] == "1"
            and row["first"] == "before" and row["second"] == "after"
        ]  # fmt: skip


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_estimates_are_the_verdict_values_times_100(
    figure_script: ModuleType,
) -> None:
    estimates = figure_script.load_estimates(VERDICT)
    rows = _source_rows()
    assert len(rows) == len(estimates) == 65
    for row in rows:
        expected = [100 * float(row[field]) for field in ("did", "low", "high")]
        assert estimates[row["model"], row["market"]] == pytest.approx(expected, abs=1e-12)


def test_the_drawn_marks_and_intervals_are_the_verdict_values(
    figure_script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept: list[Figure] = []
    monkeypatch.setattr(figure_script.plt, "close", kept.append)
    estimates = figure_script.load_estimates(VERDICT)
    figure_script.draw(estimates, tmp_path / "fig_rq4")
    axes = kept[0].axes
    assert len(axes) == 10
    panels = [figure_script.MODELS] * 5 + [figure_script.KRONOS] * 5
    for index, (axis, models) in enumerate(zip(axes, panels, strict=True)):
        market_id, market_name = figure_script.MARKETS[index % 5]
        assert axis.get_title() == market_name
        containers: list[ErrorbarContainer] = list(axis.containers)  # type: ignore[arg-type]
        assert len(containers) == len(models)
        for y, ((model_id, _), container) in enumerate(zip(models, containers, strict=True)):
            estimate, lower, upper = estimates[model_id, market_id]
            mark = container.lines[0]
            assert mark is not None
            assert np.asarray(mark.get_xdata()) == pytest.approx([estimate])
            assert np.asarray(mark.get_ydata()) == pytest.approx([y])
            (segment,) = container.lines[2][0].get_segments()
            assert np.asarray(segment) == pytest.approx(np.array([[lower, y], [upper, y]]))
    labels = [text.get_text() for text in axes[0].get_yticklabels()]
    assert labels == [label for _, label in figure_script.MODELS]
    assert axes[0].get_xlim() == pytest.approx((-2.15, 2.15))
    assert axes[5].get_xlim() == pytest.approx((-12.0, 12.0))


def test_the_figure_renders_at_its_size(figure_script: ModuleType, tmp_path: Path) -> None:
    figure_script.draw(figure_script.load_estimates(VERDICT), tmp_path / "fig_rq4")
    # drawn at 8.35 x 5.25 in, cropped to its content (0.04 in of padding) at 320 dpi
    assert Image.open(tmp_path / "fig_rq4.png").size == (2529, 1476)
    assert (tmp_path / "fig_rq4.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("repeat", "one scored result"),
        ("unscored", "one scored result"),
        ("outside", "Invalid estimate"),
        ("contrast", "Invalid estimate"),
        ("missing", "13 models"),
    ],
)
def test_a_broken_verdict_raises(
    figure_script: ModuleType, tmp_path: Path, change: str, message: str
) -> None:
    rows = _source_rows()
    first = dict(rows[0])
    if change == "repeat":
        rows.append(first)
    elif change == "unscored":
        rows[0] = first | {"status": "not scored: too few blocks"}
    elif change == "outside":
        rows[0] = first | {"low": str(float(first["did"]) + 0.1)}
    elif change == "contrast":
        rows[0] = first | {"skill_first": str(float(first["skill_first"]) + 0.1)}
    else:
        rows = rows[1:]
    with pytest.raises(ValueError, match=message):
        figure_script.load_estimates(_write(tmp_path / "verdict.csv", rows))


def test_an_interval_at_the_axis_limit_raises(figure_script: ModuleType, tmp_path: Path) -> None:
    estimates = dict(figure_script.load_estimates(VERDICT))
    estimates["chronos-2", "crypto"] = (0.0, -1.0, 2.15)
    with pytest.raises(ValueError, match="clip the interval for chronos-2, crypto"):
        figure_script.draw(estimates, tmp_path / "fig_rq4")
