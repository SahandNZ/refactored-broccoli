# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.benchmark import threshold_shares
from src.figures import draw_mde, draw_threshold_shares

if TYPE_CHECKING:
    from matplotlib.figure import Figure

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def _mde(values, horizons, markets=None):
    # an mde table: NaN means never reached
    values = np.array(values, dtype=float)
    return pd.DataFrame({
        "model": [f"m{i}" for i in range(len(values))],
        "market": markets if markets is not None else ["crypto"] * len(values),
        "horizon": horizons, "mde_pct": values, "is_reached": ~np.isnan(values),
        "largest_pct": 10.0,
    })  # fmt: skip


@pytest.mark.parametrize("threshold", [2.0, float("nan")])
def test_the_figures_are_written_with_unreached_cells_and_without_a_threshold(
    tmp_path: Path, threshold: float, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    mde = _mde(
        [0.4, 1.0, np.nan, 3.0, np.nan, 7.0],
        [1, 1, 4, 4, 24, 24],
        ["crypto", "forex", "crypto", "stock", "forex", "stock"],
    )
    draw_mde(mde, threshold_pct=threshold, pdf_path=tmp_path / "mde.pdf",
             grey_path=tmp_path / "build" / "mde_grey.png")  # fmt: skip
    assert page_size(tmp_path / "mde.pdf") == pytest.approx((6.80, 2.30))
    assert Image.open(tmp_path / "build" / "mde_grey.png").mode == "L"
    shares = threshold_shares(mde, thresholds=GRID)
    draw_threshold_shares(shares, chosen_pct=threshold, pdf_path=tmp_path / "share.pdf",
                          grey_path=tmp_path / "share_grey.png")  # fmt: skip
    assert page_size(tmp_path / "share.pdf") == pytest.approx((3.15, 2.00))
    assert Image.open(tmp_path / "share_grey.png").mode == "L"


def test_one_panel_per_horizon_of_the_data(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    draw_mde(_mde([1.0, 2.0], [2, 48]), threshold_pct=2.0, pdf_path=tmp_path / "mde.pdf",
             grey_path=tmp_path / "mde.png")  # fmt: skip
    assert page_size(tmp_path / "mde.pdf") == pytest.approx((6.80, 2.30))


def test_a_horizon_without_a_line_style_is_refused(tmp_path: Path) -> None:
    shares = threshold_shares(_mde([1.0, 2.0], [1, 2]), thresholds=GRID)
    with pytest.raises(ValueError, match="horizon 2 has no line style"):
        draw_threshold_shares(shares, chosen_pct=1.0, pdf_path=tmp_path / "share.pdf",
                              grey_path=tmp_path / "share.png")  # fmt: skip
    assert not (tmp_path / "share.pdf").exists()


def test_drawing_leaves_the_settings_unchanged(tmp_path: Path) -> None:
    before = dict(matplotlib.rcParams)
    mde = _mde([0.4, np.nan], [1, 4])
    draw_mde(mde, threshold_pct=1.0, pdf_path=tmp_path / "a.pdf", grey_path=tmp_path / "a.png")
    shares = threshold_shares(mde, thresholds=GRID)
    draw_threshold_shares(shares, chosen_pct=1.0, pdf_path=tmp_path / "b.pdf",
                          grey_path=tmp_path / "b.png")  # fmt: skip
    assert dict(matplotlib.rcParams) == before


def test_a_rerun_gives_the_same_pdf(tmp_path: Path) -> None:
    mde = _mde([0.4, 1.0, np.nan], [1, 4, 24])
    for name in ("a", "b"):
        draw_mde(mde, threshold_pct=1.0, pdf_path=tmp_path / f"{name}.pdf",
                 grey_path=tmp_path / f"{name}.png")  # fmt: skip
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()


def test_the_judged_horizon_is_drawn_in_front_with_the_required_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[Figure] = []

    def keep(figure: "Figure", pdf_path: Path, grey_path: Path) -> None:
        drawn.append(figure)

    monkeypatch.setattr("src.figures._mde.save_figure", keep)
    mde = _mde([0.4, 1.0, np.nan, 3.0, 1.5, 7.0], [1, 1, 4, 4, 24, 24])
    shares = threshold_shares(mde, thresholds=GRID)
    draw_threshold_shares(shares, chosen_pct=2.0, required_share=0.8, judged_horizon=1,
                          pdf_path=tmp_path / "s.pdf", grey_path=tmp_path / "s.png")  # fmt: skip
    axis = drawn[0].axes[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    assert "all horizons" not in lines
    assert lines["1 bar"].get_color() == "black"
    assert lines["4 bars"].get_color() != "black"
    texts = {text.get_text() for text in axis.texts}
    # both 1-bar cells have an MDE at most 2 %: 100 %; the rule needs 80 %
    assert {"X = 2 %", "required: 80 %", "100 %"} <= texts
    with pytest.raises(ValueError, match="judged horizon"):
        draw_threshold_shares(
            shares, chosen_pct=2.0, judged_horizon=12, pdf_path=tmp_path / "s.pdf",
            grey_path=tmp_path / "s.png",
        )  # fmt: skip
