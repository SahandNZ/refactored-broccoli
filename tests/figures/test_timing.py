# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from PIL import Image

from src.figures import draw_timing


def test_the_timing_figure_grows_with_the_models(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    timing = pd.DataFrame({
        "model": ["chronos:amazon/chronos-2", "timesfm:google/timesfm-2.5-200m-pytorch",
                  "t0:theforecastingcompany/t0-gamma", "kronos:kronos-small"],
        "predict_seconds": [1.5, 3.0, 0.4, 12.0], "peak_gpu_gb": [0.8, 1.6, 0.3, 2.2],
    })  # fmt: skip
    before = dict(matplotlib.rcParams)
    draw_timing(timing, pdf_path=tmp_path / "t.pdf", grey_path=tmp_path / "t_grey.png")
    assert page_size(tmp_path / "t.pdf") == pytest.approx((6.80, 0.6 + 0.25 * 4))
    assert Image.open(tmp_path / "t_grey.png").mode == "L"
    assert dict(matplotlib.rcParams) == before
    text = (tmp_path / "t.pdf").read_bytes()
    assert b"/CreationDate" not in text
