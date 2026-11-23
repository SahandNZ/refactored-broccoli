# pyright: reportUnknownMemberType=false

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from PIL import Image

from src.figures import draw_correlation
from src.figures._correlation import _market_blocks


def _matrix(size: int, seed: int = 0) -> pd.DataFrame:
    # the correlation of random returns of `size` assets
    returns = np.random.default_rng(seed).standard_normal((200, size))
    symbols = [f"A{i}" for i in range(size)]
    return pd.DataFrame(np.corrcoef(returns, rowvar=False), index=symbols, columns=symbols)


def test_the_heatmap_is_cropped_with_the_matrix_in_the_middle(
    tmp_path: Path, page_size: Callable[[Path], tuple[float, float]]
) -> None:
    markets = ["crypto"] * 16 + ["forex"] * 16 + ["stock"] * 16 + ["index"] * 8 + ["commodity"] * 5
    before = dict(matplotlib.rcParams)
    draw_correlation(_matrix(61), markets, pdf_path=tmp_path / "c.pdf",
                     grey_path=tmp_path / "c_grey.png", png_path=tmp_path / "c.png")  # fmt: skip
    width, height = page_size(tmp_path / "c.pdf")
    # cropped from the drawn size to its content, no wider than the paper's column (239.4 pt)
    assert 3.0 < width <= 239.4 / 72
    assert 1.9 < height < 2.40
    assert Image.open(tmp_path / "c_grey.png").mode == "L"
    assert Image.open(tmp_path / "c.png").size == pytest.approx((width * 360, height * 360), abs=2)
    assert dict(matplotlib.rcParams) == before


def test_the_drawn_cells_are_the_matrix_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[np.ndarray] = []
    imshow = Axes.imshow  # pyright: ignore[reportUnknownVariableType]

    def spy(self: Axes, data: np.ndarray, *args: object, **kwargs: object) -> object:
        drawn.append(np.asarray(data))
        return imshow(self, data, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(Axes, "imshow", spy)
    matrix = _matrix(4)
    draw_correlation(matrix, ["fx", "fx", "crypto", "crypto"], pdf_path=tmp_path / "c.pdf",
                     grey_path=tmp_path / "c_grey.png")  # fmt: skip
    assert len(drawn) == 1
    np.testing.assert_array_equal(drawn[0], matrix.to_numpy())


def test_a_matrix_that_is_not_a_correlation_matrix_is_refused(tmp_path: Path) -> None:
    matrix = _matrix(3)
    matrix.iloc[0, 1] = 0.9
    with pytest.raises(ValueError, match="symmetric correlation matrix with a unit diagonal"):
        draw_correlation(matrix, ["fx"] * 3, pdf_path=tmp_path / "c.pdf",
                         grey_path=tmp_path / "c.png")  # fmt: skip


def test_market_blocks_give_first_and_last_row_in_order() -> None:
    blocks = _market_blocks(["fx", "fx", "crypto", "stock", "stock", "stock"])
    assert blocks == [("fx", 0, 1), ("crypto", 2, 2), ("stock", 3, 5)]


def test_a_market_split_in_two_blocks_is_refused() -> None:
    with pytest.raises(ValueError, match="'fx' must be next to each other"):
        _market_blocks(["fx", "crypto", "fx"])


def test_a_matrix_with_other_columns_than_rows_is_refused(tmp_path: Path) -> None:
    matrix = _matrix(3)
    matrix.columns = ["A0", "A2", "A1"]
    with pytest.raises(ValueError, match="same assets as index and columns"):
        draw_correlation(matrix, ["fx"] * 3, pdf_path=tmp_path / "c.pdf",
                         grey_path=tmp_path / "c.png")  # fmt: skip


def test_one_market_per_asset_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="markets has 2 entries, expected 3"):
        draw_correlation(_matrix(3), ["fx"] * 2, pdf_path=tmp_path / "c.pdf",
                         grey_path=tmp_path / "c.png")  # fmt: skip
