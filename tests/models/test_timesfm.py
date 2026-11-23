"""TimesFM's TF32 setting stays inside its own calls (`src.models._timesfm.TF32Precision`)."""

import pytest
import torch

from src.models._timesfm import TF32Precision


def test_tf32_is_set_inside_the_block_and_restored_after_it_and_after_an_error() -> None:
    before = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with TF32Precision():
            assert torch.get_float32_matmul_precision() == "high"
        assert torch.get_float32_matmul_precision() == "highest"
        with pytest.raises(RuntimeError, match="inside"), TF32Precision():
            raise RuntimeError("inside")
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(before)
