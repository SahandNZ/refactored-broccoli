"""Time-series foundation models on crypto perpetual cross-sections."""

import os
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
"""The repository root (the `anonymous-benchmark` folder)."""

# keep model weights inside the lab unless the caller chose another cache
os.environ.setdefault("HF_HOME", str(LAB_DIR / ".hf"))

__all__ = ["LAB_DIR"]
