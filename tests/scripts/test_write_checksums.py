"""The data checksum script (scripts/write_checksums.py)."""

import hashlib
import runpy
from pathlib import Path
from typing import Any

import pytest

from src import LAB_DIR

SCRIPT = LAB_DIR / "scripts" / "write_checksums.py"


@pytest.fixture(name="script", scope="module")
def _script() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT), run_name="checksums")


def test_a_file_checksum_is_its_sha256_and_size(script: dict[str, Any], tmp_path: Path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,2\n" * 1000)
    relative, size, digest = script["file_checksum"](path, root=tmp_path)
    assert relative == "x.csv"
    assert size == path.stat().st_size
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_logs_are_left_out_and_paths_are_sorted(script: dict[str, Any], tmp_path: Path):
    (tmp_path / "b").mkdir()
    for name in ("b/z.pq", "a.csv", "download.log"):
        (tmp_path / name).write_text("x")
    files = script["data_files"](tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["a.csv", "b/z.pq"]
