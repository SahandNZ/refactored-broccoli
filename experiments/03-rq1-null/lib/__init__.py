"""The decision rule of rq1 (experiments/03-rq1-null, Q1) and the notes of its answers.

Modules import each other relatively, so the package works under any module name.
"""

from ._notes import null_note, size_note, whole_quarter_note
from ._verdict import null_verdict

__all__ = ["null_note", "null_verdict", "size_note", "whole_quarter_note"]
