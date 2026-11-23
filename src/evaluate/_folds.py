from dataclasses import dataclass

import pandas as pd

from src.utils import to_utc


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold (the folds are `DEVELOPMENT_FOLDS`, `TEST_FOLDS`, `LEAKAGE_FOLDS`).

    Attributes
    ----------
    name : str
        ``"d1"`` .. ``"d4"`` (development) or ``"t1"`` .. ``"t4"`` (test).
    train_start : pd.Timestamp
        First training time (UTC).
    evaluation_start, evaluation_end : pd.Timestamp
        The evaluation quarter, ``[start, end)``; training ends at ``evaluation_start``.
    """

    name: str
    train_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp

    @property
    def is_test(self) -> bool:
        """Whether this is a test fold (run once, for the paper's numbers)."""
        return self.name.startswith("t")


def _fold(name: str, start: str, end: str) -> Fold:
    return Fold(name, to_utc("2020-01-01"), to_utc(start), to_utc(end))


DEVELOPMENT_FOLDS = (
    _fold("d1", "2024-11-01", "2025-02-01"),
    _fold("d2", "2025-02-01", "2025-05-01"),
    _fold("d3", "2025-05-01", "2025-08-01"),
    _fold("d4", "2025-08-01", "2025-11-01"),
)
"""The folds every choice is made on."""

TEST_FOLDS = (
    _fold("t1", "2025-11-01", "2026-02-01"),
    _fold("t2", "2026-02-01", "2026-05-01"),
    _fold("t3", "2026-05-01", "2026-08-01"),
    _fold("t4", "2026-08-01", "2026-11-01"),
)
"""The folds run once, after every choice is fixed."""


LEAKAGE_FOLDS = tuple(
    _fold(
        f"l{index + 1}",
        str(start.date()),
        str(min(start + pd.DateOffset(months=3), DEVELOPMENT_FOLDS[0].evaluation_start).date()),
    )
    for index, start in enumerate(pd.date_range("2022-01-01", "2024-10-01", freq="QS", tz="UTC"))
)
"""Quarters of 2022-2024, before the foundation models' releases: the leakage periods of
06-rq4-leakage. Run only with the locked plan, like the test folds. The leakage
period ends where d1 starts, so l12 is one month (2024-10-01 .. 2024-11-01) and no decision
is in two windows."""
