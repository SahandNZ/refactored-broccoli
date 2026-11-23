"""The order of the paper's tables and figures (`src.benchmark.paper_order`)."""

import pytest

from src.benchmark import finance_last, paper_order


def test_the_late_family_follows_the_main_family_each_in_release_order() -> None:
    families = {
        "t0-gamma": "late",
        "chronos-2": "main",
        "toto-2-22m": "late",
        "chronos-bolt-small": "main",
        "climatology": "main",
    }
    assert paper_order(families) == [
        "climatology",
        "chronos-bolt-small",
        "chronos-2",
        "toto-2-22m",
        "t0-gamma",
    ]


def test_a_model_without_a_paper_name_follows_in_the_given_order() -> None:
    families = {"scale-only": "main", "centre-only": "main", "chronos-2": "main"}
    assert paper_order(families) == ["chronos-2", "scale-only", "centre-only"]


def test_a_family_other_than_main_or_late_raises() -> None:
    with pytest.raises(ValueError, match=r"families \['Late'\] are neither main nor late"):
        paper_order({"chronos-2": "main", "t0-gamma": "Late"})


def test_finance_models_close_the_main_family_and_the_late_family_follows() -> None:
    models = ["chronos-bolt-small", "kronos-small", "kronos-base", "chronos-2", "t0-gamma"]
    families = {"t0-gamma": "late"}
    assert finance_last(models, families) == [
        "chronos-bolt-small",
        "chronos-2",
        "kronos-small",
        "kronos-base",
        "t0-gamma",
    ]
