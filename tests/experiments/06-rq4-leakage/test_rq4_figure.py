"""rq4: the table of fig:rq4 and the row order of its models (tab:rq1's, as fig:rq2)."""

from datetime import timedelta
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.benchmark import SCORED
from src.evaluate import TEST_FOLDS

DAY = timedelta(days=1)
"""One calendar day (a datetime.timedelta: numpy warns on pandas' here)."""


def _verdict() -> pd.DataFrame:
    # the leakage and hourly cells of two models in two markets at 1 and 4 bars
    rows: list[dict[str, object]] = []
    for test in ("leakage", "hourly"):
        for model in ("t0-gamma", "timesfm-3"):
            for market in ("stock", "crypto"):
                for horizon in (1, 4):
                    scored = not (model == "t0-gamma" and market == "stock")
                    rows.append({
                        "test": test, "model": model, "market": market, "horizon": horizon,
                        "status": SCORED if scored else "not scored: too few blocks",
                        "first_times": 100, "second_times": 50, "first_block": 6,
                        "second_block": 6, "did": 0.01, "low": -0.01, "high": 0.03,
                        "adjusted_p": 0.5, "shows_leakage": False,
                    })  # fmt: skip
    return pd.DataFrame(rows)


def test_the_figure_table_keeps_the_leakage_cells_of_its_horizon(leakage: ModuleType) -> None:
    families = {"timesfm-3": "late", "t0-gamma": "late"}
    table = leakage.figure_table(_verdict(), families, horizon=1)
    assert list(table.columns) == leakage.FIGURE_COLUMNS
    assert len(table) == 4
    assert set(table["horizon"]) == {1}
    # markets (cache keys) in the paper's order (crypto before stocks), the family of each model
    assert table["market"].tolist() == ["crypto", "stock", "crypto", "stock"]
    assert table.set_index("model")["row_group"].to_dict() == families
    # models in tab:rq1's order: release order within a family (TimesFM 3 before t0, not
    # alphabetical), the main family first
    assert table["model"].tolist() == ["timesfm-3"] * 2 + ["t0-gamma"] * 2
    main_first = leakage.figure_table(
        _verdict(), {"timesfm-3": "late", "t0-gamma": "main"}, horizon=1
    )
    assert main_first["model"].tolist() == ["t0-gamma"] * 2 + ["timesfm-3"] * 2
    # a cell not scored keeps its row and status without a mark
    unscored = table[(table["model"] == "t0-gamma") & (table["market"] == "stock")]
    assert unscored[["did", "low", "high"]].isna().all(axis=None)
    assert unscored["status"].iloc[0] != SCORED


def test_the_figure_table_refuses_a_missing_horizon_market_or_family(
    leakage: ModuleType,
) -> None:
    families = {"timesfm-3": "late", "t0-gamma": "late"}
    with pytest.raises(ValueError, match="no leakage cell at horizon 24"):
        leakage.figure_table(_verdict(), families, horizon=24)
    other = _verdict().assign(market="moon")
    with pytest.raises(ValueError, match=r"markets \['moon'\] have no label"):
        leakage.figure_table(other, families, horizon=1)
    with pytest.raises(ValueError, match=r"models \['t0-gamma'\] have no family"):
        leakage.figure_table(_verdict(), {"timesfm-3": "late"}, horizon=1)


def test_a_model_with_fewer_test_quarters_after_its_release_is_late(
    leakage: ModuleType,
) -> None:
    starts = [fold.evaluation_start for fold in TEST_FOLDS]
    models = pd.DataFrame(
        {
            "model": ["climatology", "early", "second", "third"],
            # a baseline; released before t1; inside t1 (t2-t4 after); inside t2 (t3, t4)
            "release_date": [
                np.nan,
                str(starts[0].date() - DAY),
                str(starts[0].date() + DAY),
                str(starts[1].date() + DAY),
            ],
        }
    )
    families = leakage.figure_families(models, TEST_FOLDS, min_quarters=3)
    assert families == {
        "climatology": "main",
        "early": "main",
        "second": "main",
        "third": "late",
    }


def test_the_figure_limit_covers_marks_and_narrow_models_and_cuts_a_wide_one(
    leakage: ModuleType,
) -> None:
    # three narrow models (widest interval 0.04) and a wide one (0.4, more than 3 x 0.04)
    table = pd.DataFrame(
        {
            "model": ["a", "b", "c", "wide", "wide"],
            "did": [0.01, -0.01, 0.0, 0.05, np.nan],
            "low": [-0.01, -0.03, -0.02, -0.17, np.nan],
            "high": [0.03, 0.01, 0.02, 0.23, np.nan],
        }
    )
    # the wide model's mark (0.05) sets the reach, its interval (to 0.23) is cut
    assert leakage.figure_limit(table) == pytest.approx(leakage.LIMIT_MARGIN * 0.05)
    narrow = table[table["model"] != "wide"]
    assert leakage.figure_limit(narrow) == pytest.approx(leakage.LIMIT_MARGIN * 0.03)
    with pytest.raises(ValueError, match="no cell of the figure has a mark"):
        leakage.figure_limit(table.iloc[[4]])
