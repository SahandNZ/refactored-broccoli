"""The paper's correlation figure (``paper/scripts/make_fig_correlation.py``) from 01-universe."""

import importlib.util
from itertools import groupby
from types import ModuleType

import pandas as pd
import pytest

from src import LAB_DIR

SCRIPT = LAB_DIR / "paper" / "scripts" / "make_fig_correlation.py"
RUN = LAB_DIR / "experiments/01-universe/results/2026-11-17T0005-random-1000"

OWNER_GROUPS = (
    ("Commodities", "XAUUSD COPPER BRENT NATGAS COFFEE SUGAR SOYBEAN COCOA"),
    ("Crypto", "BTCUSDT ETHUSDT DOGEUSDT XRPUSDT NEARUSDT FILUSDT EGLDUSDT RUNEUSDT"),
    ("FX", "EURUSD USDJPY AUDNZD EURGBP EURCHF GBPNZD CHFJPY USDCNH USDMXN EURCAD GBPCHF "
     "GBPAUD SGDJPY EURSGD AUDCAD EURJPY"),
    ("Indices", "USA500 CHIA50 PLN20 ESP35 HKG JPN225 CHE20 GBR100"),
    ("Non-US stocks", "MC.PA EQNR.OL 0700.HK GSK.L MAERSK-B.CO 7974.T HSBA.L NOVO-B.CO "
     "0388.HK ERIC-B.ST BATS.L 8031.T AD.AS 8035.T ROG.SW 2318.HK"),
    ("US stocks", "TSLA EOG LLY META ABT WMT WFC NFLX LMT AMGN AMZN CAT AMD NEE ORCL PM"),
)  # fmt: skip
"""The groups and order of the owner's script (2026-11-19), which the figure must follow."""


@pytest.fixture(name="figure_script", scope="module")
def _figure_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fig_correlation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rows_follow_the_owners_groups(figure_script: ModuleType) -> None:
    matrix, labels = figure_script.read_inputs(RUN)
    expected = [symbol for _, symbols in OWNER_GROUPS for symbol in symbols.split()]
    assert list(matrix.index) == expected
    assert list(matrix.columns) == expected
    assert [(name, len(list(rows))) for name, rows in groupby(labels)] == [
        (name, len(symbols.split())) for name, symbols in OWNER_GROUPS
    ]


def test_the_matrix_is_the_committed_csv(figure_script: ModuleType) -> None:
    matrix, _ = figure_script.read_inputs(RUN)
    source = pd.read_csv(RUN / "tables" / "universe_correlation.csv", index_col=0)
    reordered: pd.DataFrame = source.reindex(index=matrix.index, columns=matrix.columns)
    pd.testing.assert_frame_equal(matrix, reordered)
