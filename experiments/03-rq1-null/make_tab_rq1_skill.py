r"""Write the body of the paper's RQ1 skill table (tab:rq1) from a 03-rq1-null run.

Reads ``tables/skill.csv`` of the run (written by analyze.py) and writes the ``tabular`` of
tab:rq1: one row per zero-shot model (`src.benchmark.MODEL_NAMES`, release order; the late
Holm family below a rule), per market (`src.benchmark.MARKET_NAMES`)
the skill and the half-width of its 95 % interval ("<est> & $\pm$\,<h>", one decimal; the
interval is a symmetric t interval, h = (high - low) / 2), at the horizon of 03-rq1-null's
``mde_horizon``; first a reference row for the best baseline (skill 0 by definition), set off
by a line space; bold for the highest printed skill per market within each Holm family (ties
at the printed precision all bold; a display mark, not the RQ1 rule), "--" where it is not
tested; minus signs in math mode ("$-$"). The rule between the main and the late Holm
family follows the run's ``family`` column (the quarters rule), not a fixed model.

Usage: .venv/bin/python experiments/03-rq1-null/make_tab_rq1_skill.py
--run-id ID [--out PATH] (default: the run's ``tables/rq1_body.tex``; the paper's copy is
``paper/tables/generated/rq1_body.tex``).
"""

import argparse
import filecmp
import logging
import math
from pathlib import Path

import pandas as pd

from src import LAB_DIR
from src.benchmark import (
    FINANCE_MODELS,
    MARKET_NAMES,
    MODEL_NAMES,
    SCORED,
    dashed_rule,
    finance_last,
    null_rule,
    paper_order,
)
from src.experiment import load_study, study_runs

logger = logging.getLogger("03-rq1-null")

STUDY = Path(__file__).parent
# the cell prints the half-width of the 95 % interval, which needs a symmetric (t) interval
SYMMETRY_TOLERANCE = 1e-9


def format_skill(value: float) -> str:
    """Return a skill in % to one decimal, minus as ``$-$``; a value rounding to zero is ``0.0``.

    Python's formatting rounds the binary float, so a value printed as x.x5 may round either way.
    """
    text = f"{value:.1f}"
    if text == "-0.0":
        return "0.0"
    return text.replace("-", "$-$")


def best_markets(at_horizon: pd.DataFrame, models: list[str]) -> dict[str, set[str]]:
    """Return, per model of `models`, the markets where its printed skill is the highest.

    The highest is taken over the scored cells of `models` (one Holm family) in each market,
    on the printed value (`format_skill`), so a tie at the shown precision marks every tied
    model.

    Parameters
    ----------
    at_horizon : pd.DataFrame
        The rows of ``skill.csv`` at the table's horizon.
    models : list of str
        The models of one family.

    Returns
    -------
    dict of str to set of str
        The markets each model is bold in (empty for a model that is best nowhere).
    """
    scored = at_horizon[at_horizon["model"].isin(models) & (at_horizon["status"] == SCORED)]
    printed = scored.assign(
        shown=[float(format_skill(float(v)).replace("$-$", "-")) for v in scored["skill"]]
    )
    best: dict[str, set[str]] = {model: set() for model in models}
    for market, rows in printed.groupby("market"):
        top = rows["shown"].max()
        for model in rows.loc[rows["shown"] == top, "model"]:
            best[str(model)].add(str(market))
    return best


def skill_row(name: str, cells: pd.DataFrame, bold: set[str]) -> str:
    r"""Return one table row: the model's name, then skill and interval half-width per market.

    Parameters
    ----------
    name : str
        The model's name in the paper.
    cells : pd.DataFrame
        The model's rows of ``skill.csv`` at the table's horizon, indexed by market.
    bold : set of str
        The markets where the skill is printed bold (`best_markets`).

    Returns
    -------
    str
        The LaTeX row, ending with ``\\\\``.
    """
    parts = [name]
    for market in MARKET_NAMES:
        if market not in cells.index or cells.loc[market, "status"] != SCORED:
            parts += ["--", "--"]
            continue
        estimate, low, high = cells.loc[[market], ["skill", "skill_low", "skill_high"]].to_numpy(
            dtype=float
        )[0]
        if math.isnan(low) != math.isnan(high) or (
            not math.isnan(low) and abs((estimate - low) - (high - estimate)) > SYMMETRY_TOLERANCE
        ):
            raise ValueError(
                f"{name} {market}: the interval [{low}, {high}] is not symmetric around {estimate}"
            )
        skill = format_skill(float(estimate))
        if market in bold:
            skill = f"\\textbf{{{skill}}}"
        half = "--" if math.isnan(low) else f"$\\pm$\\,{(high - low) / 2:.1f}"
        parts += [skill, half]
    return " & ".join(parts) + " \\\\"


def skill_body(skill: pd.DataFrame, *, horizon: int, source: str) -> str:
    """Return the ``tabular`` of tab:rq1 from ``skill.csv``.

    Parameters
    ----------
    skill : pd.DataFrame
        The ``skill.csv`` table of a 03-rq1-null run.
    horizon : int
        The horizon of the table, in bars.
    source : str
        The results file named in the generated header.

    Returns
    -------
    str
        The LaTeX body, starting with the "Generated by" comment.
    """
    at_horizon = skill[skill["horizon"] == horizon]
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for label in MARKET_NAMES.values())
    rules = " ".join(f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(MARKET_NAMES)))
    columns = " & ".join(["Skill (\\%) & $\\pm$"] * len(MARKET_NAMES))
    lines = [
        f"% Generated by experiments/03-rq1-null/make_tab_rq1_skill.py from {source}.",
        "% Do not edit by hand.",
        "\\begin{tabular}{l" + " r@{\\,}l" * len(MARKET_NAMES) + "}",
        "\\toprule",
        f"& {header} \\\\",
        rules,
        f"Checkpoint & {columns} \\\\",
        "\\midrule",
        " & ".join(["Selected baseline"] + ["0.0 & (ref.)"] * len(MARKET_NAMES)) + " \\\\",
        "\\addlinespace",
    ]
    # the order of tab:rq1 (`paper_order`): the main family, then the late family below a rule
    # (only when a main model is above it), each in release order; the finance-pretrained
    # checkpoints (`finance_last`) close the main family below a dashed rule
    families = {
        str(model): "late" if (rows["family"] == "late").all() else "main"
        for model, rows in at_horizon.groupby("model")
    }
    bold: dict[str, set[str]] = {}
    for family in ("main", "late"):
        bold |= best_markets(at_horizon, [m for m, f in families.items() if f == family])
    late_started = finance_started = False
    width = 1 + 2 * len(MARKET_NAMES)
    for written, model in enumerate(finance_last(paper_order(families), families)):
        if families[model] == "late" and not late_started:
            if written:
                lines.append("\\midrule")
            late_started = True
        if model in FINANCE_MODELS and families[model] == "main" and not finance_started:
            lines.append(dashed_rule(width))
            finance_started = True
        lines.append(
            skill_row(
                MODEL_NAMES[model],
                at_horizon[at_horizon["model"] == model].set_index("market"),
                bold[model],
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def cited_source(committed: Path, source: Path) -> Path:
    """Return the skill table the header names: the committed copy when there is one.

    Parameters
    ----------
    committed : Path
        ``results/<run>/tables/skill.csv``, the copy in git.
    source : Path
        ``runs/<run>/tables/skill.csv``, the table read.

    Returns
    -------
    Path
        `committed` when it exists, else `source`.

    Raises
    ------
    ValueError
        If the committed copy differs from the table read.
    """
    if not committed.exists():
        return source
    if not filecmp.cmp(committed, source, shallow=False):
        raise ValueError(f"{committed} differs from {source}: run analyze.py --run-id again")
    return committed


def main() -> None:
    """Write the body of tab:rq1 from the run's skill table."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    folder = study_runs(STUDY) / arguments.run_id
    rule = null_rule(load_study(folder / "config.toml"))
    source = folder / "tables" / "skill.csv"
    named = cited_source(STUDY / "results" / arguments.run_id / "tables" / "skill.csv", source)
    body = skill_body(
        pd.read_csv(source), horizon=rule.mde_horizon, source=str(named.relative_to(LAB_DIR))
    )
    out = arguments.out or folder / "tables" / "rq1_body.tex"
    out.write_text(body)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
