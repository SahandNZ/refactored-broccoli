"""The LaTeX bodies of the rq3 tables of the paper (``tab:rq3`` and its appendix table).

`break_even_body` writes the ``tabular`` of ``paper/tables/rq3.tex``: the break-even
cost per side of every zero-shot model per market, under the cost per side the market's cells
pay and the break-evens of the ridge regression and of buy and hold. `appendix_body` writes
the gross and net return, turnover and break-even cost with their intervals per model and
market. Both read only saved tables of a run, never its per-decision data;
``make_tab_rq3_break_even.py`` writes them. The names come from `src.benchmark.MARKET_NAMES`,
`src.benchmark.MODEL_NAMES` and `src.benchmark.REFERENCE_NAMES`.
"""

import math
from dataclasses import dataclass, fields, replace

import pandas as pd

from src.benchmark import (
    FINANCE_MODELS,
    MARKET_NAMES,
    MODEL_NAMES,
    REFERENCE_NAMES,
    SCORED,
    dashed_rule,
    finance_last,
)

from ._net import BUY_AND_HOLD
from ._verdict import MAIN

UNDEFINED = "n/a\\rlap{\\textsuperscript{a}}"
"""A break-even cost of a book whose mean gross return is negative (note a of the caption)."""

NO_INTERVAL = "n/a\\rlap{\\textsuperscript{b}}"
"""The break-even interval of a scored book with a draw without trades (note b)."""


def _number(value: float) -> str:
    # one decimal, a true minus, and a value that rounds to zero printed as 0.0
    text = f"{value:.1f}"
    if text == "-0.0":
        text = "0.0"
    return text.replace("-", "$-$")


def _turnover(value: float) -> str:
    # the mean absolute position change per book time, two decimals
    return f"{value:.2f}"


def _interval(low: float, high: float) -> str:
    # [low, high] of two finite bounds
    return f"[{_number(low)}, {_number(high)}]"


def _break_even_interval(cell: "_Cell") -> str:
    # empty after an n/a break-even (tables.md 9), note b when a draw had no trade
    if cell.break_even_bps < 0:
        return ""
    if math.isnan(cell.break_even_low) or math.isnan(cell.break_even_high):
        return NO_INTERVAL
    return _interval(cell.break_even_low, cell.break_even_high)


@dataclass(frozen=True, slots=True)
class _Cell:
    # the numbers of one model and market that the tables print
    family: str
    status: str
    earns_after_costs: bool
    net_mean: float
    net_low: float
    net_high: float
    gross_mean: float
    turnover: float
    break_even_bps: float
    break_even_low: float
    break_even_high: float


def _break_even(cell: _Cell) -> str:
    # the break-even cell: -- not scored, n/a for a negative gross return, bold if it earns
    if cell.status != SCORED:
        return "--"
    if cell.break_even_bps < 0:
        return UNDEFINED
    text = _number(cell.break_even_bps)
    return f"\\textbf{{{text}}}" if cell.earns_after_costs else text


def _best(rows: dict[tuple[str, str], _Cell], models: list[str], market: str) -> set[str]:
    # the models of `models` (one family) with the highest printed break-even in `market`,
    # over scored cells with a defined break-even (ties at the printed precision all count)
    shown = {
        model: float(_number(cell.break_even_bps).replace("$-$", "-"))
        for model in models
        if (cell := rows[(model, market)]).status == SCORED and cell.break_even_bps >= 0
    }
    top = max(shown.values(), default=None)
    return {model for model, value in shown.items() if value == top}


def _bold_break_even(cell: _Cell, bold: bool) -> str:
    # the break-even cell of tab:rq3: as `_break_even`, bold only when it is its family's best
    text = _break_even(replace(cell, earns_after_costs=False))
    return f"\\textbf{{{text}}}" if bold else text


def _rows(table: pd.DataFrame, models: list[str], horizon: int) -> dict[tuple[str, str], _Cell]:
    # the cells of `table` at `horizon` for `models`, by (model, market)
    chosen = table[(table["horizon"] == horizon) & table["model"].isin(models)]
    missing = sorted(set(models) - set(chosen["model"]))
    if missing:
        raise ValueError(f"no rows at horizon {horizon} for models {missing}")
    rows: dict[tuple[str, str], _Cell] = {}
    for record in chosen.to_dict("records"):
        values = {field.name: record[field.name] for field in fields(_Cell)}
        values["earns_after_costs"] = bool(values["earns_after_costs"])
        rows[(str(record["model"]), str(record["market"]))] = _Cell(**values)
    markets = set(chosen["market"])
    absent = sorted((model, market) for model in models for market in markets)
    absent = [pair for pair in absent if pair not in rows]
    if absent:
        raise ValueError(f"no rows at horizon {horizon} for (model, market) {absent}")
    return rows


def _name(model: str) -> str:
    # the paper's name of a model or of the buy-and-hold book
    return REFERENCE_NAMES[model] if model == BUY_AND_HOLD else MODEL_NAMES[model]


def _models(verdict: pd.DataFrame) -> list[str]:
    # the zero-shot models of the verdict: the main family, then the late-release family, each
    # in release order
    unknown = sorted(set(verdict["model"]) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(
            f"models {unknown} have no name in the paper, expected {list(MODEL_NAMES)}"
        )
    baselines = sorted(set(verdict.loc[verdict["kind"] != "zero-shot", "model"]))
    if baselines:
        raise ValueError(f"the verdict holds only zero-shot models, got baselines {baselines}")
    main = set(verdict.loc[verdict["family"] == MAIN, "model"])
    ordered = [name for name in MODEL_NAMES if name in set(verdict["model"])]
    return [name for name in ordered if name in main] + [
        name for name in ordered if name not in main
    ]


def _first_late(verdict: pd.DataFrame, models: list[str]) -> str | None:
    # the first model of `models` (main family first) in the late-release family, if any
    late = set(verdict.loc[verdict["family"] != MAIN, "model"])
    return next((name for name in models if name in late), None)


def _references(cells: pd.DataFrame, reference: str, horizon: int) -> dict[tuple[str, str], _Cell]:
    # the reference rows: the reference model (the ridge regression) and buy and hold (only
    # at horizon 1, so the tables are at horizon 1); never bold, in no family
    rows = cells.assign(family="reference", earns_after_costs=False)
    return _rows(rows, [reference, BUY_AND_HOLD], horizon)


def _cost_paid(verdict: pd.DataFrame, market: str, horizon: int) -> str:
    # the market's cost per side as its cells pay it: the mean of the scored zero-shot cells'
    # cost_bps at `horizon` (both families), weighted by their mean turnover per book time
    # (sum of gross minus net over the sum of turnover); -- without a scored cell
    cells = verdict[
        (verdict["market"] == market)
        & (verdict["horizon"] == horizon)
        & (verdict["status"] == SCORED)
    ]
    if cells.empty:
        return "--"
    paid = (cells["gross_mean"] - cells["net_mean"]).sum() / cells["turnover"].sum()
    return _number(paid * 1e4)


def _markets(cells: pd.DataFrame) -> dict[str, str]:
    # the markets of the run, in the order of `MARKET_NAMES`
    unknown = sorted(set(cells["market"]) - set(MARKET_NAMES))
    if unknown:
        raise ValueError(f"markets {unknown} have no column name, expected {list(MARKET_NAMES)}")
    return {market: name for market, name in MARKET_NAMES.items() if market in set(cells["market"])}


def _line(name: str, cells: list[str]) -> str:
    return " & ".join([name, *cells]) + " \\\\"


def _header(run: str, files: str) -> str:
    # the first line of a body: the script and the run's results files, from the lab root
    return (
        "% Generated by experiments/05-rq3-net/make_tab_rq3_break_even.py from "
        f"experiments/05-rq3-net/runs/{run}/tables/{files}. Do not edit by hand.\n"
    )


def break_even_body(
    verdict: pd.DataFrame, cells: pd.DataFrame, *, reference: str, horizon: int, run: str
) -> str:
    """Return the ``tabular`` of ``tab:rq3``: break-even costs per side in bps per market.

    The first row is the cost per side the market's cells pay (`_cost_paid`: the mean of the
    ``cost_bps`` of its scored zero-shot cells at `horizon`, both families, weighted by
    their mean turnover per book time), then the break-evens of the reference model (the
    ridge regression, from `cells`; buy and hold is left to the appendix tables), then the
    zero-shot models of `verdict` in release order, the main family first and the
    late-release family after a rule. A cell is bold when it is the highest printed break-even
    of its market within its family (ties all bold; the reference row never), ``--`` when it
    is not scored and ``n/a`` (note a) when its mean gross return is negative.

    Parameters
    ----------
    verdict : pd.DataFrame
        ``tables/verdict.csv`` of the run.
    cells : pd.DataFrame
        ``tables/cells.csv`` of the run (the reference rows).
    reference : str
        The model of the reference row, the ridge regression (``"linear"``).
    horizon : int
        The horizon of the table, in own bars.
    run : str
        The run id, for the header line.

    Returns
    -------
    str
        The LaTeX ``tabular``, with its first line naming the script and the run.

    Raises
    ------
    ValueError
        If a model of `verdict` has no name in the paper or is a baseline, a reference row or
        a model has no row at `horizon`, or a market has no column name.
    """
    models = _models(verdict)
    markets = _markets(cells)
    rows = _rows(verdict, models, horizon)
    references = _references(cells, reference, horizon)
    lines = [
        "\\begin{tabular}{l" + "r" * len(markets) + "}",
        "\\toprule",
        _line("(bps per side)", list(markets.values())),
        "\\midrule",
        _line("Mean cost per side", [_cost_paid(verdict, market, horizon) for market in markets]),
    ]
    # buy and hold stays in the appendix tables only: its break-even cost is not meaningful
    ridge = [_break_even(references[(reference, market)]) for market in markets]
    lines.append(_line(_name(reference), ridge))
    lines.append("\\addlinespace")
    # the finance-pretrained checkpoints close the main family below a dashed rule (display only)
    families = dict(zip(verdict["model"], verdict["family"], strict=False))
    models = finance_last(models, {model: str(families[model]) for model in models})
    first_late = _first_late(verdict, models)
    finance = [model for model in models if model in FINANCE_MODELS and families[model] == MAIN]
    best = {
        (model, market)
        for family in (MAIN, None)
        for market in markets
        for model in _best(
            rows,
            [m for m in models if (families[m] == MAIN) == (family == MAIN)],
            market,
        )
    }
    for model in models:
        if model == first_late:
            lines.append("\\midrule")
        if finance and model == finance[0]:
            lines.append(dashed_rule(1 + len(markets)))
        printed = [_bold_break_even(rows[(model, m)], (model, m) in best) for m in markets]
        lines.append(_line(_name(model), printed))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _header(run, "verdict.csv and cells.csv") + "\n".join(lines) + "\n"


def family_models(verdict: pd.DataFrame, family: str) -> list[str]:
    """Return the zero-shot models of `verdict` in one family, in release order.

    Parameters
    ----------
    verdict : pd.DataFrame
        ``tables/verdict.csv`` of the run.
    family : str
        ``"main"`` or ``"late"`` (`MAIN`, `LATE`).

    Returns
    -------
    list of str
        The model ids, in the order of `src.benchmark.MODEL_NAMES`.

    Raises
    ------
    ValueError
        As `break_even_body`.
    """
    members = set(verdict.loc[verdict["family"] == family, "model"])
    return [model for model in _models(verdict) if model in members]


def appendix_body(
    verdict: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    groups: list[list[str]],
    horizon: int,
    level: float,
    run: str,
) -> str:
    """Return a ``tabular`` of the rq3 appendix: gross, net, turnover and break-even.

    One row per model of `groups` and market, the groups in order with a rule between them:
    the mean gross return per book time in bps, the mean net return in bps with its interval,
    the mean turnover per book time and the break-even cost per side with its interval (the
    same block draws). A zero-shot model's rows come from `verdict` (bold when it earns), a
    reference model's (the ridge regression, buy and hold) from `cells` (never bold). A cell
    not scored is ``--``; the interval of an ``n/a`` break-even is empty, and one undefined
    because a draw had no trade is ``n/a`` with note b. The appendix is cut in two bodies
    (`make_tab_rq3_break_even.py`) so each fits a page.

    Parameters
    ----------
    verdict : pd.DataFrame
        ``tables/verdict.csv`` of the run.
    cells : pd.DataFrame
        ``tables/cells.csv`` of the run (the reference rows).
    groups : list of list of str
        The models of the body, in printed order, one list per group (an empty group is
        left out).
    horizon : int
        The horizon of the table, in own bars.
    level : float
        The coverage of the intervals, for the header (e.g. 0.95).
    run : str
        The run id, for the header line.

    Returns
    -------
    str
        The LaTeX ``tabular``, with its first line naming the script and the run.

    Raises
    ------
    ValueError
        As `break_even_body`, or if a model of `groups` has no rows at `horizon`.
    """
    markets = _markets(cells)
    named = set(_models(verdict))
    zero_shot = [model for group in groups for model in group if model in named]
    references = [model for group in groups for model in group if model not in named]
    reference_rows = cells.assign(family="reference", earns_after_costs=False)
    table = _rows(verdict, zero_shot, horizon) | _rows(reference_rows, references, horizon)
    interval = f"{level * 100:.0f}\\% CI"
    lines = [
        "\\begin{tabular}{llrr@{\\,}lrr@{\\,}l}",
        "\\toprule",
        f"Checkpoint & Market & Gross (bps) & Net (bps) & {interval} & Turnover & Break-even (bps)"
        f" & {interval} \\\\",
    ]
    for group in [group for group in groups if group]:
        lines.append("\\midrule")
        for model in group:
            for market, label in markets.items():
                lines.append(
                    _line(f"{_name(model)} & {label}", _appendix_cells(table[(model, market)]))
                )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _header(run, "verdict.csv and cells.csv") + "\n".join(lines) + "\n"


def _appendix_cells(cell: _Cell) -> list[str]:
    # the six numbers of one appendix row, or -- for a cell not scored
    if cell.status != SCORED:
        return ["--"] * 6
    return [
        _number(cell.gross_mean * 1e4),
        _number(cell.net_mean * 1e4),
        _interval(cell.net_low * 1e4, cell.net_high * 1e4),
        _turnover(cell.turnover),
        _break_even(cell),
        _break_even_interval(cell),
    ]
