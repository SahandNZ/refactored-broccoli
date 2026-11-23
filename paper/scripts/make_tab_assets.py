"""Write the asset list of Appendix app:assets and the counts of its candidates from 01-universe.

Reads ``tables/universe.csv`` of the final 01-universe run, one row per candidate with its pool,
pick order, ``in_universe`` flag and the ``dropped`` reason of every candidate not in the list.
Writes into ``tables/generated/``: ``assets_list_body.tex`` (a whole one-column ``tabular``, the
listed assets of each pool in pick order) and ``assets_values.tex`` (count macros), both input by
``tables/assets.tex``. It also writes ``assets_filtered_body.tex`` and ``assets_unpicked_body.tex``,
every other candidate under the step that dropped it; the paper no longer inputs them (the
supplementary universe.csv holds the same), but writing them checks that every ``dropped`` reason
parses. Binance symbols lose their USDT suffix.
Run from the lab root: ``.venv/bin/python paper/scripts/make_tab_assets.py``.
"""

import logging
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

UNIVERSE = Path("experiments/01-universe/results/2026-11-17T0005-random-1000/tables/universe.csv")
"""The candidates of the final 01-universe run, as committed under the study's results."""

STUDY = Path("experiments/01-universe/study.toml")
"""The plan of 01-universe, whose ``[options]`` hold the limits the block labels name."""

DATA_CONFIG = Path("configs/data.toml")
"""The data configuration; its ``no_reference`` lists name the stocks without a Yahoo reference."""

OUT = Path("paper/tables/generated")
"""The folder of the generated table bodies."""

POOL_NAMES = {
    "crypto": "Crypto",
    "forex": "FX",
    "commodity": "Commodities",
    "index": "Indices",
    "stock_us": "US stocks",
    "stock_non_us": "Non-US stocks",
}
"""The pools in the order and with the names of the asset list (Appendix app:assets)."""

STOCK_POOLS = {"alpaca": "stock_us", "dukascopy": "stock_non_us"}
"""The stock pool of each stock source (US stocks come from Alpaca, non-US ones from Dukascopy)."""

LIST_WIDTH = "0.74\\linewidth"
"""The width of the symbol column of the one-column asset list (after the pool column)."""

DROPPED_WIDTH = "0.8\\linewidth"
"""The width of the symbol column of the dropped candidates (after the pool and count columns)."""


@dataclass(frozen=True, slots=True)
class Reason:
    """One step of the selection rule that drops candidates, as written in ``dropped``.

    Parameters
    ----------
    label : str
        The heading of the step's block in the table, a template formatted with `_limits`.
    pattern : str
        A regular expression that matches the whole ``dropped`` text of this step.
    detail : Callable[[re.Match[str]], str] or None
        The text shown in parentheses after a symbol, from the match; None shows none.
    """

    label: str
    pattern: str
    detail: Callable[[re.Match[str]], str] | None


def _quality_detail(match: re.Match[str]) -> str:
    # "coverage 0.654 < 0.95; gap 397.64 days > 10" -> "coverage 65.4\\%, gap 397.6 days"
    parts: list[str] = []
    for part in match.group(1).split("; "):
        coverage = re.fullmatch(r"coverage ([\d.]+) < [\d.]+", part)
        gap = re.fullmatch(r"gap ([\d.]+) days > [\d.]+", part)
        if coverage:
            parts.append(f"coverage {100 * float(coverage.group(1)):.1f}\\%")
        elif gap:
            parts.append(f"gap {float(gap.group(1)):.1f} days")
        else:
            raise ValueError(f"unknown quality failure {part!r} in {match.group(0)!r}")
    return ", ".join(parts)


FILTER_REASONS = (
    Reason("Pegged FX pair ({pegged}), dropped before selection", r"pegged currency [\w, ]+", None),
    Reason("No bar in the selection window", r"no bar in the selection window", None),
    Reason(
        "First bar too late (date of the first bar)",
        r"starts (\d{4}-\d{2}-\d{2}) [\d:+]+",
        lambda match: match.group(1),
    ),
    Reason("Data-quality check failed", r"quality: (.+)", _quality_detail),
    Reason(
        "Spread above {spread_ratio} times the market median (median spread, bps)",
        r"liquidity: spread ([\d.]+) bps",
        lambda match: match.group(1),
    ),
)
"""The filters in the order they drop candidates, before the pick."""

PICK_REASONS = (
    Reason("Outside the liquid pool", r"outside liquid pool", None),
    Reason(
        "Near-duplicate: absolute correlation above {threshold} with a kept asset "
        "(kept asset, correlation)",
        r"near-duplicate of (\S+) \(([\d.]+)\)",
        lambda match: f"{_escape(match.group(1).removesuffix('USDT'))}, {match.group(2)}",
    ),
    Reason(
        "Skipped by the sector cap: its sector already holds {cap} stocks of the pool (sector)",
        r"not picked: sector cap \((.+) full\)",
        lambda match: _escape(match.group(1)),
    ),
    Reason("Not picked", r"not picked", None),
)
"""The steps of the pick in the order they drop candidates."""

REASONS = FILTER_REASONS + PICK_REASONS
"""Every step of the rule in its order; every ``dropped`` text matches exactly one."""


def _limits() -> dict[str, str]:
    # the limits of the selection rule that the block labels name, from the study's plan
    with STUDY.open("rb") as file:
        options = tomllib.load(file)["options"]
    return {
        "pegged": " or ".join(options["pegged_currencies"]),
        "spread_ratio": f"{options['max_spread_ratio']:g}",
        "threshold": f"{options['duplicate_threshold']:.2f}",
        "cap": str(options["max_per_sector"]),
    }


def _no_reference() -> set[str]:
    # the stocks that have no Yahoo Finance reference, from every group of the data config
    with DATA_CONFIG.open("rb") as file:
        groups = tomllib.load(file)["groups"]
    return {symbol for group in groups.values() for symbol in group.get("no_reference", [])}


def _escape(text: str) -> str:
    # LaTeX special characters of symbols and sector names, escaped for text mode
    for char in "\\{}^~":
        if char in text:
            raise ValueError(f"unexpected character {char!r} in {text!r}")
    for char in "&%$#_":
        text = text.replace(char, "\\" + char)
    return text


def _display(symbol: str, source: str) -> str:
    # the symbol as the tables print it: Binance perpetuals without their USDT suffix
    if source != "binance":
        return _escape(symbol)
    if not symbol.endswith("USDT"):
        raise ValueError(f"Binance symbol {symbol!r} does not end with USDT")
    return _escape(symbol.removesuffix("USDT"))


HEADER = f"% Generated by paper/scripts/make_tab_assets.py from {UNIVERSE}. Do not edit by hand.\n"
"""The first line of every written file."""


def _pools(universe: pd.DataFrame) -> pd.Series:
    # the pool of every candidate; a candidate dropped before pooling gets that of its market
    stock_pool = universe["source"].map(STOCK_POOLS)
    is_stock = universe["market"] == "stock"
    if stock_pool[is_stock].isna().any():
        raise ValueError(f"a stock has a source outside {list(STOCK_POOLS)}")
    derived = universe["market"].where(universe["market"] != "stock", stock_pool)
    pooled = universe["pool"].notna()
    if not (universe.loc[pooled, "pool"] == derived[pooled]).all():
        raise ValueError("a candidate's pool differs from the pool of its market and source")
    unknown = set(derived) - set(POOL_NAMES)
    if unknown:
        raise ValueError(f"unknown pools {sorted(unknown)}, expected {list(POOL_NAMES)}")
    return derived


def _check(universe: pd.DataFrame) -> None:
    # every listed asset has a pick order and no reason; every other candidate has a reason
    if universe["in_universe"].dtype != bool:
        raise TypeError(f"in_universe must be a bool column, got {universe['in_universe'].dtype}")
    listed = universe["in_universe"]
    if universe.loc[listed, "dropped"].notna().any():
        raise ValueError("a listed asset has a dropped reason")
    if universe.loc[~listed, "dropped"].isna().any():
        raise ValueError("a candidate not in the list has no dropped reason")
    for pool, orders in universe[listed].groupby("pool")["order"]:
        expected = list(range(1, len(orders) + 1))
        if sorted(orders.astype(int)) != expected:
            raise ValueError(f"pick orders of {pool} are not 1 to {len(orders)}")
    if universe["symbol"].duplicated().any():
        raise ValueError("a symbol appears twice in universe.csv")


def _row(cells: list[str]) -> str:
    # one table row; \tabularnewline because \raggedright in a cell redefines \\
    return " & ".join(cells) + " \\tabularnewline"


def _list_body(universe: pd.DataFrame, pools: pd.Series, no_reference: set[str]) -> str:
    # the listed assets: one row per pool with its symbols in pick order (sources in the caption)
    listed = universe[universe["in_universe"]].assign(pool=pools)
    missing = listed["yahoo_return_median_bps"].isna() & (listed["market"] == "stock")
    if set(listed.loc[missing, "symbol"]) != no_reference & set(listed["symbol"]):
        raise ValueError("listed stocks without a Yahoo comparison differ from no_reference")
    lines = [
        f"\\begin{{tabular}}{{@{{}}lp{{{LIST_WIDTH}}}@{{}}}}",
        "\\toprule",
        _row(["Pool", "Assets in pick order"]),
        "\\midrule",
    ]
    for pool, name in POOL_NAMES.items():
        rows = listed[listed["pool"] == pool].sort_values("order")
        if rows.empty:
            raise ValueError(f"pool {pool} has no listed asset")
        sources = rows["source"].unique()
        if len(sources) != 1:
            raise ValueError(f"pool {pool} mixes sources {list(sources)}")
        symbols = [
            _display(symbol, source)
            for symbol, source in zip(rows["symbol"], rows["source"], strict=True)
        ]
        lines.append(_row([name, "\\raggedright " + ", ".join(symbols)]))
    return HEADER + "\n".join([*lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def _reason_of(text: str) -> tuple[int, str]:
    # the index of the step in REASONS that wrote `text`, and the symbol's detail
    for index, reason in enumerate(REASONS):
        match = re.fullmatch(reason.pattern, text)
        if match:
            return index, reason.detail(match) if reason.detail else ""
    raise ValueError(f"unknown dropped reason {text!r}")


def _dropped(universe: pd.DataFrame, pools: pd.Series) -> pd.DataFrame:
    # the candidates not in the list with their pool, the index of their step and their entry
    dropped = universe[~universe["in_universe"]].assign(pool=pools)
    parsed = [_reason_of(text) for text in dropped["dropped"]]
    entries = [
        _display(symbol, source) + (f" ({detail})" if detail else "")
        for symbol, source, (_, detail) in zip(
            dropped["symbol"], dropped["source"], parsed, strict=True
        )
    ]
    return dropped.assign(step=[step for step, _ in parsed], entry=entries)


def _count(dropped: pd.DataFrame, reasons: tuple[Reason, ...]) -> int:
    # the number of candidates dropped by one of `reasons`
    return int(dropped["step"].isin([REASONS.index(reason) for reason in reasons]).sum())


def _dropped_body(
    dropped: pd.DataFrame, reasons: tuple[Reason, ...], limits: dict[str, str]
) -> str:
    # the candidates dropped by `reasons`: one block per step, one row per pool inside it
    lines = [
        f"\\begin{{tabular}}{{@{{}}lrp{{{DROPPED_WIDTH}}}@{{}}}}",
        "\\toprule",
        _row(["Pool", "$n$", "Candidates"]),
    ]
    for reason in reasons:
        block = dropped[dropped["step"] == REASONS.index(reason)]
        if block.empty:
            continue
        lines += [
            "\\midrule",
            _row([f"\\multicolumn{{3}}{{@{{}}l}}{{{reason.label.format(**limits)}}}"]),
        ]
        for pool, name in POOL_NAMES.items():
            rows = block[block["pool"] == pool].sort_values("symbol")
            if not rows.empty:
                entries = "\\raggedright " + ", ".join(rows["entry"])
                lines.append(_row([name, str(len(rows)), entries]))
    count = _count(dropped, reasons)
    lines += ["\\midrule", _row(["All", str(count), ""]), "\\bottomrule", "\\end{tabular}"]
    return HEADER + "\n".join(lines) + "\n"


def _values(universe: pd.DataFrame, dropped: pd.DataFrame) -> str:
    # macros for the counts the captions and the appendix text quote
    pairs = {
        "assetsCandidates": len(universe),
        "assetsListed": int(universe["in_universe"].sum()),
        "assetsDropped": len(dropped),
        "assetsFiltered": _count(dropped, FILTER_REASONS),
        "assetsPegged": _count(dropped, FILTER_REASONS[:1]),
        "assetsSpread": _count(dropped, FILTER_REASONS[-1:]),
        "assetsUnpicked": _count(dropped, PICK_REASONS),
        "assetsNotPicked": _count(dropped, PICK_REASONS[-1:]),
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value:,}}}" for name, value in pairs.items()]
    return HEADER + "\n".join(lines) + "\n"


def main() -> None:
    """Read universe.csv and write the bodies of the asset list and the dropped candidates."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    universe = pd.read_csv(UNIVERSE)
    _check(universe)
    pools = _pools(universe)
    dropped = _dropped(universe, pools)
    limits = _limits()
    values = _values(universe, dropped)
    for name, body in (
        ("assets_list_body.tex", _list_body(universe, pools, _no_reference())),
        ("assets_filtered_body.tex", _dropped_body(dropped, FILTER_REASONS, limits)),
        ("assets_unpicked_body.tex", _dropped_body(dropped, PICK_REASONS, limits)),
        ("assets_values.tex", values),
    ):
        (OUT / name).write_text(body)
        logger.info("wrote %s", OUT / name)
    logger.info("counts:\n%s", values)


if __name__ == "__main__":
    main()
