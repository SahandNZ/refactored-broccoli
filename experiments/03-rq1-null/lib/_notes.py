"""The notes of 03-rq1-null's answers, computed from its tables and 09's size check."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import SCORED, NullRule
from src.stats import BLOCK_OVERLAP_FLOOR

CELL_KEYS = ("model", "market", "horizon")
"""The columns that name a cell."""


def whole_quarter_note(
    cells: pd.DataFrame, verdict: pd.DataFrame, rule: NullRule, *, share: float
) -> str:
    """Return the Q2 note on cells counted by quarters with data, not by whole quarters.

    Q2 counts the cells the Q1 rule can pass (enough quarters with data); the families count
    whole quarters after the scoring start. The note names the late models Q2 counts and the
    share that counting whole quarters would give, with its verdict (a note only; the rule is
    fixed).

    Parameters
    ----------
    cells : pd.DataFrame
        The MDE rows Q2 judges (``model``, ``market``, ``horizon``, ``is_reached``,
        ``mde_pct``).
    verdict : pd.DataFrame
        The Q1 verdict, with ``full_quarters``.
    rule : NullRule
        The rule of rq1.
    share : float
        Q2's share as computed by the rule.

    Returns
    -------
    str
        The note, starting with a space; empty when no counted model is in the late family.

    Raises
    ------
    ValueError
        If a cell has no row in the verdict.
    """
    keys = list(CELL_KEYS)
    whole = cells.merge(verdict[[*keys, "full_quarters"]], on=keys, validate="one_to_one")
    if len(whole) != len(cells):
        raise ValueError(f"{len(cells) - len(whole)} power cells have no row in the verdict")
    is_whole = whole["full_quarters"] >= rule.main_family_min_quarters
    late = sorted(set(whole.loc[~is_whole, "model"].astype(str)))
    kept = whole[is_whole]
    if not late or kept.empty:
        return ""
    is_resolved = kept["is_reached"] & (kept["mde_pct"] <= rule.mde_threshold_pct)
    is_same = (float(is_resolved.mean()) >= rule.power_share) == (share >= rule.power_share)
    same = "same" if is_same else "opposite"
    return (
        f" Q2 counts the cells the Q1 rule can pass (at least {rule.min_winning_quarters} "
        f"quarters with data), so {', '.join(late)} (late family: fewer than "
        f"{rule.main_family_min_quarters} whole quarters after the scoring start) "
        f"{'count' if len(late) > 1 else 'counts'} for Q2; "
        f"counted by whole quarters the share would be {is_resolved.mean():.0%} "
        f"({int(is_resolved.sum())} of {len(kept)}), with the {same} verdict."
    )


def size_note(study_dir: Path) -> str:
    """Return the size check of the shared bootstrap tests at the adopted block floor.

    Read from ``tables/bootstrap_size.csv`` of the one 09-mde-threshold run that 09's results
    README names (``Runs: <id>.``), at `src.stats.BLOCK_OVERLAP_FLOOR`.

    Parameters
    ----------
    study_dir : Path
        03-rq1-null's folder (09 is its sibling).

    Returns
    -------
    str
        The sentence for the Q1 limitations.

    Raises
    ------
    ValueError
        If 09's README names no run or several, or the table has no row at the floor.
    """
    results = study_dir.parent / "09-mde-threshold" / "results"
    runs = re.search(r"Runs: ([^.\n]+)\.", (results / "README.md").read_text())
    if runs is None:
        raise ValueError(f"{results / 'README.md'} names no run")
    named = [name.strip() for name in runs.group(1).split(",")]
    if len(named) != 1:
        raise ValueError(f"{results / 'README.md'} names {len(named)} runs, expected one")
    run = named[0]
    size = pd.read_csv(results / run / "tables" / "bootstrap_size.csv")
    floor = size[size["floor"] == BLOCK_OVERLAP_FLOOR]
    if floor.empty:
        raise ValueError(f"no size row at the block floor {BLOCK_OVERLAP_FLOOR} in 09 {run}")
    draws, nominal = int(floor["draws"].iloc[0]), float(floor["nominal"].iloc[0])
    error = np.sqrt(nominal * (1 - nominal) / draws)
    mean, difference = floor["mean_test_size"], floor["difference_test_size"]
    return (
        "the size check of the shared bootstrap tests with overlapping 24-bar labels at the "
        f"adopted block floor of {BLOCK_OVERLAP_FLOOR} x the overlap, 09-mde-threshold {run} "
        f"tables/bootstrap_size.csv: the mean test {mean.min():.1%}-{mean.max():.1%}, the "
        f"difference test {difference.min():.1%}-{difference.max():.1%} at a nominal "
        f"{nominal:.0%}, series of {', '.join(str(n) for n in floor['length'])} decision times, "
        f"{draws} draws each (standard error about {error:.1%})"
    )


def null_note(mde: pd.DataFrame, verdict: pd.DataFrame, *, resamples: int) -> str:
    """Return the Q2 note on the detection rate with no planted edge (``zero_rate``).

    Its largest cell, its median, how many cells fire at least once (and how that compares to
    their level at the resolution of the resamples) and above ten times their level, and the
    real tests' smallest p before and after Holm for comparison.

    Parameters
    ----------
    mde : pd.DataFrame
        The MDE table (``zero_rate``, ``false_positive_level``, ``model``, ``market``,
        ``horizon``, ``times``).
    verdict : pd.DataFrame
        The Q1 verdict (``status``, ``p_value``, ``adjusted_p``).
    resamples : int
        The bootstrap resamples per cell.

    Returns
    -------
    str
        The note, starting with a space.
    """
    rates = mde["zero_rate"].to_numpy(dtype=float)
    levels = mde["false_positive_level"].to_numpy(dtype=float)
    top = int(rates.argmax())
    model, market = str(mde["model"].iloc[top]), str(mde["market"].iloc[top])
    horizon, times = int(mde["horizon"].iloc[top]), int(mde["times"].iloc[top])
    scored = verdict[verdict["status"] == SCORED]
    p_values = scored["p_value"].to_numpy(dtype=float)
    adjusted = scored["adjusted_p"].to_numpy(dtype=float)
    resolution = 1 / resamples
    fired = int((rates > 0).sum())
    return (
        f" Under no edge the rule fired in at most {rates[top]:.1%} of the resamples of "
        f"any cell (median {np.median(rates):.3g}; one-sided level alpha / m at most "
        f"{levels.max():.2g}); the largest is {model} {market} {horizon} bar(s), "
        f"{rates[top]:.3f} with {times} decision points. {fired} of {len(rates)} cells fire at "
        f"least once"
        + (
            f", which at the resolution of {resolution:.3g} is above their level"
            if resolution > levels.max()
            else f", {int((rates > levels).sum())} above their level"
        )
        + f"; {int((rates > 10 * levels).sum())} exceed ten times it. The real tests: the smallest "
        f"one-sided p of a scored cell is {p_values.min():.2g}, {adjusted.min():.2g} after Holm."
    )
