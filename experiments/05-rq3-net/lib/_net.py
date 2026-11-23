"""Trading the sign of each model's median after each market's costs (rq3).

`book_returns` gives the equal-weight sign book of one market and horizon, and `net_tables`
its gross and net result, turnover and break-even cost for every model, market and horizon,
with a bootstrap interval of the mean net return and of the break-even cost.

The book of a model, market and horizon ``h`` runs on the model's own decision times in that
cell and the times its positions end. A position is held for ``h`` of the asset's own bars,
the label's horizon, so its realised return is the decision's cached ``log_return``; an asset
opens a new position only at its first decision at or after the end of its last one, so
holdings never overlap and the decisions in between are not used. Returns are per book
time, as simple returns, and are annualised with the book's own frequency.
"""

import math

import numpy as np
import pandas as pd

from src.benchmark import (
    CELL,
    SCORED,
    ForecastCache,
    StudySettings,
    cell_stream,
    complete_cells,
    full_quarters,
    release_date,
    require_scored,
    scored_decisions,
    split_models,
)
from src.evaluate import Fold, portfolio_returns, sharpe_ratio, strategy_returns
from src.stats import MIN_BLOCK_VALUES, MIN_BOOTSTRAP_BLOCKS, has_enough_blocks, is_constant
from src.utils import last_at_or_before

from ._bootstrap import BOOTSTRAP_COLUMNS, book_block, book_bootstrap, is_effective

BUY_AND_HOLD = "buy-and-hold"
"""The reference book that holds every asset of a market long over the scored folds."""

NEVER_TRADES = "not scored: never trades"
"""The ``status`` of a book whose positions are always flat (e.g. a median always zero)."""

BOOK_COLUMNS = ("gross", "net", "turnover")
"""The columns of `book_returns`."""


def _openings(rows: pd.DataFrame) -> pd.DataFrame:
    # One asset's chain of positions: the first decision opens one; the next opening is the
    # first decision at or after the label time of the last (each depends on the one before,
    # so the chain is a loop); the decisions in between are dropped.
    rows = rows.sort_values("time")
    decision_times = pd.DatetimeIndex(rows["time"])
    label_times = pd.DatetimeIndex(rows["label_time"])
    is_opening = np.zeros(len(rows), dtype=bool)
    position = 0
    while position < len(rows):
        is_opening[position] = True
        position = int(decision_times.searchsorted(label_times[position]))
    return rows[is_opening]


def _holdings(rows: pd.DataFrame, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    # one asset's signal and realised return at `times`: an open position keeps its median
    # (no trade) until its label time, and its return counts once, at its opening
    openings = _openings(rows)
    last = last_at_or_before(pd.DatetimeIndex(openings["time"]), times)
    chosen = np.maximum(last, 0)
    is_held = (last >= 0) & (times < pd.DatetimeIndex(openings["label_time"])[chosen])
    is_opened = is_held & (times == pd.DatetimeIndex(openings["time"])[chosen])
    signal = np.where(is_held, openings["median"].to_numpy(dtype=float)[chosen], np.nan)
    realised = np.where(
        is_opened, np.expm1(openings["log_return"].to_numpy(dtype=float))[chosen], 0.0
    )
    return signal, realised


def book_returns(
    medians: pd.DataFrame, costs_bps: pd.Series, *, times: pd.DatetimeIndex
) -> pd.DataFrame:
    """Return the equal-weight sign book of one market and horizon at each of `times`.

    At each time of `times` an asset of `costs_bps` without an open position opens one: the
    sign of its median forecast at that time, with weight ``1 / n_assets`` (``n_assets`` is
    every asset of the market, so an asset without a position is cash), held until its
    ``label_time`` (``h`` own bars later). The position's simple return
    ``exp(log_return) - 1`` counts at its opening time. An asset without a forecast and
    without an open position (no labelled decision, model not run) holds nothing. A trade
    pays the asset's cost per side on the change of its position
    (`src.evaluate.strategy_returns`): opening or closing costs one side, a flip from long to
    short two. Medians at other times are not used.

    Parameters
    ----------
    medians : pd.DataFrame
        One row per decision: ``time``, ``asset``, ``median`` (of the forecast label),
        ``log_return`` (realised over the label's bars) and ``label_time``.
    costs_bps : pd.Series
        Cost per side in bps, indexed by every asset of the market, shape ``(n_assets,)``.
    times : pd.DatetimeIndex
        The book times (the decision times and the end times of positions), in time order,
        shape ``(n_times,)``.

    Returns
    -------
    pd.DataFrame
        Indexed by `times`, shape ``(n_times, 3)``: ``gross`` and ``net`` (the book's simple
        return before and after costs) and ``turnover`` (mean over the assets of the absolute
        position change, between 0 and 2).

    Raises
    ------
    ValueError
        If `medians` has an asset that is not in `costs_bps`, a ``label_time`` at or before
        its ``time`` or a ``log_return`` that is not finite.
    """
    unknown = sorted(set(medians["asset"]) - set(costs_bps.index))
    if unknown:
        raise ValueError(f"assets {unknown} have no cost per side, got {list(costs_bps.index)}")
    if (medians["label_time"] <= medians["time"]).any():
        raise ValueError("every decision's label_time must be after its time")
    if not np.isfinite(medians["log_return"].to_numpy(dtype=float)).all():
        raise ValueError("every decision needs a finite log_return, got NaN or inf")
    assets = list(costs_bps.index)
    signal = pd.DataFrame(np.nan, index=times, columns=assets)
    realised = pd.DataFrame(0.0, index=times, columns=assets)
    on_grid = medians[medians["time"].isin(times)]
    for asset, rows in on_grid.groupby("asset"):
        signal[asset], realised[asset] = _holdings(rows, times)
    strategy = strategy_returns(signal, realised, costs_bps)
    return pd.DataFrame(
        {column: portfolio_returns(strategy, kind=column) for column in BOOK_COLUMNS}
    )


def _book_times(decisions: pd.DataFrame, openings: pd.DataFrame) -> pd.DataFrame:
    # the cell's decision times and the label times its positions (`openings`) end at (so a
    # position that ends before the next decision is closed there, the last one too; the
    # ends of dropped decisions are no book times), each with the window of its decision
    ends = openings[["label_time", "window"]].rename(columns={"label_time": "time"})
    times = pd.concat([decisions[["time", "window"]], ends], ignore_index=True)
    return times.drop_duplicates("time").sort_values("time", ignore_index=True)


def _overlap(book: pd.DataFrame, openings: pd.DataFrame, *, horizon: int) -> int:
    # how many later effective times of the book a position spans, the median over its
    # positions (openings with a nonzero median), rounded up, as rq1 counts its labels'
    # overlap (`cell_timing`): 0 when every asset's positions open and close at the same
    # times, more when they do not (a market that pools sessions, or chains out of phase)
    openings = openings[openings["median"] != 0]
    if openings.empty:
        return 0
    effective = pd.DatetimeIndex(book.loc[is_effective(book, horizon=horizon), "time"])
    starts = effective.searchsorted(pd.DatetimeIndex(openings["time"]), side="right")
    later = effective.searchsorted(pd.DatetimeIndex(openings["label_time"])) - starts
    return math.ceil(float(np.median(later)))


def _books(medians: pd.DataFrame, cache: ForecastCache) -> pd.DataFrame:
    # one book per market and horizon of `medians`, on the decision times of that cell, with
    # the overlap of its positions (`_overlap`)
    costs = cache.assets.set_index("asset")["cost_per_side_bps"]
    assets = cache.assets.groupby("market")["asset"].apply(list)
    books: list[pd.DataFrame] = []
    for _, chosen in medians.groupby(CELL):
        market, horizon = str(chosen["market"].iloc[0]), int(chosen["horizon"].iloc[0])
        openings = pd.concat([_openings(rows) for _, rows in chosen.groupby("asset")])
        times = _book_times(chosen, openings)
        book = book_returns(chosen, costs[assets[market]], times=pd.DatetimeIndex(times["time"]))
        book = book.reset_index(names="time").assign(window=times["window"].to_numpy())
        overlap = _overlap(book, openings, horizon=horizon)
        books.append(book.assign(market=market, horizon=horizon, overlap=overlap))
    if not books:
        return pd.DataFrame(columns=["time", *BOOK_COLUMNS, "window", *CELL, "overlap"])
    return pd.concat(books, ignore_index=True)


def _delayed(medians: pd.DataFrame, next_bars: pd.DataFrame) -> pd.DataFrame:
    # the one-bar delay check: each median trades at the asset's next own bar, so it takes
    # the label of the decision there (same horizon); a median whose next bar has no
    # decision of the model (Kronos, every 167 h) is dropped
    moved = medians[["time", "asset", "horizon", "median"]].merge(next_bars, on=["time", "asset"])
    moved = moved.drop(columns="time").rename(columns={"next_time": "time"})
    return medians.drop(columns="median").merge(moved, on=["time", "asset", "horizon"])


def _summary(book: pd.DataFrame) -> dict[str, float]:
    # the break-even cost is the one cost per side at which the mean net return is zero; the
    # cost paid is the turnover-weighted mean cost of the market's assets; both are undefined
    # (NaN) for a book that never trades; the Sharpe ratios are annualised with the book's
    # own frequency: its times per year
    gross = float(book["gross"].mean())
    net = float(book["net"].mean())
    turnover = float(book["turnover"].mean())
    years = (book["time"].max() - book["time"].min()) / pd.Timedelta(days=365.25)
    periods_per_year = (len(book) - 1) / years if years > 0 else math.nan
    return {
        "times": book["time"].nunique(),
        "gross_mean": gross,
        "net_mean": net,
        "turnover": turnover,
        "cost_bps": (gross - net) / turnover * 1e4 if turnover > 0 else math.nan,
        "break_even_bps": gross / turnover * 1e4 if turnover > 0 else math.nan,
        "gross_sharpe": sharpe_ratio(book["gross"], periods_per_year=periods_per_year),
        "net_sharpe": sharpe_ratio(book["net"], periods_per_year=periods_per_year),
    }


def _status(book: pd.DataFrame, *, horizon: int, effective: int) -> tuple[str, int | None]:
    # why a book with `effective` effective times is not scored, or `SCORED`, and its block
    # (None when there is none), as in rq1, rq2 and rq4: it must trade, have enough times for
    # the automatic block, hold at least `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks and have a
    # net return that varies (a constant one has no bootstrap error to test with)
    if book["turnover"].sum() == 0:
        return NEVER_TRADES, None
    if effective < MIN_BLOCK_VALUES:
        reason = f"{effective} effective book times, fewer than {MIN_BLOCK_VALUES} for the block"
        return f"not scored: {reason}", None
    block = book_block(book, horizon=horizon, overlap=int(book["overlap"].iloc[0]))
    if not has_enough_blocks(effective, block):
        reason = f"{effective} effective book times, fewer than {MIN_BOOTSTRAP_BLOCKS} blocks"
        return f"not scored: {reason} of {block}", block
    net = book.loc[is_effective(book, horizon=horizon), "net"].to_numpy(dtype=float)
    if is_constant(net, scale=float(np.abs(net).mean())):
        return "not scored: no spread in the net return", block
    return SCORED, block


def _cells(books: pd.DataFrame, settings: StudySettings, *, model: str) -> pd.DataFrame:
    # every book's summary and bootstrap (`book_bootstrap`), with the cell's own random
    # stream (`src.benchmark.cell_stream`, as rq1); a book that is not scored (`_status`) has
    # NaN bootstrap columns; the bootstrap runs on the effective times only (`is_effective`:
    # all at horizon 1, those with a return or a trade above), so its block and the blocks a
    # draw holds are counted in effective times
    bootstrap = settings.bootstrap
    rows: list[dict[str, object]] = []
    for (market, horizon), book in books.sort_values("time").groupby(CELL):
        steps = int(book["horizon"].iloc[0])
        effective = int(is_effective(book, horizon=steps).sum())
        status, block = _status(book, horizon=steps, effective=effective)
        row: dict[str, object] = {"market": market, "horizon": horizon, "status": status}
        row["quarters"] = book["window"].nunique()
        row.update(_summary(book))
        row["effective_times"] = effective
        row["overlap"] = int(book["overlap"].iloc[0])
        row["block"] = math.nan if block is None else block
        row["blocks"] = math.nan if block is None else effective / block
        row.update(dict.fromkeys(BOOTSTRAP_COLUMNS, math.nan))
        if status == SCORED and block is not None:
            result = book_bootstrap(
                book,
                horizon=steps,
                block=block,
                level=bootstrap.level,
                n_resamples=bootstrap.n_resamples,
                rng=cell_stream(bootstrap.seed, "net", model, market, steps),
            )
            row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)


def _post_release_quarters(cache: ForecastCache, model: str, folds: tuple[Fold, ...]) -> int:
    # the scored folds that start on or after the model's release (`full_quarters`, as rq1);
    # all of them for a baseline
    release = release_date(cache, model)
    if pd.isna(release):
        return len(folds)
    return full_quarters(release, folds)


def _buy_and_hold(cache: ForecastCache, windows: list[str]) -> pd.DataFrame:
    # the decisions of the buy-and-hold book: a median of +1 at every 1-bar label of the
    # scored folds, so each asset is long from its first bar to its last (one entry and one
    # final close, and a close and reopen wherever its 1-bar labels have a gap, e.g. at a
    # fold boundary outside the asset's trading hours; the same 1 / n weights and cost rule
    # as every book)
    labels = cache.labels[(cache.labels["horizon"] == 1) & cache.labels["window"].isin(windows)]
    labels = labels.merge(cache.assets[["asset", "market"]], on="asset")
    return labels.assign(median=1.0)


def _next_bars(cache: ForecastCache) -> pd.DataFrame:
    # the time of each asset's next own bar after a decision: the label time at horizon 1
    if 1 not in set(cache.labels["horizon"]):
        raise ValueError(
            f"the one-bar delay check needs labels at horizon 1, got "
            f"{sorted(set(cache.labels['horizon']))}"
        )
    labels = cache.labels[cache.labels["horizon"] == 1]
    next_bars = labels[["time", "asset", "label_time"]].drop_duplicates(["time", "asset"])
    return next_bars.rename(columns={"label_time": "next_time"})


def net_tables(
    cache: ForecastCache, models: list[str], settings: StudySettings
) -> dict[str, pd.DataFrame]:
    """Score the sign book of every model of `models` per market and horizon (rq3).

    The book (`book_returns`) runs on the model's `scored_decisions` in the cell (a zero-
    shot model only after its release date, 00:00 UTC) and the times its positions end; the
    last closing trade is charged too. Each asset pays the ``cost_per_side_bps`` of the
    cache's ``assets.csv``. A cell without any decision of the model, whose book never
    trades (`NEVER_TRADES`), has fewer than ``src.stats.MIN_BLOCK_VALUES`` (11) effective
    times (`is_effective`; too few for the automatic block) or fewer than
    ``src.stats.MIN_BOOTSTRAP_BLOCKS`` (5) blocks of `book_block`, or whose net return has
    no spread, is not scored (``status``). The circular-block bootstrap (`book_bootstrap`)
    uses ``settings.bootstrap``'s level and draws, and a stream per cell from its seed. A
    reference book, `BUY_AND_HOLD` (kind ``"reference"``), holds every asset of a market
    long over the scored folds at horizon 1: one entry and one final close per asset (and a
    close and reopen wherever its 1-bar labels have a gap, e.g. at a fold boundary outside the
    asset's trading hours), the same weights and costs.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache, with labels at horizon 1 (the one-bar delay check).
    models : list of str
        The models to score (their kind comes from the cache).
    settings : StudySettings
        The scored folds (in time order) and the bootstrap of the mean net return.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"quarters"``: one row per model, market, horizon and fold. ``"cells"``: one row per
        model, market and horizon over all folds (every cell, with ``status``; the
        `BUY_AND_HOLD` rows at horizon 1 after the models), plus
        ``quarters`` (folds with a book time), ``post_release_quarters`` (scored folds that
        start on or after the model's release), ``invalid``, ``effective_times`` (the book
        times that count, `is_effective`), ``blocks`` (effective times / block), ``overlap``
        (the median number of later effective times a position spans, `_overlap`), ``block``
        (`book_block`, NaN when not scored), the columns of `book_bootstrap` (``net_low``,
        ``net_high``, ``p_value`` of H1 ``net_mean > 0``, ``break_even_low``,
        ``break_even_high``). ``"delayed"``: the
        one-bar delay check, one row per model, market and horizon whose medians trade at
        the asset's next own bar (no Kronos rows). Each of the three has ``model``, ``kind``,
        ``times``, ``gross_mean``, ``net_mean``, ``turnover`` (means per book time),
        ``cost_bps``, ``break_even_bps``, ``gross_sharpe`` and ``net_sharpe`` (annualised
        with the book's times per year). ``"costs"``: per market, the number of its assets
        (``assets``) and their mean ``cost_per_side_bps``.

    Raises
    ------
    ValueError
        If a model or a fold is not in the cache, the cache has no labels at horizon 1, or
        no cell of a zero-shot model is scored.
    """
    windows = [fold.name for fold in settings.scored_folds]
    split_models(cache, models, windows=windows)
    next_bars = _next_bars(cache)
    kinds = cache.models.set_index("model")["kind"]
    quarters: list[dict[str, object]] = []
    delayed: list[dict[str, object]] = []
    cells: list[pd.DataFrame] = []
    for name in models:
        decisions, invalid = scored_decisions(cache, name, windows=windows)
        medians = decisions.rename(columns={"q50": "median"})
        books = _books(medians, cache)
        names = {"model": name, "kind": kinds[name]}
        for (market, horizon, window), book in books.groupby([*CELL, "window"]):
            cell = {"market": market, "horizon": horizon, "window": window}
            quarters.append({**names, **cell, **_summary(book)})
        for (market, horizon), book in _books(_delayed(medians, next_bars), cache).groupby(CELL):
            delayed.append({**names, "market": market, "horizon": horizon, **_summary(book)})
        scored = complete_cells(
            _cells(books, settings, model=name), cache, name, invalid=invalid, windows=windows
        )
        released = _post_release_quarters(cache, name, settings.scored_folds)
        cells.append(scored.assign(kind=kinds[name], post_release_quarters=released))
    hold = _books(_buy_and_hold(cache, windows), cache)
    for (market, horizon, window), book in hold.groupby([*CELL, "window"]):
        cell = {"market": market, "horizon": horizon, "window": window}
        quarters.append({"model": BUY_AND_HOLD, "kind": "reference", **cell, **_summary(book)})
    folds = len(settings.scored_folds)
    hold_cells = _cells(hold, settings, model=BUY_AND_HOLD).assign(
        model=BUY_AND_HOLD, kind="reference", invalid=0, post_release_quarters=folds
    )
    cells.append(hold_cells)
    costs = cache.assets.groupby("market")["cost_per_side_bps"].agg(["size", "mean"])
    tables = {
        "quarters": pd.DataFrame(quarters),
        "cells": pd.concat(cells, ignore_index=True),
        "delayed": pd.DataFrame(delayed),
        "costs": costs.set_axis(["assets", "cost_per_side_bps"], axis=1).reset_index(),
    }
    zero_shot = tables["cells"][tables["cells"]["kind"] == "zero-shot"]
    require_scored(zero_shot, study="rq3", windows=windows)
    return tables
