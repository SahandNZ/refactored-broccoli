# rq3 Net: what is left after costs, and the break-even cost

## Questions

**Q1** (RQ-net). Does trading the sign of a model's median earn money after each market's
costs, and up to which cost?
Decision rule: yes if any cell of the main family earns money after costs: its net Sharpe
ratio is positive and its one-sided bootstrap p-value (H1: the mean net return is above 0),
Holm-adjusted within its family over all horizons and markets, is below 0.05; no otherwise.
The late-release family gets its own verdict by the same rule. The break-even cost and its
interval are reported for every cell whatever the answer.
Primary metric: break-even cost per side in bps (the cost at which the mean net return is
zero).

## Hypothesis

- H3: costs remove any centre skill; break-even costs are below every market's real cost.

## Design

Forecasts come from the paper's shared forecast cache: every model and baseline forecasts
once per decision time and asset, and rq1-rq4 only read that cache (see
[../README.md](../README.md#shared-design)). This study adds no forecasts of its own; it reads
the 02-forecasts run named by `forecast_cache` in `study.toml` through
`src.benchmark.read_cache`.

The code is the study's `lib/` (`_net.py` the book and its tables, `_bootstrap.py` the
bootstrap, `_verdict.py` the decision rule, `_table.py` the paper's table bodies, written by
`make_tab_rq3_break_even.py`), built on the shared decisions of `src.benchmark`, with the same
choices as rq1 where the plan does not say otherwise:

- **Signal**: the sign of the median (`q50`) of each forecast, on the model's valid decisions
  (`scored_decisions`, as in rq1: invalid forecasts left out and counted per cell; Kronos's one
  forecast per decision pools the 60 sampled paths of its seeds 0-2, 20 each, into one set of
  quantiles in 02-forecasts, so it holds one position). A median of exactly zero holds nothing
  (the random walk, ewma-normal and garch-normal never trade; climatology trades the sign
  of its training median, which rarely changes). The ridge regression (`linear`) is the
  reference row: its median is the forecast `x . w` from the asset's recent own returns
  (`linear_lags`) plus the median residual of its training window (`src.benchmark._baselines`),
  so its sign comes from recent returns and that constant offset.
- **Book** (`book_returns`), per (model, market, horizon `h`): the book's times are the model's
  own decision times in the cell (in the scored folds, `[folds] use`: t1-t4 for the paper, d1-d4
  while the plan is open; a zero-shot model only after 00:00 UTC of its release date, as in rq1)
  and the label times its positions end at (not those of dropped decisions). A position is held
  for `h` of the asset's own bars, the label's horizon, until its label time. Positions never
  overlap: an asset opens a new position only at its first decision at or after the end of its
  last one, and the decisions in between are dropped (for crypto every `h`-th hour; for a market
  that closes a position can span several decision times, without a trade). The return of a
  position is the simple return `exp(log_return) - 1` of the cache's label, counted at its
  opening time. Weights are `1 / n` over all `n` assets of the market in the cache: an asset
  without an open position at a book time is cash (weight `1 / n`, return 0), so a position that
  ends before the next decision is closed at its end and reopening is paid. The stock market
  pools the US and the non-US stocks of the universe in one book. Kronos books trade only at
  Kronos's own grid (one decision per asset at each 167 h grid point): each position is held for
  its `h` bars and the book is flat until the next Kronos decision.
- **Costs**: each asset pays its `cost_per_side_bps` from the cache's `assets.csv` (set by
  02-forecasts: the Binance USD-M taker fee of 5 bps for crypto, half the median Dukascopy
  spread for the other markets, measured over its `spread_window` 2020-01-01 to 2024-01-01
  (the selection window of 01-universe, before every fold), an assumed 1 bp half spread for
  US stocks from Alpaca) on the
  absolute change of its position (`src.evaluate.strategy_returns`): opening or closing is one
  side, a flip from long to short two. The closing trade of every position is charged, the
  last one of the scored folds too.
- **Per cell** (`net_tables`): over the book's times in the scored folds, the mean gross and net
  return of the book per time, the mean turnover per time (mean over the market's assets of the
  absolute position change, 0 to 2), the cost paid (`cost_bps` = (gross - net) / turnover, the
  turnover-weighted mean cost of the market's assets) and the **break-even cost per side**
  (`break_even_bps` = mean gross return / mean turnover, in bps: the one cost per side at which
  the mean net return is zero). So `net_mean = turnover * (break_even_bps - cost_bps) / 10^4`.
  Both are empty (NaN) for a book that never trades, which is not scored ("not scored: never
  trades"); a book with fewer than 11 effective times (too short for the automatic block,
  `src.stats.MIN_BLOCK_VALUES`) or fewer than 5 blocks (`src.stats.MIN_BOOTSTRAP_BLOCKS`, as
  in rq1, rq2 and rq4), or whose effective net return has no spread (no bootstrap error to
  test with), is not scored either. The effective times (`effective_times`,
  `is_effective`) are every book time at `h = 1`; for `h > 1` only the book times with a return
  or a trade, since the times in between hold a position without a new return. The same per fold
  in `quarters.csv` (the per-quarter net mean; rq3 has no quarter condition): a book time takes
  the fold of a decision at that time, or, when it is only the end of a position, the fold of
  that position's decision (so the last ends, after the scored folds, count in the last fold,
  and a close at the first decision time of the next fold counts in that fold).
- **Interval and test** (`book_bootstrap`), the scheme of rq1, rq2 and rq4: a circular-block
  bootstrap (`src.stats.circular_block_indices`) of the book's series in time order at its
  effective times (for a horizon above 1 the zero times between the returns would hide their
  autocorrelation from the block choice), with the cell's own random stream (`bootstrap_seed`
  and a hash of the model, market and horizon, `src.benchmark.cell_stream`, as rq1). The block
  (`book_block`) is automatic: the longest of the Politis-White lengths
  (`src.stats.politis_white_block`: Politis and White 2004 with the correction of Patton,
  Politis and White 2009, through `arch`) of the effective net returns, gross returns and
  turnover, so the break-even interval keeps the autocorrelation of the most persistent of them,
  with the floor of rq1: at least `src.stats.BLOCK_OVERLAP_FLOOR` (4) times the book's overlap
  and at least 2. The overlap (`overlap`) is the median number of later effective times a
  position spans, as rq1 counts its labels' overlap: 0 when every asset's positions open and
  close at the same times, more when they do not (a market that pools sessions, such as the
  stock market, or chains of positions out of phase); openings that hold nothing do not count.
  One set of draws (`bootstrap_resamples`) resamples the same blocks of the three series. The
  mean net return is tested as `src.stats.bootstrap_mean_test` tests a mean: its bootstrap
  standard error against Student's t with the blocks less one as degrees of freedom, the
  one-sided p-value of H1 "the mean net return is above 0" (`p_value`) and the t interval at 1 -
  `alpha` (`net_low`, `net_high`), scaled from the effective times to all the book's times (the
  left-out times are 0), so the interval is per book time and the p-value does not depend on the
  zero times. The break-even cost keeps the percentile interval of the same draws at 1 - `alpha`
  (`break_even_low`, `break_even_high`: mean gross over mean turnover of each draw; NaN when a
  draw has no trade), the owner's rule of 2026-11-17 for rq3, while rq1 and rq2 give their
  ratios a t interval. `block` is in effective times and `blocks` = effective times / block; a
  scored cell with fewer than `bootstrap_min_blocks` (10) blocks is flagged (`has_few_blocks`),
  not dropped. The block can depend on the costs, since the net series is one of the three.
- **Families** (`net_verdict`, in `analyze.py`): a model is in the main family if at least
  `main_min_quarters` (3) of the 4 scored folds start on or after its release
  (`post_release_quarters`, `src.benchmark.full_quarters` as in rq1; a fold the model
  enters mid-way does not count), else in the late-release family. Holm runs over all the
  scored zero-shot cells of one family (3 horizons x 5 markets x the family's models), so Q1's
  one answer controls its family-wise error; the late-release family is judged by the same
  rule with its own Holm and its own verdict. The ridge regression and the other baselines
  are not tested; they are in `cells.csv` (the ridge also in `reference.csv`) for reference.
- **Buy and hold** (`BUY_AND_HOLD`, kind `reference`, a second reference row): per market a
  book that is long every asset from its first 1-bar label of the scored folds to its last
  (a median of +1 at every bar), with the same 1 / n weights and cost rule: one entry and one
  final close per asset (and a close and reopen wherever its 1-bar labels have a gap, e.g.
  at a fold boundary outside the asset's trading hours or at a dropped label). It is
  at horizon 1 in `cells.csv`, `quarters.csv`, `reference.csv` and the tables. It shows how
  much of a book that is nearly always long is the market's drift. It covers all scored folds,
  while a late-release model's cells cover only its decisions after its release.
- **Low turnover** (a flag, `is_low_turnover` in `verdict.csv`): a scored cell whose turnover
  per effective time (`turnover_per_effective`: total turnover over its effective times, so the
  same scale at every horizon) is below `min_turnover` (0.05: one side every 20 positions, per
  asset when the market's assets share their bar times; pooled sessions such as the stock market
  dilute it) rarely changes its sign; it is close to buy and hold and its break-even cost is
  gross over almost no turnover. The flag never changes the answer.
- **Decision rule**: per (zero-shot model, market, horizon) cell, the cell earns money after
  costs if its net Sharpe is positive and its Holm-adjusted p-value is below `alpha`. Every cell
  has a row; a cell without any decision of the model, whose book never trades, is too short for
  the automatic block or holds fewer than 5 blocks is not scored (`status`, no test). Q1 is yes
  if any main-family cell earns and lists those cells; every cell's break-even cost is in
  `verdict.csv` whatever the answer. rq3 makes no choice of its own on the development folds.
- **One-bar delay** (robustness, `delayed.csv`): the same book with each median trading at
  the asset's next own bar (the label time of its 1-bar label), with the label of the
  model's decision there at the same horizon (both decisions valid). It needs a decision at
  the next bar, so Kronos has no row. Its numbers never change the answer.

`run.py` writes `tables/cells.csv` (one row per model, market and horizon, all models of
`[models]`), `tables/quarters.csv` (the same per fold), `tables/delayed.csv` (the delay check)
and `tables/costs.csv` (per market, the number of its assets, `assets`, and their mean cost per
side, for reference); `analyze.py` writes `tables/verdict.csv` (the zero-shot cells with
`family`, `adjusted_p`, `turnover_per_effective`, `earns_after_costs`, `has_few_blocks` and
`is_low_turnover`), `tables/reference.csv` (the cells of the ridge regression and of buy and
hold) and the results README; `make_tab_rq3_break_even.py --run-id ID` then writes the bodies of
the paper's tables at `table_horizon` (1 bar) from that run's tables to
`papers/benchmark/tables/generated/`: `rq3_body.tex`
(`tab:rq3`: the cost per side the market's cells pay, the break-evens of the ridge and of buy
and hold, then each zero-shot model's break-even per market, bold if it earns, the late-release
family after a rule) and the appendix table (gross, net, turnover and break-even with their
intervals per model and market) in two bodies of at most 40 rows, so each fits a page:
`rq3_appendix_main_body.tex` (the main family) and `rq3_appendix_late_body.tex` (the
late-release family, then the ridge regression and buy and hold). The names of markets and
models come from `src.benchmark.MARKET_NAMES`, `MODEL_NAMES` and `REFERENCE_NAMES` (buy and
hold).

## Data

The T1 universe of 01-universe, 1h bars; the post-release test windows of the shared design.

## Metrics

- Per cell and per fold: mean gross and net return per book time, mean turnover per book
  time, cost paid and break-even cost per side in bps, gross and net Sharpe ratio
  (annualised with the book's own frequency: its times per year, from its first and last
  time).
- Per cell: the 95 % circular-block bootstrap intervals (t for the mean net return,
  percentile for the break-even cost) of the mean net return and of the break-even
  cost, the one-sided p-value and its Holm-adjusted value (`verdict.csv`), the block (in effective
  times), `effective_times` and the number of blocks (effective times / block).
- Per market: the cost per side its scored zero-shot cells pay at the table's horizon (the
  first row of `tab:rq3`: the mean of their `cost_bps`, both families, weighted by their mean
  turnover per book time, i.e. sum of gross minus net over the sum of turnover; each cell's
  verdict uses its own `cost_bps`) and the plain mean of its assets'
  costs (`costs.csv`); the buy-and-hold book's numbers (`reference.csv`).

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
(after each model's release, see the paper's shared design) runs once for the final numbers.
The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions).

## Compute

Minutes on CPU once the shared forecasts exist.

## Status

Split out of 02-zero-shot on 2026-11-16. Plan approved and locked by the owner on 2026-11-17
after the development run `dev-2026-11-17` (cache `dev-2026-11-17` of 02, d1-d4: Q1 no); the
test run reads the cache `test-2026-11-17` (t1-t4). The trading
and `run.py` / `analyze.py` are coded and tested on a synthetic cache
(`tests/experiments/01-benchmark/05-rq3-net/`: `test_net.py`, `test_net_bootstrap.py`,
`test_net_verdict.py`, `test_net_table.py` and `test_rq3_net_study.py`); no
run yet, the shared forecast cache does not exist yet (`forecast_cache` is empty).

Open for the owner: none of the 2026-11-17 review's method questions remains open (decisions
of 2026-11-17, below).

Changed on 2026-11-16 after an independent code review, before any run: positions are held
for `h` own bars without overlap (the label's horizon), the shared decision rule of every
study (valid forecasts, cells not scored reported), and the bootstrap block from the spacing
of the book.

Changed again on 2026-11-16 after a second independent review, before any run: the book runs
on the cell's own decision times (no calendar grid), a position that ends before the next
decision is closed at its end, the Sharpe ratios are annualised with the book's own frequency,
and a book that never trades is not scored.

Changed on 2026-11-17 after a review of the paper against the code, before any run (owner's
decisions): one-sided tests (H1: mean net return above 0); Holm per horizon within a main
family and a separate late-release family (fewer than 3 of the 4 scored folds after the
release), each with its own verdict; the automatic Politis-White block in place of the
24-hour block; a bootstrap interval of the break-even cost from the same draws; the final
closing trade charged (as the code already did) and said so; the one-bar delay check; the
market's mean cost per side, the ridge regression's reference rows and the bodies of the
paper's tables; the 1 / n weights with cash and the pooled stock market written out.

Changed again on 2026-11-17 (owner's decisions, before any run): `bootstrap_min_blocks` stays
10; one block for the net and gross returns and the turnover, the longest of their three
Politis-White lengths; for `h > 1` the bootstrap, its block and the sample checks (11 times
for the block, the count of blocks) use only the book times with a return or a trade; at
`h = 1` every book time counts.

Changed on 2026-11-17 (owner's rules, before any run): the table bodies are written by
`make_tab_rq3_break_even.py` (now allowed in a study folder) to the paper's
`tables/generated/`, no longer by `analyze.py`; the names come from the shared
`src.benchmark.MARKET_NAMES` and `MODEL_NAMES`.

Changed on 2026-11-17 (owner's decisions, before any run): one Holm family over all horizons
and markets for the main models and one for the late-release models (Q1 is yes if a
main-family cell earns); the intervals stay at 95 %, the level rq1 uses (`1 - alpha`); a
low-turnover flag (`min_turnover`); a buy-and-hold reference row per market; the cost row of
`tab:rq3` is the turnover-weighted mean of the cells' costs.

Changed on 2026-11-17 (owner's decision, before any run): the bootstrap is the scheme of rq1,
rq2 and rq4: circular blocks, the mean net return tested as `src.stats.bootstrap_mean_test`
tests a mean (bootstrap standard error, Student's t with the blocks less one, its t interval),
at least 5 blocks for a test and a stream per cell; a net return without spread is not scored.
The block is still the longest of three Politis-White lengths over the effective times, now
with rq1's floor (4 times the book's overlap, at least 2); the 95 % level and the percentile
interval of the break-even cost are unchanged.
