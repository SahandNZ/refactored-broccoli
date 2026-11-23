# rq5 Off-hours: feeding weekend crypto to a zero-shot model at the reopening

The secondary answer of the paper's RQ5, "When a market is closed, how should a zero-shot
model be fed?". The main answer is [08-closures](../08-closures/README.md) (trading hours
only, closed hours forward-filled, or closed hours missing); this study asks whether
information from a market that trades during the closure, crypto, helps the forecast of the
reopening (Q1), and checks with a regression that the input is worth trying (Q2,
supporting).

## Questions

**Q1** (RQ-off-hours). Does feeding the weekend crypto move to a zero-shot model as a covariate improve its forecast of the first bar after a weekend reopening?
Decision rule: yes for a market and model if the covariate arm has a lower CRPS than each of
the four other arms (the same model in calendar time without the covariate, the same model in
event time, the correction and the model-free forecast): all four one-sided
Diebold-Mariano tests (H1: the covariate arm has the lower expected CRPS) have a p-value below
0.05 after Holm correction over all tests of all cells (the four tests of every cell and the
tests of the correction and the model-free forecast against event time); no otherwise. Q1 is
yes if any cell says yes, no if every tested cell says no, inconclusive if no cell is tested.
Primary metric: CRPS of the first bar after the reopening, in the benchmark's label units.

**Q2** (RQ-off-hours). Does the weekend crypto move explain the reopening gap of closed-market assets beyond Friday's information?
Decision rule: yes for an asset if the weekend crypto coefficient of a regression of the
reopening gap on Friday's information and the weekend crypto move is significant (two-sided
p-value of its HAC t, Holm-adjusted over the tested assets, below 0.05) and its rank
correlation with the gap, without the 5 weekends with the largest crypto moves, has the sign
of the coefficient; no otherwise. Q2 is yes if any asset says yes. Reported per asset, over
time (half-years) and by direction, with the sign of each coefficient. Q2 is inconclusive
if no asset is tested.
Primary metric: the crypto coefficient's two-sided Holm-adjusted p-value per asset (its
sign, HAC interval, rank correlation and R^2 increment are reported next to it).

## Hypothesis

- H5: the weekend crypto move explains part of the reopening gap of index CFDs (with a sign
  that is not known in advance for forex and commodities); zero-shot models given the move
  as a covariate pass on little of it, so a simple correction or a model-free forecast from
  it is as good as or better than the covariate arm.

## Design

The code is the study's `lib/` (`_settings.py` for the settings and the asset sets,
`_weekends.py` for the weekends, `_reopening.py` for the forecasts, `_comparisons.py` for the
tests of Q1 and the covariate use, `_gaps.py` for Q2); `run.py` and `analyze.py` only call it.
Every value below in `code font` is an option of `study.toml`.

- **Assets** (`universe_assets`): both sets come from the study's own 01-universe run
  (`universe_run`, today `2026-11-17T1235-data-filter-column`, the universe of the
  benchmark's `2026-11-17T0005-random-1000` with a `passed_data_filter` column), its
  `tables/universe.csv`. Q1 forecasts the closed-market assets of its universe
  (`in_universe`: 16 forex pairs, 8 commodities and 8 index CFDs), as 08-closures does; that
  universe must be the paper's, the one of the `universe_run` pinned by the 02-forecasts run
  named by `forecast_cache`, or the run stops with an error (a later universe run may add
  columns for Q2, never another universe). Q2 regresses every closed-market candidate that
  passed 01-universe's data filter on its selection window 2020-2023 (`passed_data_filter`:
  no pegged currency, bars from the start and a passed quality check; 52 forex pairs, 15
  commodities and 18 index CFDs), in the universe or not: the checks end before the weekends
  Q1 scores, so the set does not require the data to stay good up to 2026; the candidate
  lists themselves are from 2026 (a limitation). 1h bars, mid prices.
- **Crypto basket** (`crypto_basket`, `basket_return`): BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
  and BNBUSDT, Binance USD-M perpetual closes, equally weighted: the crypto move is the plain
  mean of the five assets' log returns over the same hours. The basket was fixed in the plan
  of 2026-11-16, with the probes that motivated the study, not chosen from data; it is not
  the universe's crypto set, and SOL and BNB are outside the 72 benchmark assets. That is on
  purpose and kept: the basket is only an input (the covariate and the regressor), never a
  scored asset, so it needs no place in the universe; what matters is a broad crypto move
  that trades during the closure, and a fixed basket keeps that input the same in both
  questions and over time.
- **Stocks stay out.** Index CFDs stand in for equities: they reopen with the index futures
  on Sunday evening, while a stock reopens on Monday after its index has already priced the
  weekend, so a stock's gap is mostly the index gap plus company news that a crypto basket
  does not carry; a stock also closes every night, so its weekend closure is one of five a
  week (08-closures covers stock closures).
- **Weekends** (`weekend_closures` on `src.utils.closures`, the closure finder 08-closures
  uses too; `weekend_rows`): a closure of at least `min_closed_hours` (24) and at most
  `max_closed_hours` (96) hours from the close of a bar to the open of the next one, after
  which the asset reopens on a Sunday or Monday (UTC); midweek holidays and Tuesday
  reopenings are not weekends. `close_time` is the close of the last bar before the closure,
  `reopen_time` the open of the first bar after it, `label_time` that bar's close, `week`
  the Monday 00:00 UTC that starts the ISO week of the Friday close (one point per weekend
  for every asset, however staggered its reopening: forex on Sunday 22:00, a European index
  on Monday 07:00). A week has at most one weekend per asset, its first closure: a second
  one that starts in the same week (a gap in the data right after a Sunday reopening) is
  left out and logged.
- **No look-ahead.** The inputs of a weekend use only what is known at `reopen_time`:
  Friday's information ends at `close_time` (`friday_target`: the asset's log return over the
  `friday_hours` (6) hours before the close, from its last close at or before the start;
  `friday_crypto`: the basket over the same hours), the crypto move `weekend_crypto` covers
  exactly the closed hours, `close_time` to `reopen_time`. A crypto asset without a bar at
  exactly one of the times leaves the basket move missing (never filled); such weekends are
  counted and left out: in Q2 as `incomplete` (an asset without any complete weekend is
  listed as not tested), in Q1 as `excluded` (they are forecast, but have no correction and
  no model-free forecast). In effect both questions start at the first weekend
  after the last basket asset lists on Binance USD-M (SOLUSDT, 2020-09-14): before that the
  basket move is missing.
- **Targets.** Q2's target is the gap, the log of the first bar's open over Friday's last
  close; Q1's target is the first bar's close over Friday's last close, scored as the
  benchmark's label at horizon 1: that log move over `sigma`, the EWMA volatility of hourly
  log returns at `close_time` (span `span_bars`, 96, the benchmark's, from the 02-forecasts
  run's `config.toml`), as 08-closures scales its labels. The two targets differ by the
  first bar's own move, so Q2 shows whether the input carries information and Q1 whether a
  model can use it.
- **Q1 forecasts** (`reopening_forecasts`): the weekends of the Q1 assets that reopen from
  `split_date` (2024-01-01). Every model of `[models]` (Chronos-2,
  TiRex-2, t0-gamma, built as the 02-forecasts run builds them, from its `config.toml`; each
  model's Hugging Face `main` is checked against the benchmark's pinned commit,
  `src.models.require_revision`, before it is built) forecasts one step, the close of the
  first bar after the reopening, from the same `lookback` (512, the benchmark's) calendar
  hours whose close is at or before `reopen_time` (`calendar_windows`), in three arms:
  - *event* (trading hours only): the asset's own closes inside those hours, in order (the
    last is Friday's close), exactly as 08-closures builds its "trading hours only" representation
    (`src.benchmark.own_bars_only`): a shorter context (about 70 % of the hours for forex),
    given to the model without padding, one call per context length, so the model never sees a NaN;
  - *calendar*: every one of those hours, NaN while the asset is closed; it isolates the
    covariate's own effect from the change of time axis;
  - *covariate*: the calendar context with one past covariate on the same hours: the
    basket's mean log return since `close_time` (zero at Friday's close, the weekend crypto
    move at the last hour; NaN at an hour where a basket asset has no bar, and at every hour
    if one has no bar at Friday's close), `Forecaster.predict_with_covariates`.
    Chronos-2 takes the series and its covariate as one group (group attention, dict input
    with `past_covariates`), TiRex-2 its `past_covariates`, t0 the covariate as a further
    variate of the same series with the role "historical" (read over the context only, not
    forecast); each keeps the target's forecast. All three read NaN as missing.

  A weekend without `lookback` hours of history has no forecast in the model arms (nor a
  correction) and is counted as excluded. The price quantiles become label quantiles
  (`label_quantiles` with `sigma`).
  **TimesFM 3.0 is not in `[models]`**: it fills NaN by linear interpolation before the model
  sees it (its library's `linear_interpolation`, `src.models.missing_value_reason`), so it
  has no calendar and no covariate arm in the same hours; `off_hours_settings` refuses any
  model that fills NaN or takes no past covariate. Two more arms use no further model input:
  - *correction* (`corrected_quantiles`, `fit_correction`): the event-time label quantiles
    `q` of the same model moved to a centre `beta * x` and widened, `beta * x + width * (q -
    median(q))`, with `x` the weekend crypto move over the asset's `sigma` (label units).
    For each weekend, `beta` (least-squares slope through the origin of the label on `x`)
    and `width` (the factor of `width_factors`, 0.5 .. 10.0 in steps of 0.05, with the lowest
    mean CRPS; the label covers the whole closure, often several hourly sigmas, while the
    event forecast is one bar wide, so the grid reaches far, and `width_at_edge` in
    `reopening_cells` reports the share of fits at an end of the grid) are fitted on the
    earlier weekends of the same asset and model whose label was known at its Friday close
    (`label_time <= close_time`), whose label and crypto move are finite and whose event
    forecast is valid;
  - *model-free* (`model_free_quantiles`): `beta * x` plus the nine `QUANTILE_LEVELS`
    quantiles of the residuals `label - beta * x` of the same earlier weekends (without the
    condition on an event forecast, so the arm is the same for every model); a model-free
    reference that the model arms must beat.

  A weekend with fewer than `min_fit_weekends` (26) such earlier weekends has no correction
  and no model-free forecast. Both fits use the forecast weekends only (from `split_date`), so
  the first scored weekends fit on about 44 earlier ones on d1-d4 and about 96 on t1-t4.
- **Q1 tests** (`reopening_tables`): the weekends that reopen inside the scored folds;
  earlier ones only fit the correction and the model-free arm. An asset-weekend is scored
  when all five arms are valid (nine finite, ordered quantiles) and its label is known; the
  others are counted (`excluded`). A cell is a market (forex, commodity, index) and model:
  its assets' CRPS (twice the mean pinball loss over the nine quantiles) is averaged once
  per `week`, as 02-06 pool a market's assets at each decision. Per cell and pair of
  `COMPARISONS` (covariate against calendar, event, correction and model-free; correction
  and model-free each against event): the one-sided Diebold-Mariano test of that weekly
  series (`src.stats.diebold_mariano` with the overlap of the weekly labels,
  `label_overlap`, 0 here: Newey-West variance with at least the overlap and Andrews'
  automatic number of lags, Harvey-Leybourne-Newbold correction, Student's t with `n - 1`
  degrees of freedom; H1: the arm has the lower CRPS), its skill `100 (1 - CRPS arm / CRPS
  reference)` with a `interval_level` (95 %) moving-block bootstrap interval
  (`bootstrap_resamples` 2000 draws, blocks of the Politis-White length of the loss
  difference, at least four times the overlap (`src.stats.BLOCK_OVERLAP_FLOOR`) and at least
  1; `src.stats.politis_white_block`); a worse arm shows as a skill interval below zero. All
  p-values of all cells are Holm-adjusted together. A cell with fewer than `min_weekends`
  (20) weekly points is not tested and not counted by Holm. Nor is a comparison whose weekly
  series holds fewer than `src.stats.MIN_BOOTSTRAP_BLOCKS` (5) blocks (`has_enough_blocks`,
  the owner's rule for every study): it gets no p-value and no skill interval, and its
  `status` says why. Nor is a comparison of two arms with identical losses (no variance of
  the difference, so no p-value; such a cell's covariate arm cannot be the best). Next to
  the arms, untested, every cell reports the benchmark's naive reference `ewma-normal` (the
  standard normal in label units) on the same weekly points (`crps_ewma-normal`,
  `coverage_ewma-normal`). 07 compares ways of feeding a zero-shot model.
- **Covariate use** (`covariate_use`, no extra forecasts; descriptive, no interval, decides
  nothing): per market and model, on the asset-weekends Q1 scores,
  `use_slope`, the slope (least squares with an intercept) of the difference of the medians
  of the covariate and the calendar arm (back in log-return units, times `sigma`) on the
  weekend crypto move; `move_slope`, the slope of the realised move on the crypto move over
  the same asset-weekends; their ratio `use_share` (near 0 when a model ignores the
  covariate, near 1 when it passes on as much of the move as the data hold); and next to
  them `gap_beta`, the median Q2 coefficient of the market's tested Q1 assets (a partial
  slope of the gap with Friday's information, over all Q2 weekends, so not like for like).
- **Q2** (`gap_tables`): per asset, least squares of `gap ~ 1 + friday_target +
  friday_crypto + weekend_crypto` with Newey-West errors (`hac_lags` = 4 weekends, Bartlett
  weights, small-sample factor n / (n - k); `src.stats.newey_west_ols`); the R^2 increment is
  the full model's R^2 minus that of the model without `weekend_crypto`. The crypto term's t
  is read against Student's t with n - k degrees of freedom: the two-sided p-value (the sign
  is not known in advance: a safe-haven currency or gold can move against crypto) is
  Holm-adjusted over the tested assets, and the coefficient's `interval_level` interval
  comes from the same HAC standard error, so it leaves out zero exactly when the unadjusted
  p is below 0.05. Rank correlation is Spearman's of `weekend_crypto` with the gap; the
  trimmed one leaves out the `trim_largest` (5) weekends with the largest absolute crypto
  move. An asset says yes when its adjusted p is below `alpha` (0.05) and its trimmed rank
  correlation has the sign of its coefficient (`sign`). The same statistics are reported per
  asset and calendar half-year of the reopening and per asset and direction (crypto move up
  or down; a weekend without a crypto move is in neither); a group with fewer than
  `min_weekends` (20) weekends is counted, not tested. Weekends reopen from `start`
  (2020-01-01, in effect from SOLUSDT's listing) to the end of the scored folds (2025-10 with
  the development folds, 2026-10 with the test folds).
- **Size of the tests** (synthetic checks in the tests): the one-sided Diebold-Mariano test
  on 48 autocorrelated weekly points rejects about 5 % of true nulls at 5 %; the two-sided
  HAC t of Q2 with 4 lags rejects about 7 % on 100 weekends (Bartlett HAC is a little
  liberal in small samples), a limitation of Q2.
- **No release cut.** Every weekend of the scored folds is scored for every model, also
  before its release (the `release_dates` of the 02-forecasts run are not read): the five
  arms of a model are compared on the same weekends. On the development folds (2024-11 ..
  2025-10) every model of `[models]` is scored before its release (Chronos-2 2025-10-30,
  TiRex-2 2026-07-02, t0 2026-10-15). On the test folds (2025-11 .. 2026-10) Chronos-2 is
  after its release on every weekend, TiRex-2 is before it until 2026-07-02 (about two
  thirds of the weekends) and t0 is before it on almost every weekend (all but the last
  two). Pre-training may include some of these weekends; if a model had
  memorised a reopening, that could favour its event, calendar and covariate arms over the
  model-free arm (and partly the correction, which keeps the width of the event forecast but
  not its centre); we cannot measure this, and the results README names it as a limitation.
- **Deterministic models**, one seed (`[seeds]`), set by `src.experiment.seed_everything`
  before each model is built.

Probes of 2026-11-16 (outside this code): corr 0.51-0.69 on 12 US and European indices since
2024, +22-37 points of R^2 over Friday information, Chronos-2 ignores the covariate, the
correction gains 8.5 % CRPS.

`run.py` writes `tables/reopening_cells.csv` and `tables/reopening_tests.csv` (Q1),
`tables/covariate_use.csv`, `tables/gap_assets.csv`, `tables/gap_half_years.csv` and
`tables/gap_directions.csv` (Q2), `tables/weekends.csv` (every weekend with its inputs and
targets), `tables/asset_sets.csv` (each asset and whether Q1 forecasts it) and
`forecasts.parquet` (every weekend, model and arm, with the correction's fitted `width`);
`analyze.py` writes the results README from the tables.

## Data

Q1: the 32 closed-market assets of the benchmark's universe (16 forex pairs, 8 commodities, 8
index CFDs), weekends from 2024-01 (scored on the weekends of d1-d4, or of t1-t4 with the
locked plan). Q2: every forex pair, commodity and index CFD that passed 01-universe's data
checks of 2020-2023, weekends reopening from 2020-09 (SOLUSDT's listing) to the end of the
scored folds (.. 2025-10 while the plan is open, .. 2026-10 with the test folds). 1h bars.

## Metrics

- Q1, per market and model: CRPS of the first bar after the reopening (label units) and
  10-90 % coverage for each arm; per comparison the skill and its bootstrap interval, the
  Diebold-Mariano statistic, its lags, the one-sided p-value and the Holm-adjusted one; the
  covariate use (`use_slope`, `move_slope`, `use_share`, `gap_beta`).
- Q2, per asset, half-year and direction: number of weekends, Pearson and Spearman
  correlation of the weekend crypto move with the gap, the trimmed Spearman, the crypto
  term's coefficient, its HAC t, two-sided p-value and interval, R^2 of the Friday model and
  of the full model and their difference; per asset also the Holm-adjusted p-value and the
  sign.

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
runs once for the final numbers. Unlike the paper's shared design, Q1 makes no cut at a
model's release (see the Design section). The model-against-reference tests of Q1 are
one-sided at `alpha` (H1: the arm is better than its reference); a worse arm shows in the
skill interval. Q2 is two-sided.

Two stated exceptions to "training ends before each quarter":

- The correction and the model-free arm are fitted again before every weekend, on all
  earlier weekends since `split_date` (2024-01-01) whose label was known at that weekend's
  Friday close (at least `min_fit_weekends`, 26), including earlier weekends of the same
  quarter (also in t1-t4). This is online estimation from past labels
  only, like the benchmark's EWMA scale; no future label enters a fit. A fit frozen at the
  start of each quarter would give these arms less data than the model arms, whose contexts
  read the earlier weeks of the quarter too, and so would favour the covariate arm.
- Q2 is an in-sample regression over every weekend from `start` to the end of the scored
  folds: its test run includes the development months and the whole sample of the
  development run (about 85 % of its weekends). Its options were fixed before any run, but
  its test-run answer is not on unseen data; it supports Q1 and decides nothing about the
  models.

The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions).

## Compute

Q2 is CPU only (seconds to minutes). Q1 is three one-step forecasts (event, calendar and
covariate) per weekend, asset and model: with the development folds about 95 weekends
(2024-01 .. 2025-10) x 32 assets x 3 = about 9,000 contexts of at most 512 steps per model,
with the test folds about 148 weekends, about 14,000; the event arm is one call per context
length. The correction, the model-free arm and the bootstrap are CPU. Estimate on the local
RTX 3500 (12 GB): loading each model 1-2 minutes, forecasting under 8 minutes per model, so
about 0.3-0.5 GPU hours for all three; `estimated_hours = 1.0` in `study.toml` (the runner
stops at twice that). No rented GPU, no cost.

## Status

Plan of 2026-11-16, coded on 2026-11-16 and tested on synthetic bars with fake models
(`tests/experiments/01-benchmark/07-rq5-off-hours/`: `test_off_hours.py`, `test_gaps.py`,
`test_reopening.py`, `test_comparisons.py`, `test_rq5_off_hours_study.py`;
`tests/models/test_covariates.py`). Run on real data on 2026-11-17 (below).
Plan approved and locked by the owner on 2026-11-17, with the dev and test runs together:
`[plan] locked = true`, `[folds] use = "test"` for the test run `test-2026-11-17` (t1-t4).
The dev run `dev-2026-11-17` (d1-d4) runs the same plan with `use = "development"`, its
`config.toml` keeps that copy. Both run on a rented RTX 4090 (vast.ai), one model at a time.

Low power: a cell has one weekly point per weekend, 52 in a year of folds (d1-d4 or t1-t4),
so only large CRPS differences can pass the Holm-adjusted one-sided tests; the skill
intervals show what is left out.

Changed on 2026-11-16 after the second code review, before any run: the model builds and the
context length are read from the 02-forecasts run's `config.toml` (`forecast_cache`), not
copied into this study's `study.toml`; the weekend closures come from the closure finder
shared with 08-closures (`src.utils.closures`).

Changed on 2026-11-16 after the third code review, before any run: the gap regression decides
on a Holm-adjusted p-value (the option `t_threshold` is gone); the forecasts have a fourth
arm, *calendar*, and test the covariate against all others; a cell pools a market's assets;
the headline differences get block-bootstrap intervals; `run.py` refuses `--models` (the
tests are Holm-adjusted over every model of `study.toml`).

Changed on 2026-11-17 after a step-1 review, before any run: the study is the secondary
answer of RQ5 ("how should a zero-shot model be fed?"), so the forecast question is Q1 and
the regression is the supporting Q2. Q1: a cell pools its assets once per weekend (`week` of
the Friday close; before, per exact reopening time, which split staggered reopenings into
dependent points); the CRPS is in the benchmark's label units (the move over the EWMA sigma
at Friday's close; before, raw log moves); the event arm holds the own bars of the same 512
calendar hours as 08-closures' "trading hours only" (before, the last 512 own bars);
TimesFM 3.0 is out of `[models]`; a fifth arm, *model-free*; the tests are one-sided
Diebold-Mariano tests (Andrews lags, t reference) with Politis-White bootstrap skill
intervals, the correction and the model-free arm are also tested against event time in the
same Holm family, and the
"lowest CRPS" condition is gone; a cell needs `min_weekends` weekly points; each model's
revision is checked against the benchmark's pin; the covariate use is reported. Q2: two-sided
HAC t with a t reference and a HAC interval (before, one-sided normal p and a bootstrap
interval), the sign is reported, the rank-correlation condition follows the sign, a weekend
without crypto move is in neither direction; the assets are those that passed 01-universe's
data checks of 2020-2023 (before, `passed` in `data/manifest.json`, over 2020-2026); stocks
stay out. The options `bootstrap_block_hours` and `bootstrap_min_block` are gone and
`interval_level` is new.

Changed on 2026-11-17 after the first referee review of that change, before any run: a week
has at most one weekend per asset; in Q1 a weekend without a crypto move is forecast and
counted as excluded (before, dropped unseen); an asset without a complete weekend is listed
in Q2 as not tested; the covariate use is compared with the slope of the realised move over
the same weekends (`move_slope`) and called descriptive; the Politis-White block is
`src.stats.politis_white_block`. After the second referee review: the covariate use is
measured on the Q1-scored asset-weekends only (a missing crypto move made it NaN); every cell
reports `ewma-normal`; a run without any full calendar context stops with an error.

After the third referee review: a closure longer than `max_closed_hours` (96) is not a
weekend (a data outage or a holiday that starts midweek); the width grid of the correction
reaches 10.0 (before, 4.0), each fitted width is kept in `forecasts.parquet` and every cell
reports `width_at_edge`; the README says when a comparison has no p-value.
The event arm's cut is `src.benchmark.own_bars_only`, shared with 08-closures (2026-11-17;
before, a copy in this study's `lib/`).

The study names its own 01-universe run (`universe_run`, 2026-11-17; before, it read the
02-forecasts run's), checked to hold the paper's universe.
The README states the crypto basket, its equal weights and why it stays outside the
universe (2026-11-17); the 08-closures representation is called "trading hours only".
The study's 01-universe run is `2026-11-17T1235-data-filter-column` (same universe), whose
`passed_data_filter` column gives the Q2 set (before, matching the drop reasons by prefix;
the same 85 closed-market candidates either way).
The comparisons of Q1 need at least `src.stats.MIN_BOOTSTRAP_BLOCKS` (5) bootstrap blocks
(owner's rule, 2026-11-17); Q2 uses no bootstrap (HAC intervals), so the rule does not apply
there.
`forecast_cache` names the 02-forecasts run `test-2026-11-17` (its locked configuration:
lookback 512, span 96, the model builds and their pinned revisions; 2026-11-17); the check
against the benchmark's pinned universe reads only that run's universe, since its
`universe.csv` predates `passed_data_filter`.
