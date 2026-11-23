# 08 Closures: how closed market hours are shown to a model

The main answer of the paper's RQ5, "When a market is closed, how should a zero-shot model be
fed?": three representations of the same 512 calendar hours. The other input, weekend crypto
next to a closed market, is studied in [07-rq5-off-hours](../07-rq5-off-hours/README.md).

## Questions

**Q1** (RQ-closures). Does any model forecast the first bars after a reopening better when the
closed hours are forward-filled or marked as missing than with trading hours only?
Decision rule: trading hours only is the reference, fixed before the run. For each model,
forward-filled against the reference and missing against the reference, each by a one-sided
Diebold-Mariano test (H1: the alternative has the lower CRPS) on the series of closure days,
Holm-adjusted over every test of the models on time (at least 3 of the 4 scored quarters
start at or after their scoring start, as in rq1); an alternative is better when its adjusted p is below 0.05.
A model's verdict is trading hours only unless an alternative is better (if both are, the one
with the larger skill). Q1 is yes if at least one tested model on time has a better
alternative, no if every tested model on time keeps trading hours only, and inconclusive if
no model on time could be tested (fewer than 20 closure days or `src.stats.MIN_BOOTSTRAP_BLOCKS` bootstrap blocks, a loss
difference without variance, or no scored decision). The late models (fewer than 3
quarters) form their own Holm family with their own yes / no / inconclusive, reported next
to the answer.
Primary metric: CRPS of the first 3 bars after closures of 12 hours or more.

**Q2** (RQ-closures). Is the result of Q1 consistent with an effect of the representation rather
than of the market?
Decision rule: yes if every model has the same best representation (its lowest mean CRPS,
not Q1's tested verdict, among its representations: three, or two without the missing arm)
on a 24/7 market (crypto) with its weekends cut out as on the closed markets; no
otherwise; inconclusive without a model scored in both groups.
Primary metric: CRPS of the first 3 bars after an artificial closure, crypto.

## Hypothesis

- H6: trading hours only is best for every model; NaN hurts models that keep the positions of
  masked patches.

## Design

The code is the study's `lib/` (`_closure_contexts.py`, `_closures.py`, `_verdict.py`,
`_table.py`): `run.py` calls `closure_losses`, `analyze.py` calls `closure_tests`,
`closure_verdict` and `best_representations`, and `make_tab_rq5_closures.py` calls
`rq5_table_body`.
The label, the context length (`lookback`, 512), the EWMA span, how each model is built, its
pinned commit and its scoring start are the benchmark's: they come from the `config.toml` of
the 02-forecasts run named by `forecast_cache`, and its `assets.csv` gives the assets
(`benchmark_settings`, `closure_settings`, `benchmark_assets`). This study makes its own
forecasts; it does not read the forecast cache.

- **Closures** (`reopening_decisions` on `src.utils.closures`, the closure finder 07 uses too):
  a closure is at least `min_closure_hours` (12) hours without an own bar between two own
  bars; the bar after it reopens the market. Its targets are the reopening bar and the next
  own bars, `reopen_bars` (3) of them or fewer when the next closure comes first. Typical
  closures: forex, commodities and indices at weekends (48 hours), stocks every night and
  weekend. An exchange that opens off the hour grid has a half bar as its reopening bar: a
  US stock the bar 09:30-10:00 New York (the first bar of the regular session on the 1h grid,
  closing at 10:00), a Hong Kong stock 09:30-10:00 Hong Kong (01:30-02:00 UTC). It is kept as
  the reopening bar, so its target covers half an hour and its decision is half an hour
  before the open (09:00 New York, 01:00 UTC).
- **Decisions**: one per target bar, one hour before it closes (when it opens on the 1h grid;
  for a half bar half an hour before the open), in the scored
  folds (`[folds] use`: d1-d4 while the plan is open, t1-t4 for the paper). So the target is
  the next own bar in every arm: step 1 in event time and in calendar time. The decision
  before the reopening bar is the exchange's opening time, known in advance; nothing after
  the decision time enters a context (tested). Each decision belongs to the **closure day**
  of its closure: the UTC day on which the closure ends (its reopening bar opens), also for
  the later target bars that fall on the next day.
- **Label** (`closure_contexts`): the benchmark's vol-scaled return at horizon 1: with `end`
  the last own close at or before the decision, `log(P[end + 1] / P[end]) / sigma`, `sigma`
  the EWMA volatility of own-bar returns at `end` (span `span_bars` of the benchmark). For
  the reopening bar this is the return across the closure.
- **Three representations of the same 512 calendar hours** (`closure_contexts` on
  `calendar_windows`), `lookback` values each, oldest first. Every representation sees the
  same hours, the last 512 up to the decision, so they differ only in how the closed hours
  are shown, not in how far back they reach (tested):
  - **trading hours only** (the reference): only the own closes inside those hours, in
    order, right-aligned and NaN-padded on the left: a shorter context
    (`ClosureContexts.own_bars` values; about 70 % of the hours for forex, about 20 % for a
    stock in its session). The padding is cut off before the model is called: contexts of
    equal length go in one call, so the model never sees a NaN in this arm;
  - **forward-filled**: the same hours, each closed hour repeating the last close. These are
    made-up flat prices: this arm is the object of study, the only place in the lab where a
    price is filled; the closed hours are flagged (`ClosureContexts.is_closed`) and their
    share is reported (`closed_share`). Closed hours at the start of the window hold the last
    close before it (for a US stock nearly every window starts in the night), so this arm
    carries one price older than the 512 hours; nothing after the decision enters it;
  - **missing**: the same calendar hours with NaN at every closed hour.
- **Benchmark input** (descriptive, never tested): the last 512 own closes, the input of the
  main results (02-forecasts), which reaches further back than the 512 calendar hours. It is
  forecast and reported in `summary.csv` so that the reference can be compared with the main
  results' input; it is in no test, no Holm family, no bold and not in Q2.
- A decision is kept only when the asset has 512 calendar hours and 512 own closes before it,
  at least `min_own_closes` (32) own closes in those 512 hours (its trading-hours context)
  and a finite label, so every arm has exactly the same decisions and labels. 512 calendar
  hours hold about 100 own closes for a US stock, about 85-95 for the shortest stock
  sessions (Tokyo, Hong Kong; about 50-60 across a holiday week) and many more for forex,
  indices and commodities, so 32 drops only contexts damaged by a data gap; their number per
  group is in `summary.csv` (`short_contexts`), in the Q1 evidence (every group) and in the
  Q2 evidence (the control).
- **Control** (Q2, `cut_weekends`): every crypto asset with the bars that open on a Saturday or
  a Sunday (UTC) removed, then the same closures, decisions and contexts: one 48-hour
  closure every weekend, decisions on Mondays 00:00-02:00 UTC. This study has no crypto arm
  of its own: weekend crypto as an input is 07-rq5-off-hours.
- **One seed** (`[seeds]`, 0): every model of the study is deterministic for a given context
  (no sampling; Kronos, which samples, is not coded), so the seed spread of experiments rule 4
  is zero and the noise of an answer is the bootstrap interval over closure days.
- **Forecasts** (`closure_losses`): each zero-shot model of `[models]` forecasts one step
  from each context (per group and arm, and for trading hours only per context length, in
  calls of at most the benchmark's `chunk` contexts, 1024, or `moirai_chunk` for Moirai, read
  from the 02-forecasts run, so a call needs no more GPU memory than in 02-forecasts), only
  for decisions after its scoring start (00:00 UTC; the later of its release date and the upload date of its pinned weights, from the 02-forecasts run); a model
  with no decision after it is not built and its rows of `summary.csv` say "released after
  the scored folds" (a run where no model has a decision stops with an error). Before a
  model is built, its repository's `main` must be the commit pinned in the 02-forecasts run
  (`src.models.require_revision`), and `seed_everything` runs. The price quantiles become
  label quantiles with the benchmark's transform (`label_quantiles`). A decision is scored
  for a model only when its forecast is valid (nine finite, ordered quantiles) in every
  representation the model has (the benchmark input also needs its own forecast valid);
  invalid ones are counted. `ewma-normal` (the standard normal in label units) is the
  baseline: it reads no context, so it is the same in every arm and is not tested. It scores
  every decision of the scored folds, a zero-shot model only those after its scoring start,
  so the baseline's CRPS level and a late model's are on different days; the comparisons
  that answer the questions (F and M against T) are within one model.
- **Toto's patch**: Toto 2.0 (both sizes) needs a context whose length is a multiple of
  its patch of 32. Its wrapper marks every value as observed, so a NaN would be read as data:
  a shorter trading-hours context (e.g. 60 own closes) is left-padded with its first value to
  the next multiple of 32 (`LENGTH_MULTIPLES`), in both groups; a 512-value context goes
  unchanged. For Toto the trading-hours arm therefore holds up to 31 repeated oldest prices,
  which also enter Toto's own scaling of the context, so Toto's reference arm is not strictly
  the same input as the other models' (stated in the Q1 and Q2 limitations). Every other
  model takes any length (t0 checked on CPU; Chronos, TimesFM, TiRex and Moirai on vast.ai).
- **Missing arm not applicable**: TimesFM 2.5 and TimesFM 3.0 fill NaN by linear
  interpolation inside their packages, and the Toto 2.0 wrapper marks every value as
  observed; for these four models the missing arm is reported as not applicable with the
  reason from `src.models.missing_value_reason` (one table by model kind, read by 07 too).
  Nothing is filled for them. Chronos-Bolt, Chronos-2, TiRex 1.1, TiRex-2, t0 and Moirai 2.0
  read NaN as missing (checked in the installed packages on 2026-11-16).
- **Kronos** and **joint inputs** (a closed-market asset next to BTC in one multivariate
  model) are not coded; see [Status](#status).
- **Closure-day series** (`closure_day_means`): a model's scored losses in one arm are
  averaged over the assets at each decision time, then over the decision times of each
  closure day; the days are the units of every mean, test and interval (the decisions of one
  reopening are dependent; days share no label, so the label overlap is 0). Each day weighs
  the same, each decision time weighs the same within its day.
- **Test** (Q1, `closure_tests`): on the closed markets, for each zero-shot model, forward-
  filled and missing against trading hours only by the one-sided Diebold-Mariano test
  (`src.stats.diebold_mariano` with `overlap` 0: Newey-West variance with
  `max(overlap, Andrews automatic)` lags, Harvey-Leybourne-Newbold correction, Student's t
  with n - 1 degrees of freedom; H1: the alternative has the lower expected CRPS, `p_value`
  is that one-sided p). A difference without variance is not tested. The **skill**
  `1 - CRPS_alternative / CRPS_reference` (positive: the alternative is better) gets a
  moving-block bootstrap interval of level `1 - alpha` (95 %) (`skill_interval`, `bootstrap_resamples` 2,000 draws,
  `bootstrap_seed`), the same blocks of days for both series, block length by Politis-White
  on the difference series (`block_length` on `src.stats.politis_white_block`, the
  circular-bootstrap length of `arch.bootstrap.optimal_block_length` rounded up, at least
  `src.stats.BLOCK_OVERLAP_FLOOR` (4) times the label overlap and at least 1, at most
  the number of days; the closure-day series has overlap 0, so the overlap floor is 0 and
  only the minimum block of 1 acts), each test with its own
  stream of draws. A test whose days hold fewer than `src.stats.MIN_BOOTSTRAP_BLOCKS` (5) blocks
  (`src.stats.has_enough_blocks`,
  the owner's rule of 2026-11-17) gets neither test nor interval; so does a model without a
  scored decision on the closed markets. The interval is descriptive; an
  alternative whose whole interval is below zero
  is reported as worse (`is_worse`), never tested. A model with fewer than
  `min_closure_days` (20) closure days gets neither test nor interval.
- **Families**: a model with fewer than `min_quarters` (3) of the 4 scored quarters wholly
  after its scoring start (quarters that start at or after it, `src.benchmark.full_quarters`,
  the rule of rq1) is late. The tests of the models on
  time and those of the late models are Holm-adjusted separately (`closure_verdict`), each
  family with its own verdict.
- **Decision rule** (`closure_verdict`, in `analyze.py`): an alternative is better when its
  Holm-adjusted one-sided p is below `alpha` (0.05); a model's verdict is the better
  alternative (of two, the larger skill) or else trading hours only; the models on time
  answer Q1, the late family is reported next to it.
- **Control** (Q2, `best_representations`): for each zero-shot model scored in both groups,
  its best representation (lowest mean CRPS over closure days, split `all`, of its
  representations) on the closed markets next
  to its best on the control. Q2 is yes when every model has the same best in both groups,
  no otherwise, and inconclusive without such a model. The best is not tested here, so a
  best inside the noise counts (a limitation in the results).
- **Paper table** (`rq5_table_body`, in `make_tab_rq5_closures.py --run-id ID`, run after
  `analyze.py`): the rows of tab:rq5, from the run's
  `tables/summary.csv` and `tables/verdict.csv`, written to
  `papers/benchmark/tables/generated/rq5_body.tex` (`--out-dir` for another folder): per model the mean CRPS with trading hours only (T) and
  the change of forward-filled and missing against it, in %, on the closed markets and on
  the control; bold is each model's verdict on the closed markets (a model not tested has no
  bold). Rows: the baseline, then
  the models on time and, after a `\midrule`, the late models, each in order of scoring
  start (the paper has no shared `tables/models.tex` yet, so the order comes from the
  scoring starts of the 02-forecasts run, which agree with the release order for all eleven
  models today; the paper names are `src.benchmark.MODEL_NAMES`, shared with the other
  tables). The baseline row covers every decision (see Forecasts).

`run.py` writes `metrics.parquet` (one row per model, group, arm and decision: `model`,
`group`, `representation`, `asset`, `market`, `window`, `time`, `bar`, `closed_hours`,
`closure_day`, `label_time`, `label`, `last_close`, `sigma`, `crps`, `is_covered`,
`is_valid`, `is_scored`, `closed_share`, and the same four scores with the quantiles sorted:
`crps_sorted`, `is_covered_sorted`, `is_valid_sorted`, `is_scored_sorted`) and `tables/summary.csv` (one row per
model, group, arm and split: `status`, `short_contexts` (the group's count over all scored
folds, the same on every row of the group), `decisions`, `days`, `times`, `invalid`, `crps`,
`coverage`, `closed_share`). `run.py --models` runs a subset of the models, only on the
development folds (the baseline `ewma-normal` is always added), and the run's
`tables/models.csv` names the models it ran (`--models` plus the baseline; its
`config.toml` is the whole `study.toml`; a rerun passes them as `--models`); the test run
forecasts every model. `analyze.py` analyses the models the
run built (those in its `summary.csv` that were not released after the scored folds; those
stay in the Q1 evidence). The Q1 evidence lists the decisions each group dropped for a short
trading-hours context and the invalid forecasts per
model and arm (an invalid forecast in one representation removes the decision from all of
them), and its limitations name the small-sample size of the test when a tested model has
fewer than 80 closure days. `analyze.py` writes
`tables/verdict.csv` (every test: `model`, `alternative`, `family`, `quarters`, `days`,
`reference_crps`,
`alternative_crps`, `skill`, `skill_low`, `skill_high`, `block`, `dm_lags`, `dm_statistic`,
`p_value`, `status`, `adjusted_p`, `is_better`, `is_worse` and the model's `verdict`),
`tables/best.csv` (`model`, `closed_best`, `control_best`, `is_same`),
`tables/robustness.csv` (the sorted-quantile robustness row: per model and alternative the
scored decisions, closure days and skill of the primary rule and with sorted quantiles) and
the results README.

## Data

The T1 assets of the 02-forecasts run in `forecast_cache` (its `assets.csv`, from the
01-universe run it names): 16 forex pairs, 8 commodities, 8 indices and 32 stocks (16 US
stocks in their regular session, 09:30-16:00 New York, and 16 non-US) for Q1, and the 8 T1
crypto assets for the control; 1h bars. The scored folds of `[folds] use`; every zero-shot
model only after its scoring start.

Coverage of a development run (d1-d4, 2024-11 .. 2025-10): Chronos-Bolt small and base from
2024-11-25 (on time, 3 quarters wholly after: d2-d4), TiRex 1.1 from 2025-05-26 (1 quarter),
Moirai 2.0 from 2025-08-06 and TimesFM 2.5 from 2025-09-02 (0) have decisions but are late;
Chronos-2 (released 2025-10-30) has two days, fewer than 20 closure days, and is not tested; TimesFM 3.0,
TiRex-2, both Toto 2.0 models and t0 are released after d4, are not built and are reported as
released after the scored folds. Only the test run (t1-t4) covers all eleven models; with the
scoring starts of the current 02-forecasts `study.toml`, both Toto 2.0 models (2026-04-14 and
2026-05-12), TiRex-2 (2026-07-02), TimesFM 3.0 (2026-08-24) and t0 (2026-10-15) are late
there; the other six are on time. t0 has about 14 closure days before t4 ends, fewer than
20, so it is not tested: at most 10 models are tested in the test run.

## Metrics

- CRPS of the vol-scaled return of the target bar from the 9 quantiles (levels 0.1 .. 0.9):
  twice the mean pinball loss, `src.evaluate.quantile_crps`; per model, group and arm the
  mean over closure days (`closure_day_means`).
- 10-90 % coverage (the label between the 0.1 and the 0.9 quantile), averaged the same way.
- Per model, group and arm: scored decisions, closure days, decision times, invalid
  forecasts, the share of closed hours in the contexts and the `status` (scored, not scored
  or not applicable, with the reason); per group the decisions dropped for fewer than
  `min_own_closes` own closes (`short_contexts`).
- Descriptive splits in `summary.csv` (`split`): `all` (pooled, the primary metric), each bar
  after the reopening (`bar 0`, `bar 1`, `bar 2`) and each kind of closure (`night`, under
  24 hours; `weekend`, 44 hours or more, so the 47-hour forex weekend at the spring
  daylight-saving switch counts; `other`, 24 to 44 hours, e.g. a one-day stock holiday of
  about 41 hours from the 16:00 close to the 09:00 open of the half bar).
  The splits never change an answer.
- Per test: the skill with its 95 % interval, the Diebold-Mariano statistic and its one-sided
  p-value, the Holm-adjusted p-value within the family, the bootstrap block and the
  Newey-West lags.
- Only the first 3 bars after a closure are scored here; the CRPS over all hours in event
  time is the benchmark's (02-forecasts and 03-rq1-null).

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
(after each model's release, see the paper's shared design) runs once for the final numbers.
The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions).

## Compute

About 30,000 decisions per year of folds (16 forex pairs x about 52 reopenings x 3 bars;
32 stocks x about 250 nights and weekends x 3; 8 indices and 8 commodities about 52 x 3
each; the crypto control 8 x 52 x 3), each forecast once per arm: about 120,000 one-step
forecasts from 512 values per model (90,000 for the four models without the missing arm).
At the benchmark's speed (02-forecasts: about 2.8 million contexts per model; about an
hour for the three Chronos models together and three hours for TimesFM 2.5 on the local
RTX 3500), each model needs a few minutes plus loading: about 2 to 10 GPU minutes per Chronos
model, 10 for TimesFM 2.5, an untimed few minutes for the 2026 models (TimesFM 3.0, TiRex,
Toto 2.0, t0) and Moirai 2.0 (worker process), so about an hour for all eleven models on the
local GPU, no cost. `estimated_hours` is 1.5 (the runner stops a run at twice that). CPU:
building the contexts takes minutes; the contexts need about 0.17 GB of memory per 10,000
decisions; the bootstrap intervals take seconds.

A development run builds only the six models with decisions in d1-d4 (see Data), most of them
on part of the folds: about 0.3 GPU hours, well inside `estimated_hours`.

## Status

Plan locked on 2026-11-17 (the owner's go, 20:06 UTC) for the test folds t1-t4, with the
02-forecasts test cache `test-2026-11-17` (`study.toml`). The development run
`dev-2026-11-17` uses the same plan with `[folds] use = "development"`, `forecast_cache =
"dev-2026-11-17"` and `locked = false`, from a copy of this folder (both runs on one vast.ai
RTX 4090, one model on the GPU at a time). No result was seen before the lock.

Runs on 2026-11-17 (UTC): `dev-2026-11-17` on a vast.ai RTX 4090 (code at 1236ce1, 554 s; Q1
no, Q2 no on d1-d4) and `test-2026-11-17` on a vast.ai RTX 4090 (code at 6be9225, the
instance had no `.git`; 1,334 s): Q1 no, Q2 no ([results](results/README.md)). A first test
attempt stopped in toto-2-22m (Toto's patch of 32, fixed in 831822a) and a first development
attempt with a CUDA out-of-memory error (fixed in dc84434); neither scored anything. A rerun
of the test folds on the local RTX 3500, `test-2026-11-17-sorted` (6be9225, 2,419 s), checks
the robustness row.

New on 2026-11-16 (from the question list). Plan waiting for the owner's approval. Coded and
tested on synthetic data on 2026-11-16, recoded to the decisions D1-D9 on 2026-11-17
(`tests/experiments/01-benchmark/08-closures/test_closure_contexts.py`, `test_closures.py`,
with `run.py` and `analyze.py` end to end); no run, and
`forecast_cache` is empty until 02-forecasts has run.

Not coded, waiting for the owner:

- **Kronos** (kronos-small, kronos-base; left out of `[models]`): the benchmark gives it real
  candles with their timestamps, so its calendar-time arms need a decision on what a closed
  hour holds (a flat candle at the last close with zero volume?) and its missing arm is not
  possible (Kronos refuses NaN input, `vendor/Kronos/model/kronos.py`).
- **Joint inputs** (a closed-market asset next to BTC in one multivariate model): no wrapper
  takes several series as one multivariate input (`ChronosForecaster` gives Chronos-2 one
  variate per series). The smallest path would be BTC as a past covariate of the asset in
  calendar time, e.g. through the covariate interface that 07-rq5-off-hours adds.

The next three paragraphs are history; where they differ from the Design above, the plan
changes of 2026-11-17 below replaced them.

Changes made while coding, before any run: the decision rule is unchanged; Q1's yes / no
(at least one model with a winning representation), Q2's "same ranking" (every model, by mean
CRPS) and the Diebold-Mariano horizon (`reopen_bars`) are this study's reading of the rule,
for the owner to confirm; the data uses all 16 T1 crypto assets (the plan said 8); the
options `lookback` and `levels` were removed (the lookback is the benchmark's, the levels are
`src.models.QUANTILE_LEVELS`); `forecast_cache`, `alpha` and `missing_not_applicable` were
added.

Changed on 2026-11-16 after the second code review, before any run: the benchmark's freshness
limit (`max_age_hours`) is gone. It had no effect on the benchmark's own decisions, and here
it only dropped the reopenings after a closure longer than 96 hours (long holidays), which
belong in this study; such reopenings are kept now. The closures come from the closure finder
shared with 07-rq5-off-hours (`src.utils.closures`).

Changed on 2026-11-16 after the third code review, before any run: the dropped arm holds the
own closes of the same 512 calendar hours as the other arms (before: the last 512 own closes,
a longer history); Q2 compares each model's best representation on both groups (before: the
whole ranking); the differences get block-bootstrap intervals; Q1 is inconclusive when no
model was tested; a model without a decision after its release is not built; the option
`missing_not_applicable` moved to `src.models.missing_value_reason`.

### Plan changes

2026-11-17, before any run (decisions D1-D9 of the RQ5 review thread, on the owner's
delegation; no result was seen):

- D1: the control is the 8 crypto assets of the universe (the README said 16; the universe
  has 8). The Data counts now match the universe (8 crypto, 16 forex, 8 commodities, 8
  indices, 32 stocks).
- D2: "trading hours only" (the former "dropped") stays the confirmatory arm on the same
  512 calendar hours; the benchmark input (the last 512 own bars) is added as a descriptive,
  untested arm, so the reference can be set next to the main results' input.
- D3: three arms only, no BTC arm; weekend crypto as an input is 07-rq5-off-hours.
- D4: trading hours only is the reference, fixed in advance; each model tests forward-filled
  and missing against it, one-sided (before: the best by mean CRPS against the others,
  two-sided). Bold and verdict: the reference unless an alternative is significantly better
  after Holm (if both, the larger skill).
- D5: models with fewer than 3 of the 4 scored quarters wholly after their scoring start form
  a separate Holm family with their own verdict (option `min_quarters`; the quarter count of
  rq1, `src.benchmark.full_quarters`).
- D6: losses are averaged per closure day (the UTC day the closure ends); the test and the
  interval run on that daily series, with Newey-West lags `max(0, Andrews)` and a
  Politis-White block (before: per decision time, Newey-West over `reopen_bars - 1` lags and
  a 24-hour block). A model is tested only with at least 20 closure days, the same gate for
  test and interval (option `min_closure_days`; before: more than `2 x reopen_bars` decision
  times). The options `bootstrap_block_hours` and `bootstrap_min_block` are gone. The block
  floor is the lab's `src.stats.BLOCK_OVERLAP_FLOOR` (4 times the label overlap since
  2026-11-17, before 2); with overlap 0 it leaves the minimum block of 1.
- The skill `1 - CRPS_alt / CRPS_ref` with its 95 % interval (`skill`, `skill_low`,
  `skill_high`, 2,000 draws) replaces the interval of the CRPS difference (`low`, `high`).
- D7: descriptive splits by bar after the reopening and by kind of closure in `summary.csv`;
  the pooled CRPS stays the primary metric. A weekend is read as 44 hours or more (D7 said 48)
  so that the 47-hour forex weekend at the spring daylight-saving switch is one too.
- D8: the US half bar 09:30-10:00 stays the reopening bar (Design and the limitations say
  so).
- D9: Q2 asks whether the result is consistent with an effect of the representation; H6
  drops its clause on interval width (not tested).
- Q2 rests on the best representation by mean CRPS without an interval or a test (D9 keeps
  it a descriptive consistency check); this is an exemption from experiments rule 13, stated
  in the Q2 limitations. The closed-market skills of Q1 carry their intervals.
- Each model's repository is checked against its pinned commit before it is built; the
  scoring start is the 02-forecasts run's (the later of the release date and the upload date
  of the pinned weights).
- `run.py --models` is allowed on the development folds (refused for the test run);
  `--last-days` is refused (the study scores whole folds).
- `make_tab_rq5_closures.py` writes the rows of the paper's table tab:rq5
  (`papers/benchmark/tables/generated/rq5_body.tex`; never run here), from the run's
  summary and verdict.
- A built model without a decision after its scoring start in one group (e.g. the Monday
  control for a model released in the last days of the folds) gets pooled rows with the
  status "not scored: no decision after its scoring start".
- The bootstrap stream of each test is seeded by the model's name (CRC-32) and the
  alternative, so an interval does not depend on the other models of a run.
- Known limit of the test (the shared `src.stats.diebold_mariano`, Newey-West with Andrews
  lags): with independent closure days its one-sided size at alpha 0.05 is about 5 %; with
  dependent days (AR(1), rho 0.3) it is liberal, about 8 % at 40 days, 6.5 % at 250 (a test
  run) and 5.4 % at 500, measured on 4,000 simulations each while coding. The tests check,
  through `closure_tests`, about 5 % with 250 independent days and more than 6 % with 40
  dependent ones. The Q1 limitations always say the test is somewhat liberal with dependent
  days, and name the small sample when a tested model has fewer than `small_sample_days` (80)
  closure days.

2026-11-17, evening, before any run (the owner's answers to the open points, through the
thread):

- O1: a decision needs at least `min_own_closes` (32) own closes in its trading-hours
  context (before: at least 1). 512 calendar hours normally hold about 100 own closes for a
  US stock, at least about 50 for the shortest sessions even across a holiday week, and more
  for forex, indices and commodities, so 32 drops only contexts damaged by a data gap; the
  filter applies to every arm, so all arms keep scoring the same decisions, and the dropped
  decisions are counted per group (`short_contexts` in `summary.csv`, the Q1 evidence for
  every group, the Q2 evidence for the control).
- O2: Q2 stays descriptive (mean CRPS only, no test); the exemption from experiments rule 13
  above stands.
- O3: the shared lag rule of the Diebold-Mariano test stays (rq1); the limitation with the
  measured size above stays, and so does the minimum of 20 closure days.
- O4: the weekend threshold of 44 hours is confirmed. O5: about 30,000 decisions per year
  (Compute) is confirmed.
- O6: the benchmark input does not reuse the 02-forecasts cache; this study forecasts it
  itself (a quarter of its forward passes, about 15 GPU minutes in a test run).

2026-11-17, 20:20 UTC, before any result (the first development attempt on a vast.ai RTX 4090
stopped with a CUDA out-of-memory error in its first call, so it scored nothing): every
predict call now holds at most the benchmark's `chunk` contexts (`moirai_chunk` for Moirai),
as in 02-forecasts; before, all contexts of one group, arm and length went in one call, which
the Chronos wrapper does not batch. The forecasts are the same (tested).

2026-11-17, 21:10 UTC, before any test result was read (the first test run on vast.ai
stopped in toto-2-22m: a 60-value trading-hours context is not a multiple of Toto's patch of
32): Toto's shorter trading-hours contexts are left-padded with their first value to the next
multiple of 32, inside this study, just before the call (Design, "Toto's patch"). A candidate
shared fix for later, in the Toto wrapper (`src.models`): pad with a `target_mask` of False
and `has_missing_values=True`, so the padding is ignored rather than read as prices.

2026-11-17, 21:25 UTC, a plan addendum written before any test number was read (the owner's
delegated choice, through the thread):

- Invalid forecasts are kept as the primary rule (a decision with a forecast that is not nine
  finite, non-decreasing quantiles leaves every representation of that model). Their cause,
  found by re-forecasting the development contexts on the local GPU (RTX 3500): quantile crossing,
  finite quantiles out of order, after a long flat forward-filled tail when the forecast
  spread collapses. The re-forecast: tirex-1.1 forward-filled 1,836 of 14,207 invalid
  (12.9 %; the vast.ai run had 1,818, 12.8 %: GPU numerics), all finite and crossed (median
  drop 114 float32 ulps, about 1e-5 relative; 36 % of the reopening-bar decisions crossed,
  under 2 % of the later bars); moirai-2-small forward-filled 1,519 of 7,842 (19.4 %, all
  crossed), missing 513 (6.5 %; 460 crossed, 53 with a quantile at or below zero, so no log
  return), trading hours only 49 (0.6 %), benchmark input 7 in the run; the missing and
  trading-hours arms of tirex-1.1 had none. The run's counts are in
  `summary.csv` (`invalid`) and in the Q1 and Q2 evidence. Limitation: the dropped decisions
  are the hardest forward-filled ones (the reopening after a long flat tail), which can
  flatter forward-filled.
- A descriptive, untested robustness row: every forecast is also scored with its nine
  quantiles sorted, the monotone rearrangement of V. Chernozhukov, I. Fernández-Val and
  A. Galichon, "Quantile and Probability Curves Without Crossing", Econometrica 78(3), 2010.
  A sorted forecast is valid when all nine label quantiles are finite (all nine price
  quantiles positive); sorting makes every finite crossed forecast valid. `analyze.py`
  reports the skills of forward-filled and missing with sorted quantiles next to the primary
  ones (`robustness.csv`, Q1 evidence); never in Holm, bold or the answers. A run written
  before this change (the development run `dev-2026-11-17`) has no sorted scores and is
  reported without the row. The test run on vast.ai is the primary result; its robustness row
  comes from a later rerun on the local GPU under its own run id. Of that rerun only
  `robustness.csv` is used (its own primary numbers may differ slightly by GPU numerics and
  are not reported), and it is analysed from a copy of this folder so that
  `results/README.md` stays the test run's.
