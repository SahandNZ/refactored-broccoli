# rq4 Leakage: do models remember their pretraining window?

## Questions

**Q1** (RQ-leakage). Does each model do better before its own release than after it, beyond
the change the best baseline shows between the same periods?
Decision rule: yes for a (model, market, horizon) cell (a sign of memorised pretraining data)
if its one-sided p-value (circular-block bootstrap standard error with a Student-t
reference, `src.stats.bootstrap_difference`; H1: the difference in differences is positive)
is below 0.05 after Holm correction over all scored cells of the test `leakage`; the answer
lists the cells, or "none".
Before = the model's decisions in the leakage quarters whose label time is before its
pretraining cutoff and its release; after = its decisions in the scored folds made after its
release, thinned to the decisions the 4-hour grid of the leakage quarters would make (Kronos
on its own 167-hour grid in both).
Primary metric: difference in differences of CRPS skill, the mean skill before minus the mean
skill after, where the skill of a decision is the best baseline's CRPS minus the model's,
averaged over the market's assets at each decision time.

## Hypothesis

- H4: models look better before their release than after. (Kronos, pretrained on market data
  up to 2024-06, is the model where leakage is most plausible; this is not tested: no test
  compares models.)

## Design

Forecasts come from the paper's shared forecast cache: every model and baseline forecasts
once per decision time and asset, and rq1-rq4 only read that cache (see
[../README.md](../README.md#shared-design)). This study adds no forecasts of its own; it reads
the 02-forecasts run named by `forecast_cache` in `study.toml` through
`src.benchmark.read_cache`. The test cache of RQ1 (`test-2026-11-17`) holds d1-d4
and t1-t4 of every model (Kronos included) and l1-l12 of random-walk, climatology,
ewma-normal, timesfm-2.5, timesfm-3, tirex-1.1 and tirex-2 (checked on 2026-11-17 by the
windows of each forecasts file). The l1-l12 of the other models (chronos-bolt-small and
-base, chronos-2, moirai-2-small, t0-gamma, toto-2-22m and -313m, garch-normal, linear,
kronos-small and kronos-base) are built into a second 02-forecasts run, `leakage-2026-11-18`,
with `run.py --windows l1 .. l12` (added by the shared patch of 2026-11-17 together with the
GARCH rule below), named by `leakage_cache` (`lib/_cache.py`, `leakage_cache`). The two
Kronos models were sampled on vast.ai on 2026-11-17 (owner's request; seeds 0-2, 20 paths
each, one window or half a window per unit with the pinned Kronos commit) and their samples
pooled into that run's forecasts, as one `run.py` call would write them; `not_built` is
empty. `join_caches` refuses two runs that differ in the data version,
seeds or Kronos commit of their `meta.json`, in the `[folds]` or `[options]` of their
`config.toml` (not the plan's hash, which is the hash of the 02-forecasts README), or in
their labels, `models.csv` or `assets.csv` (within a relative 1e-9 and an absolute 1e-12,
the rounding of another CPU; times and names exactly); a model's forecasts are its rows of
both runs, each window from one run only (a window in both raises). `require_windows` then
raises if a model rq4 scores has no forecast in a window it reads (d1-d4, the scored folds
and, locked, l1-l12 except for `not_built`). A GARCH(1,1) fit that does not converge
(NATGAS in l3) gives that asset no forecast in that window and is listed by 02-forecasts in
`not_fitted.csv` (owner, 2026-11-17); a single asset left out of a window is not caught by
`require_windows`, and where garch-normal is a cell's best baseline that asset's decisions of
that window cannot be paired with the cell's control and leave that cell (the other assets of
those times stay): `cells.csv` counts them in `unpaired` (decisions, of the test's two periods,
without a valid forecast of the control), `run.py` copies the listed fits into
`tables/not_fitted.csv` of the run, and `analyze.py` names the fits and the cells in the
answer's limitations. With the control chosen on d1-d4 (climatology everywhere but forex 1 and
4 bars, linear, and stock 24 bars, garch-normal) the listed fits of the leakage run touch one
cell: NFLX's l4 decisions (153 per hourly model) leave stock 24 bars; NATGAS l3 (commodity,
climatology) stays. Both Kronos models, where leakage is most plausible (H4), are tested.
The test
cache's `meta.json` records the run's start, not how each of its forecasts files was made
(several were built by scratch drivers or copied in from vast.ai runs, see the 02-forecasts
README), so the join checks what defines the forecasts (configuration, data, seeds, Kronos
commit, package versions), not their history; the forecast code between the two runs'
commits is compared by hand before the run and recorded here. An empty `leakage_cache` reads
the first run alone.

The code is the study's `lib/` (`_cache.py`: the two runs as one cache; `_leakage.py`:
periods, grid and tables; `_compare.py`: the
tests and the verdict), built on the scoring of rq1 in `src.benchmark`
and `src.stats`:

- **Loss per decision** (`decision_losses`, as in rq1): the CRPS of each valid forecast from
  its nine quantiles (`_decisions.valid_decisions`: the validity rule of every study, invalid
  decisions counted per cell). Kronos's forecast pools the 20 paths of each of its 3 seeds
  (60 paths) before the quantiles, in the cache.
- **Release date** (from the cache's `models.csv`): the later of the creation of the model's
  Hugging Face repository and the upload of the weights it loads (02-forecasts
  `release_dates` and `weight_dates`), 00:00 UTC.
- **Control** (`best_baseline`, as in rq1; the owner's choice over a second difference
  against Chronos-Bolt): per window, market and horizon, the baseline of `[models]` with the
  lowest mean CRPS on the development folds (on d1-d4 for the leakage and test quarters;
  while the plan is open, each development fold on the other three), the same for every
  model. A change between the periods that hits the model and the baseline alike (a calmer or
  wilder market) cancels out. The baselines are refitted per window on the 730 days before
  it, so in 2022 they learn from 2020-2021 only; the choice of which baseline is the control
  uses d1-d4, after the leakage quarters (a choice of the control, not an input of any
  forecast).
- **Periods** (`leakage_periods`), per model, from `release_date` and `pretraining_cutoff`
  of the cache's `models.csv` (00:00 UTC of the dates). The label time of a decision is the
  close time of its label's last bar (`label_time` of the cache, `h` own bars after the last
  close at or before the decision):
  - *before*: a decision in the leakage quarters l1-l12 whose label is earlier than the
    cutoff and the release (data the model may have seen);
  - *between*: a decision in l1-l12 or in the development folds d1-d4 made at or after the
    cutoff whose label is earlier than the release (only for a model with a cutoff, today
    Kronos: 2024-07 .. 2025-06; on its 167-hour grid l10-l12 alone give at most 20 times, too
    few to test, d1-d3 add about 36; the baseline of a development fold is chosen on the other
    three);
  - *after*: a decision in the scored folds (`[folds] use`: t1-t4 for the paper) made after
    the release, as in rq1.
  A decision whose label crosses the cutoff or the release is left out.
- **Same grid in every period** (`leakage_grid`). The leakage quarters decide every 4 hours
  from each window's start (02-forecasts `step_hours.leakage`): an asset decides at a point
  only when one of its bars closed since the previous point, with its last close as the
  context end. The development and test quarters decide after every own close. Every period
  of a model that decides every hour keeps, per asset and 4-hour point, the asset's last
  decision in (point - 4 h, point], re-timed to the point before its period is assigned (so
  a decision has the period of its point): the same context, label and
  forecast as the leakage grid would give, for every market (a market that opens at 14:00
  decides at 16:00, as in the leakage quarters); a leakage quarter is on that grid already
  and keeps every decision. The grid is taken from the cache's labels before the validity
  filter and the pairing with the baseline, so an invalid decision is missing, as it would
  be in the leakage quarters; a point at the window's end is left out. One point can differ:
  at a window's start the leakage grid may use a close of the previous window's last hours,
  which the hourly labels of a window do not hold (at most one point per window and asset).
  `grid_hours` = 4 is checked against 02-forecasts by a test. The full hourly *after*
  period is the check `hourly`. Kronos decides every 167 hours in every window, keeps all
  its decisions and has no check `hourly`.
- **Decision times** (`decision_points`, as in rq1): a time is a decision time, and for
  Kronos its grid point, so the assets of a market that decide at different hours of one
  Kronos point (Tokyo, Frankfurt, New York) are one time. A Kronos decision gets its period
  from its grid point (a point after the release is after, a point at or after the cutoff is
  between), so no time is in two periods. The label overlap of a period is counted on these
  times (the last label time of each), as in rq1.
- **Skill** (`leakage_tables`): per decision the best baseline's CRPS minus the model's
  (positive: the model is better), averaged over the market's assets at each decision time
  (the assets of one market move together).
- **Tests** per (model, market, horizon) (`leakage_cells`, `TESTS`): the mean skill of a
  first period minus that of a second, each the mean over decision times.
  - `leakage` (primary, Q1): before minus after on the 4-hour grid, the difference in
    differences. Positive means the model gains more over the baseline on data it may have
    seen than on data after its release.
  - `hourly` (check, models that decide every hour): before minus the hourly after period.
  - `cutoff` (check, models with a cutoff: Kronos): before minus between. Its second period
    gives the per-time mean skill between the cutoff and the release (`skill_second`,
    `second_times`), the same statistic as the difference in differences.
  - `placebo` (check, `placebo_models`: Chronos-Bolt small and base): the before period split
    at `placebo_split` (2024-01-01, decision time), 2022-2023 minus 2024, with a two-sided
    p-value. Both halves are before the release, so a model without a time trend in its gain
    gives zero; a rejection either way warns that the gain over the baseline drifts in time
    without any release (it tests the assumption of the difference in differences only for
    these models, and only inside 2022-2024).
- **Bootstrap** (`compare_periods`): the circular-block bootstrap standard error with a t
  reference of `src.stats.bootstrap_difference` (`bootstrap_resamples` = 2000 draws,
  `bootstrap_seed` = 0). The times of each period are resampled on their own with circular
  blocks of their own length, the rule of rq1: the Politis-White automatic block length of
  the period's skills (`src.stats.politis_white_block`), at least 4 times the period's label
  overlap (`src.stats.BLOCK_OVERLAP_FLOOR`, the owner's rule for every study; the overlap is
  the number of later decision times that share a label, `cell_timing`) and at least 2. The
  difference over its bootstrap standard error is read against Student's t with Welch's
  degrees of freedom from the two periods' block counts; the interval is the matching 95 %
  t interval. The p-value is one-sided (`p_greater`, H1: the first period's skill is
  higher), two-sided for the placebo. `bootstrap_block_hours` and `bootstrap_min_block` stay
  in `study.toml` only because the shared `src.benchmark.read_bootstrap` requires them; rq4
  does not use them.
- **Size check** (synthetic, one-sided at a nominal 5 %, rq4's blocks, 400 draws; no period
  fell below 5 blocks). The test `test_the_one_sided_test_keeps_its_size` (300 series of 600
  and 400 times, Monte Carlo standard deviation 1.3 %) gives 5.3 % for AR(1) skills (phi 0.5,
  started at the first draw) and 8.3 % for skills of labels that overlap 5 later decisions
  (the 24-bar cells on the 4-hour grid, about 24 blocks of 20 per period; the earlier floor
  of 2 overlaps gave 8.7 % and the old moving-block bootstrap with a normal reference 9.7 %,
  on the same series). A separate check with stationary AR(1) series (scratch script, at
  HEAD 5e0f666) gives, per period length (AR(1) / overlapping labels; mean blocks per
  period): 600 and 400 times 8.7 % / 8.3 % (50 / 24; 300 series, standard deviation 1.3 %),
  2,400 times 7.0 % / 8.5 % (135 / 99; 200 series, 1.5 %), 8,800 times (about a year of
  hourly decisions) 6.7 % / 5.3 % (314 / 240; 150 series, 1.8 %); on
  the 4-hour grid of the test run the 24-bar `leakage` cells of the models that decide every
  hour have 302-2,170 times after the release, the 2,400-time case or shorter (about 8.5 %),
  the shortest below the 400 times the check covers (Kronos's 24-bar labels do not overlap
  on its 167-hour grid). Shorter cells are therefore somewhat
  liberal (a limitation of the answer, in `analyze.py`).
- **Not scored**: every (model, market, horizon) cell of each test that applies to the model
  has a row; a cell without decisions in a period, with fewer than 11 times in a period
  (`src.stats.MIN_BLOCK_VALUES`, the least the Politis-White estimate takes) or with fewer
  than 5 blocks in a period (`src.stats.MIN_BOOTSTRAP_BLOCKS`, `has_enough_blocks`, the
  owner's rule for every study) is not scored (`status`, no test, left out of Holm).
- **Not built** (`not_built`; empty in the test run: both Kronos models were built on
  2026-11-17, after kronos-base had first been listed): with the locked plan, every cell
  of a listed model has the status "not scored: not built on the leakage quarters" and no
  test (`did`, `low`, `high`, `p_value` empty; the period columns still show its after
  period); a listed model with any forecast in l1-l12 raises, and the Q1 answer names the
  listed models as not tested.
- **Decision rule** (`leakage_verdict`, in `analyze.py`): Holm separately in each test
  family (`test`); a cell rejects if its Holm-adjusted p (one-sided; two-sided for the placebo)
  is below `alpha` (0.05).
  A cell of `leakage` that rejects shows leakage; the answer is yes if any cell shows
  leakage and lists those cells. The checks are reported and never change the answer. All
  models, early and late releases, are in the one `leakage` family: the rule has no
  quarter count that a late model could not reach, so the separate family of rq1's late
  models does not apply here; a late model has a short after period and a wide interval.
- **Folds.** With the locked plan (`[folds] use = "test"`) *before* is l1-l12 and *after*
  t1-t4. While the plan is open (`"development"`) both come from d1-d4 (before and after
  each release inside them), so the leakage quarters are not looked at before the lock;
  there the baseline of each fold is chosen on the other three folds and before, between
  and after share the same quarters, so a development run checks the code and is not a
  leakage estimate. A model
  with no decision before or after its release in those folds has its cells reported as not
  scored, as in every study.

`run.py` writes `tables/cells.csv` (one row per test, model, market and horizon: `test`,
`first`, `second` (the periods), `first_times`, `second_times`, `first_block`, `second_block`,
`skill_first`, `skill_second`, `status`, `did`, `low`, `high`, `p_value` (one-sided; for the
placebo two-sided), `invalid` (the model's invalid decisions of the cell in every window the
study reads, the same in each test), `unpaired` (the decisions of the test's two periods
without a valid forecast of the cell's control, left out)), `tables/not_fitted.csv` (the
baseline fits 02-forecasts left out, copied from its runs) and `tables/periods.csv` (decisions
and mean CRPS of model and baseline per period, weighted by decision; `after-hourly` is the
hourly after period, only for a model that decides every hour; `early` and `late` split the
`before` decisions of a placebo model, so a sum over periods counts decisions twice);
`analyze.py` writes `tables/verdict.csv` (cells with `adjusted_p`, `rejects` and
`shows_leakage`), `tables/rq4.csv` and `figures/rq4.pdf` (fig:rq4, `lib/_figure.py` and
`src.figures.draw_leakage`: the `leakage` cells at `figure_horizon` (1 bar), one row per model
in the order and with the names of tab:rq1 and fig:rq2 (`paper_order`; a model with fewer
than `figure_family_min_quarters` (3) test quarters wholly after its release goes after the
others, which only orders the rows), one mark per market (legend: the paper's market names),
the difference in differences of the CRPS gain with its unadjusted 95 % interval, filled
where the cell shows leakage, a cell not scored not drawn; the x-axis covers every mark and
every interval but those of a model whose widest interval is more than 3 times the median
model's (`figure_limit`: Kronos at 1 bar), which stop at the edge with an arrowhead; 6.80 in
wide, 1.0 + 0.25 in per model high; `rq4.csv` in the figure's row order, with the cache's market keys and `row_group`, the
model's rq1 family that only orders the rows) and the results
README, which copies `verdict.csv`, `periods.csv`, `not_fitted.csv`, `rq4.csv` and
`rq4.pdf` into `results/<run>/`.

Three windows per model where its dates allow, as above: a memorising model drops at the
cutoff, a regime change need not. A probe on 2026-11-16 (250 decisions per window, 6 assets)
found a difference in differences near zero for Kronos.

## Data

The T1 universe of 01-universe, 1h bars; leakage quarters l1-l12 (2022-01 .. 2024-10, l12 the
one month 2024-10, so they end where d1 starts) and the post-release windows.

## Metrics

- CRPS of the vol-scaled return from the 9 quantiles (`src.evaluate.quantile_crps`), mean per
  model, market, horizon and period (`periods.csv`).
- Skill over the best baseline (baseline CRPS minus model CRPS, in the units of the
  vol-scaled target: the CRPS gain, not the paper's percent skill) per period, and per test the
  difference of two periods' mean skill (for `leakage` the difference in differences, skill
  before minus skill after) with its 95 % t interval (circular-block bootstrap standard
  error), one-sided p-value (two-sided for the placebo) and Holm-adjusted p-value within its
  test family (`cells.csv`, `verdict.csv`).

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
(after each model's release, see the paper's shared design) runs once for the final numbers.
The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions).

## Compute

The leakage quarters add three years of forecasts to the shared cache (sampled every 4
hours); Kronos is the slowest model (see the paper README).

## Status

Split out of 02-zero-shot on 2026-11-16; the after-cutoff window added from the probe. The
plan was locked on 2026-11-17 (`[plan] locked = true`, `[folds] use = "test"`) and the test
run `test-2026-11-18` reads the test cache `test-2026-11-17` joined with the leakage run
`leakage-2026-11-18`, every model of `[models]` built on l1-l12 (both Kronos models
included, `not_built` empty). Answer and limitations: [results/README.md](results/README.md).
The scoring and `run.py` / `analyze.py` are tested on a synthetic cache whose leakage quarters
decide every 4 hours, as in the real cache (`tests/experiments/01-benchmark/06-rq4-leakage/`:
`test_leakage_cache.py`, `test_leakage_periods.py`, `test_leakage_scores.py`,
`test_rq4_figure.py` and `test_rq4_leakage_study.py`). fig:rq4 (`figure_horizon`,
`figure_family_min_quarters`) was added after the lock; it draws the cells of the locked
rule and changes no test. In the test run the hourly check rejects only stock 1-bar cells, those
of every hourly model but t0-gamma (10 of 11; t0-gamma's difference 0.005, adjusted p 1) (differences of 0.004 to 0.036, against -0.003 to 0.004 for the same
cells on the 4-hour grid of the primary test): for stocks at 1 bar the size of the
difference depends on thinning the after period to the grid of the leakage quarters, which
is why the primary test compares periods on that one grid.

### Plan changes

- 2026-11-17, before the lock: an interim run on the test folds
  (`runs/test-2026-11-18-interim/`, written by a scratch script with in-memory overrides of
  the open plan: test folds, both caches, both Kronos models not built while their leakage
  quarters were sampled) answered Q1 "no" (0 of 158 scored cells). It is not a run of the
  runner, so `runs/test-log.csv` has no row for it (the scratch script's path is in its
  `INTERIM.txt`). It changed no rule; the only later change to a rule is `not_built` emptied once both Kronos models were pooled
  into `leakage-2026-11-18`, as the owner asked (Kronos added to the leakage run,
  2026-11-17).
- 2026-11-17: the locked run `test-2026-11-18` ran first at 6be9225 with the locked
  `study.toml` (test folds, both caches, `not_built = []`) not yet committed. Its recorded plan
  (hash 0e73599d, this README as of 2b2957e) still listed kronos-base as not built and not
  tested; the README was brought in line with `not_built = []` in 7dacb9a (with fig:rq4), after that run,
  so the run at 7dacb9a (after fig:rq4 was added) records another plan hash (e6f843fa). It was
  rerun so that its `config.toml` holds the figure's options (`analyze.py` reads the run's
  configuration) and its recorded plan and commit are those of the committed locked plan.
  `runs/test-log.csv` lists both under the same run id (the commit tells them apart); the
  second run's tables are identical to the first's (kept in `runs/test-2026-11-18-first/`).
  Both runs record `git_dirty` true because `git status --porcelain` was not empty: the
  uncommitted `study.toml` in the first, and in both other studies' untracked files.
- 2026-11-17, after the test run (owner's request): fig:rq4 and its options `figure_horizon`
  and `figure_family_min_quarters`; they draw the cells of the locked rule and change no test
  or answer. After the reviews of the figure, `results/` was regenerated by `analyze.py` from
  the tables of `test-2026-11-18` at later commits (the figure's code; see the git log of
  `results/`), with no new test run.

Changed on 2026-11-17, before any run (owner's decisions of the second review round): the
p-value is one-sided (H1: leakage); the blocks are Politis-White per period with a floor of
4 times the label overlap (owner's rule, 2026-11-17; first 2 times), as in rq1 (replacing a
fixed day of decisions), and a period needs
5 blocks (the shared rule); the test is the circular-block bootstrap standard error of
`src.stats` with a t reference and t interval (replacing the normal reference and the
percentile interval; for overlapping 24-bar labels the size stays about the same, see the
size check); the primary after period is the decisions the 4-hour grid of the leakage
quarters would make (every period on that grid), the hourly one a check; between also takes the
development folds before the release; Kronos's decisions of one grid point are one time; the
cutoff test (Kronos, before minus between) and the two-sided Chronos-Bolt placebo are
reported checks, each its own Holm family; H4 no longer claims an order of models. The Kronos
cutoff of 02-forecasts is 2024-07-01 (its paper, Appendix D: pre-training data "up to June
2024", test period "begins in July 2024").

Changed on 2026-11-16 after an independent code review, before any run: the decision rule
and the primary metric above now state what the code does (the owner chose the best baseline
as the control, without a second difference against Chronos-Bolt; Holm over the scored cells
on the bootstrap p-value; "before" by label time); each period's bootstrap block follows its
spacing; cells not scored are reported.
