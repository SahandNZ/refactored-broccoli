# rq1 Null: does any zero-shot model beat the best simple baseline?

## Questions

**Q1** (RQ-null). Does any zero-shot foundation model beat the best simple baseline on any
market at 1, 4 or 24 bars, on data after the model's scoring start?
Decision rule: yes for a (model, market, horizon) cell if its mean CRPS is below the best
baseline's in at least 3 of the 4 scored quarters and the one-sided Diebold-Mariano test
(H1: the model has the lower CRPS; Newey-West lags max(h - 1, Andrews), t with n - 1 degrees of
freedom) is significant after Holm correction over the scored cells of the main family
(adjusted p < 0.05). The main family holds the models with at least 3 of the 4 scored quarters
wholly after their scoring start; it gives the answer, which lists the cells, or "none". The
other (late) models form their own Holm family with its own verdict and each one's MDE,
reported next to the answer, never changing it. The families are formed per scored window
(development or test).
Primary metric: CRPS of the vol-scaled return.

**Q2** (RQ-null). Could the Q1 rule detect a small but real edge, i.e. is a "no" in Q1
adequately powered?
Decision rule: on the scored folds, yes if at least `power_share` (K = 80 %) of the scored
cells at 1 bar (`mde_horizon`) have a minimum detectable effect (MDE) at or below
`mde_threshold_pct` (X, chosen by 09-mde-threshold on the development folds; 2 % of the best
baseline's CRPS until then); no otherwise. K = 50 % and 90 % are reported as sensitivity rows
only.
Primary metric: MDE per cell: the smallest planted CRPS gain over the best baseline that the
Q1 rule detects in at least 80 % of block-bootstrap resamples, in % of the baseline's CRPS.

## Hypothesis

- H1: no model beats the best baseline on the return at any horizon (the null holds).
- H2: at 1 bar the Q1 rule detects a 2 % CRPS gain in most cells of the hourly models; the Kronos
  cells (one decision every 167 hours, about 13 per quarter) may not reach that power.

## Design

Forecasts come from the paper's shared forecast cache: every model and baseline forecasts
once per decision time and asset, and rq1-rq4 only read that cache (see
[../README.md](../README.md#shared-design)). This study adds no forecasts of its own; it reads
the 02-forecasts run named by `forecast_cache` in `study.toml` through
`src.benchmark.read_cache`.

The code is `src.benchmark` (`_decisions.py`, `_scores.py`, `_power.py`), shared with
rq2-rq4, `src.stats` (the tests) and the study's `lib/` (`_verdict.py`, the decision rule);
`run.py` calls `power_tables` (the cells, their power and their MDE in one function, shared
with 09-mde-threshold):

- **Decisions** (`_decisions.scored_decisions`, the same in every study): a model's valid
  decisions in the scored folds (`[folds] use`: t1-t4 for the paper, d1-d4 while the plan is
  open), a zero-shot model only those made after 00:00 UTC of its scoring start (the later
  of its release and the upload of its pinned weights, `release_date` of the cache's
  `models.csv`). A decision whose forecast is not nine finite, ordered quantiles is left out
  and counted per cell (`invalid`); Kronos's forecast pools its 60 paths (3 seeds x 20).
- **Loss per decision** (`decision_losses`): the CRPS of that forecast from its nine quantiles.
- **Best baseline** (`best_baseline`): per scored fold, market and horizon, the baseline of
  `[models]` with the lowest CRPS on the development folds, on the decisions every baseline
  forecast, averaged as the comparison averages (over the market's assets at each decision
  time, then over the times). With the test folds it is chosen on d1-d4 for every fold; while
  the plan is open (development folds scored), each fold's baseline is chosen on the other
  three folds (leave one fold out), so no fold is scored against a choice made on itself. A
  choice decision made before a scored window whose label ends inside it is never used. The
  choice is the same for every model and reported in the `baseline` column of the tables.
- **Cells** (`null_tables`): for each zero-shot model of `[models]`, each market and horizon,
  the model's decisions and the best baseline on exactly those decisions. The loss of the
  model and of the baseline is first averaged over the market's assets at each decision point
  (the assets of one market move together), every decision point with the same weight: the
  decision time, or for Kronos the 167-hour grid point (`decision_points`), since the assets
  of a market that closes take their first decision after the grid point at different times
  and are not independent points. The cell's mean CRPS, its skill, its winning quarters and
  its test all use that series, unweighted, so their signs agree.
- **Families**: a model is in the main Holm family when at least `main_family_min_quarters`
  (3) scored quarters start at or after its scoring start (`full_quarters`), else in the late
  family (in the test window: Toto 2.0, TiRex-2, TimesFM 3 and t0).
- **Not scored**: every (model, market, horizon) cell has a row in `cells.csv`; a cell without
  any decision (`status` "not scored: released after the scored folds" when the model was
  released after them, else "not scored: no valid decision") has no test; a cell with fewer
  than five blocks of decision points (its block below; `src.stats.has_enough_blocks`, owner's
  rule of 2026-11-17) is "untested": no test, no skill interval, no MDE, and it is left out of
  its Holm family. Holm counts only the tested cells.
- **Test** per cell (`src.stats.diebold_mariano`): on the series in time order, the overlap of
  the cell's labels (`overlap`, `_decisions.cell_timing`: for each decision point, the number
  of later points of the cell before its label time, the median rounded up; 23 for hourly
  crypto at 24 bars, 0 for Kronos) gives the floor of the Newey-West lags, which are
  `max(overlap, Andrews' AR(1) automatic lags)` (`dm_lags` reports them), with Bartlett
  weights, the Harvey-Leybourne-Newbold correction for `h = overlap + 1` and Student's t with
  n - 1 degrees of freedom; the p-value is one-sided (`p_value`, H1: the model has the lower
  CRPS). A loss difference without variance has no test (NaN).
- **Bootstrap block** (`block`): the Politis-White automatic block of the cell's loss
  difference (`src.stats.politis_white_block`, circular block), at least four times the
  overlap (`src.stats.BLOCK_OVERLAP_FLOOR`) and at least 2.
- **Skill** per cell: `skill = 100 (1 - model_crps / baseline_crps)` with a 95 % interval
  (`skill_low`, `skill_high`): skill +- t(n // block - 1, 0.975) x the standard error of the
  skill over `skill_resamples` (2000) circular-block bootstrap draws of
  the cell's decision points, with the cell's block (`src.stats.circular_block_indices`), each cell its
  own stream of `bootstrap_seed`.
- **Decision rule** (`null_verdict` in the study's `lib/`, applied by `analyze.py`): Holm
  within each family over its scored cells; a cell beats the baseline if it wins at least
  `min_winning_quarters` (3) of the scored quarters, its adjusted one-sided p is below `alpha`
  (0.05) and the test statistic favours the model.

- **Power** (`src.benchmark.power_table`, Q2): in every cell a planted model gets, at each
  decision point, the best baseline's loss plus the real model's loss difference to it with
  its mean (over the cell's points) taken out, minus a gain of g % of the baseline's mean
  CRPS, for g in `power_improvements_pct` (0 to 10 in steps of 0.5). So the planted model has
  the cell's real points, number of assets, noise and autocorrelation and a true edge of
  exactly g % (g = 0: no edge). It is a planted signal of known size, not a forecast. The
  cell's points are resampled `bootstrap_resamples` (1000) times with a circular-block bootstrap
  (the cell's block; each cell its own stream of `bootstrap_seed` = 0; a point keeps its
  quarter), and each resample is judged by the Q1 rule: at least 3 winning quarters and a
  one-sided Diebold-Mariano test (the cell's overlap, the resample's own Andrews lags) in
  favour of the planted model with p x m < 0.05, where m is the number of scored cells of the
  cell's family. p x m is Holm's adjusted p of the most significant cell, Holm's smallest
  adjustment; using it for every cell is conservative (Bonferroni), so the real rule has at
  least this power. The detection rate is the share of resamples where the cell is detected.
  With g = 0 it must stay near the one-sided level 0.05 / m (a built-in size check, reported
  as `zero_rate` next to `false_positive_level`). A gain shifts every difference by the same
  constant, so each resample's long-run variance is computed once for all gains.
- **MDE** (`src.benchmark.minimum_detectable_effects`): per cell the smallest g whose
  detection rate reaches `power_target` (0.8), by linear interpolation between the two grid
  points around the first one that reaches it; "never reached" (above 10 %) otherwise.

- **Q2 rule** (in `analyze.py`): the share of the scored cells at `mde_horizon` (both
  families) whose MDE is reached and at most `mde_threshold_pct`, against `power_share`
  (0.8) and the sensitivity shares `power_share_sensitivity` (0.5, 0.9). Every option, the
  threshold included, comes from the run's `config.toml`: 09-mde-threshold sets the threshold
  before the test run. The answer also reports the size check.

`run.py` writes `tables/cells.csv` (one row per model, market and horizon, with `status`,
`family`, `full_quarters`, `times`, `spacing_hours`, `overlap`, `block`, `dm_lags`,
`dm_statistic`, `p_value`, `skill`, `skill_low`, `skill_high` and `invalid`),
`tables/quarters.csv` (the same per fold), `tables/power.csv` (one row per scored cell and
planted gain: `detection_rate`, `planted_crps`, `baseline_crps`) and `tables/mde.csv` (one row
per scored cell: `mde_pct`, `is_reached`, `zero_rate`, `false_positive_level`); `analyze.py`
writes `tables/verdict.csv` (cells with `adjusted_p` and `beats_baseline`), `tables/skill.csv`
(the skill table the paper's Table rq1 is drawn from), `tables/mde_threshold.csv` (the share
of cells with MDE at or below each X of `mde_threshold_grid_pct`, over all and per horizon),
`figures/mde_threshold.pdf` (that share, one line per horizon, the threshold X marked; drawn
from `tables/mde_threshold.csv`), `tables/power_rule.csv` (the Q2 rule at K and the
sensitivity shares) and the results README.

## Data

The universe of 01-universe (72 assets over five markets; run `2026-11-17T0005-random-1000`),
1h bars; the post-release test windows of the shared design.

## Metrics

- CRPS of the vol-scaled return from the 9 quantiles (levels 0.1 .. 0.9): twice the mean
  pinball loss, `src.evaluate.quantile_crps`; per cell and per fold, the unweighted mean over
  decision points of the cross-asset mean loss at each point.
- Skill per cell, `100 (1 - model CRPS / baseline CRPS)`, with its 95 % t interval from a
  circular-block bootstrap standard error.
- One-sided Diebold-Mariano statistic and p-value per cell on that series; Holm-adjusted
  p-values within each family.
- Winning quarters per cell: scored folds where the model's mean CRPS is below the best
  baseline's on the same decisions.
- Per cell: decisions, decision points, their spacing, the overlap, the lags, the bootstrap
  block, the invalid decisions left out, the family and the `status` (scored or why not).
- Detection rate per cell and planted gain, and the MDE per cell (% of the best baseline's
  CRPS); the detection rate without an edge against its expected level 0.05 / m.

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
(after each model's release, see the paper's shared design) runs once for the final numbers.
The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions). Q2's threshold is chosen on the
development folds only, by the pre-registered rule of
[09-mde-threshold](../09-mde-threshold/README.md) (the smallest X of the grid at which at
least K = 80 % of the 1-bar cells of d1-d4 have an MDE at or below X), and copied into
`mde_threshold_pct` before the test run; Q2 runs on the same folds as Q1. The rule's settings
(`alpha`, `min_winning_quarters`, `main_family_min_quarters`, `skill_resamples`, the power and
bootstrap options, `mde_horizon`, `mde_threshold_grid_pct`, `power_share`,
`power_share_sensitivity`) live only in this study's `study.toml`; 09-mde-threshold reads them
from here.

## Compute

Minutes on CPU once the shared forecasts exist (their GPU hours are in the paper README,
[Compute of the shared forecasts](../README.md#compute-of-the-shared-forecasts)). The power
analysis computes each resample's long-run variance once (by FFT) for all 21 planted gains.

## Status

Results (2026-11-17): the answers in `results/README.md` are those of the locked test run
`test-2026-11-17` (t1-t4). The development runs are kept as evidence next to it, with their
tables: `results/dev-2026-11-17/` (all 18 models; Q1 no, Q2 yes) and
`results/dev-2026-11-17-nokronos/` (before Kronos was built; Q1 no, Q2 yes).

History (the state before the runs). Split out of 02-zero-shot on 2026-11-16 (one study per
question, owner's rule). Plan waiting for the owner's approval. The scoring and `run.py` /
`analyze.py` are coded and tested on a synthetic cache (`tests/benchmark/test_scores.py`,
`tests/experiments/01-benchmark/03-rq1-null/`);
no run yet, the shared forecast cache does not exist yet (`forecast_cache` is empty). Q2
(power, owner's request, 2026-11-16) is coded and tested on synthetic data
(`tests/benchmark/test_power.py`); its threshold comes from 09-mde-threshold, 2 % until then.
Changed on 2026-11-16 after an independent code review, before any run: one decision rule
for every study (valid forecasts, seeds' quantiles averaged, cells not scored reported),
lags and bootstrap blocks from the spacing of the decisions, the leave-one-fold-out baseline
choice in development, the cell CRPS as an unweighted mean over decision times, and the Q2
rule above (knee fraction of the share at 10 %, in place of "every 1 h cell").

Changed again on 2026-11-16 after a second independent review, before any run: the lags and
blocks count the overlap of each cell's labels directly, the best baseline is chosen by
per-time means and never on a label that reaches into the scored window, Q2 compares the
threshold with the knee (`threshold_shares`, `knee_threshold`), and the threshold is read from
the run's `config.toml`.

Changed on 2026-11-17 after a third review, before any full run (owner's decisions): the
power question is an absolute rule (K = 80 % of the 1-bar cells with MDE at or below X, with
K = 50 % and 90 % as sensitivity rows) in place of the knee; the main Holm family holds the
models with at least 3 of 4 quarters after their scoring start, the late models get their own
family and verdict; the Diebold-Mariano p-value is one-sided, with Newey-West lags max(h - 1,
Andrews), a t(n - 1) reference and NaN for a difference without variance; the bootstrap block
is Politis-White with a floor of twice the overlap; the skill gets a 95 % block-bootstrap
interval; Kronos's decisions of one grid point are one point; each cell draws from its own
seed stream; the planted gains are 0 to 10 % in steps of 0.5; horizons are counted in bars;
the per-market MDE figure is dropped (the share figure is drawn from this study's mde.csv).

Changed again on 2026-11-17 (owner's decisions, before any scored run): a cell needs five
blocks of decision points to be tested (`src.stats.has_enough_blocks`), else it is
"untested" and left out of its Holm family; the skill interval and the power resamples use
the circular-block bootstrap of `src.stats` (the shared bootstrap fix); the power share and
09's threshold count only the cells with enough quarters for the rule (`detectable_cells`);
each cell draws from a stream named by the cell (`cell_stream`); a family with no tested cell
answers inconclusive.

Changed on 2026-11-17 (owner's decision for every study, before any scored run): the block
floor is four times the labels' overlap (`src.stats.BLOCK_OVERLAP_FLOOR`; it was twice);
the skill interval is skill +- t(n // block - 1) x its circular-bootstrap standard error.
