# rq2 Centre vs scale: is the skill in direction or in risk?

## Questions

**Q1** (RQ-center-vs-scale). Do the models' medians predict the sign of the return better
than climatology does on the same decisions?
Decision rule: yes for a cell if its hit rate is above climatology's on the same decisions
and the one-sided circular-block bootstrap t test of the difference (H1: the model is
better; at least 5 blocks of decision times, otherwise the cell is untested) is significant
after Holm correction within the cell's family (adjusted p < 0.05), and the rank IC of the
median is positive in at least 3 of the 4 test quarters; Q1 is yes if any cell of
the main family passes; no otherwise.
Primary metric: hit-rate difference, the sign hit rate of the median minus climatology's.

**Q2** (RQ-center-vs-scale). Are the models' forecast intervals calibrated, and do they track
risk better than an EWMA scale?
Decision rule: a cell is calibrated if the coverage of its 10-90 % interval is within 0.05 of
0.80 in at least 3 of the 4 test quarters; its scale is better than EWMA if its coverage error
and its PIT Kolmogorov-Smirnov distance are both below the EWMA-normal baseline's in at least
3 of the 4 test quarters; Q2 is yes if any cell is both; no otherwise.
Primary metric: coverage error of the 10-90 % interval, |coverage - 0.80|.

## Hypothesis

- H2: whatever skill exists is in the scale (risk), not in the centre (sign). H2 has no
  verdict of its own: it is read from the answers to Q1 and Q2.

## Design

Forecasts come from the paper's shared forecast cache: every model and baseline forecasts
once per decision time and asset, and rq1-rq4 only read that cache (see
[../README.md](../README.md#shared-design)). This study adds no forecasts of its own; it reads
the 02-forecasts run named by `forecast_cache` in `study.toml`
(`experiments/01-benchmark/02-forecasts/runs/<forecast_cache>/`) through
`src.benchmark.study_cache`.

The code is the study's `lib/` (`_centre_scale.py`, `_bootstrap.py`, `_figure.py`,
`_settings.py`; the figure itself is drawn by `src.figures.draw_centre_scale`), built on the
shared scoring of `src.benchmark` and `src.stats`, with the same
choices as rq1 where the plan does not say otherwise:

- **Scored decisions** (`centre_scale_tables` on `scored_decisions`, as in rq1): every model
  of `[models]` (or of `--models`) on its valid decisions in the scored folds (`[folds] use`:
  t1-t4 for the paper, d1-d4 while the plan is open); a zero-shot model only after 00:00 UTC of
  its scoring start (the later of its release date and the upload of its pinned weights,
  02-forecasts). A decision with an invalid forecast is left out and counted per cell
  (`invalid`); Kronos's forecast pools its 60 paths (3 seeds x 20) in the cache, so every
  decision has one forecast, as in rq1. The two references are scored on
  exactly the same decisions: `scale_reference` (EWMA-normal) for Q2 and `centre_reference`
  (climatology, whose median is the median of the asset's training labels) for Q1; a decision
  either did not forecast is left out and counted per cell (`unpaired`). The scale reference
  is always scored too, so the figure can show it. A decision whose label ends after its
  quarter is not in the cache, so it is never scored. Baselines are in the tables for
  reference; the decision rule judges the zero-shot models.
- **Equal weight per decision time.** The hit rates and the CRPS of a cell or quarter are
  means over its decision times of the means over the market's assets at that time (the
  assets of one market move together, as in rq1's test). As in rq1, a decision time is a
  decision point (`src.benchmark.decision_points`): for Kronos, which decides on a 167 h grid,
  the grid point, whose assets decide at different times when their market closes. The PIT
  shares, the coverage and the KS distance are plain means over decisions.
- **Centre**: a decision has a sign only when its label, the model's median (`q50`) and the
  centre reference's median are all non-zero; a model whose median is always zero (EWMA-normal,
  random walk) has no hit rate and no test. Climatology's median is the median of the asset's
  training labels, so it is exactly zero for an asset with many zero training returns (mostly
  FILUSDT, RUNEUSDT, NEARUSDT, EGLDUSDT, 0700.HK, 0388.HK, EURGBP and AUDNZD in the dev
  cache); those decisions leave the centre test, which then rests on fewer decisions of those
  markets. Each cell reports the share it loses this way, `zero_reference_share` in
  `tables/cells.csv`, `tables/quarters.csv` and `tables/verdict.csv`, and the decisions it
  keeps, `signed_decisions` (which also leaves out zero labels and zero model medians). The
  rule depends only on the training period, so it does not bias the test; counting a zero
  median as chance instead would let a median that follows a drift pass (owner's decision,
  2026-11-17). On those decisions
  the hit of the model is whether
  its median has the label's sign, and the reference hit the same for climatology; the
  per-time difference of the two shares is the series the cell tests. A target that drifts
  has a chance level other than 0.5, and an always-positive median then hits as often as
  climatology does: the difference, not the hit rate against 0.5, carries the skill.
- **Block and test**: every interval and the test of a cell use one circular-block bootstrap
  (`src.stats.circular_block_indices`; a block may wrap around the end of the series) with
  the cell's `block`: the Politis-White block (`src.stats.politis_white_block`, arch's
  automatic choice) of the tested series (the hit difference per decision time; the coverage
  per time for a cell without signs), at least four times the overlap of its labels
  (`src.stats.BLOCK_OVERLAP_FLOOR`; owner's rule, 2026-11-17)
  (`cell_timing`: the median number of later decision times inside a decision's label) and
  at least 2, as in rq1. A cell with fewer decision times than its block, or without any
  decision, is not scored (`status`) and passes no rule. A bootstrap test or interval needs
  at least `src.stats.MIN_BOOTSTRAP_BLOCKS` (5) blocks of the decision times it resamples
  (owner's rule, 2026-11-17): a scored cell with fewer decision times with a sign has no
  centre test (NaN p, outside Holm; `centre_test` says why, and its hit rate is grey in the
  figure), and one with fewer decision times has no coverage interval. A difference without
  spread (the two medians always agree in sign; `src.stats.is_constant`) has no test either
  and is marked untested. A cell not scored keeps its row in `tables/rq2.csv` but has no mark
  in the figure. The test is `src.stats.bootstrap_mean_test`, as in rq1: one-sided (H1: the
  model's hit rate is above climatology's), `d / se` read against Student's t with
  `n // block - 1` degrees of freedom (n the decision times with a sign), with `d` the mean
  difference and `se` the standard deviation of `bootstrap_resamples` (2000) circular-block
  bootstrap means (`bootstrap_seed`, one independent stream per model and cell, seeded from
  their names, so a cell's draws do not depend on the other models of a run). On synthetic
  AR(1) series (1000 series, 1000 draws) a 5 % test on exactly 5 blocks rejected a true null
  5.9-6.7 % of the time with this t when the block is long enough for the dependence
  (9-16 % with a normal reference, owner's choice of the t, 2026-11-17); a block too short for
  it stays liberal (about 10 % at rho 0.5 with block 4, 12.5 % at rho 0.8 with block 10),
  which the automatic block is meant to prevent. On overlapping labels (sums of h
  independent steps, overlap h - 1; 4000 series, 1000 draws) the floor of four times the
  overlap gives 6.8 % at h = 4 and 5.8 % at h = 24, against 7.8 % and 7.3 % with the earlier
  floor of twice the overlap; `test_the_overlap_floor_keeps_the_size_of_overlapping_labels`
  is a size sanity check of the floor at h = 24. Intervals are 95 % t intervals (level
  `1 - alpha`): the estimate plus or minus the t quantile times the bootstrap standard
  error, unadjusted and descriptive only: the hit rate (`hit_low`, `hit_high`), the
  difference (`diff_low`, `diff_high`: it excludes 0 exactly when the two-sided test at
  `alpha` rejects) and the coverage (`coverage_low`, `coverage_high`, `ratio_interval`: each
  draw divides the labels inside the interval by the decisions of the drawn times, so the
  interval belongs to the plain-mean coverage). The test and every interval of a cell use
  its one block and seed: the test and the hit-rate interval resample the decision times
  with a sign, the coverage interval all decision times. The block is tuned to the tested
  series (the hit difference), not to how coverage misses cluster, so the coverage interval
  may be too narrow under volatility clustering. A cell's
  centre is called worse (`has_worse_centre`) when the whole interval of the difference is
  below 0; this is descriptive, not a test. No HAC variance is used in this study.
- **Holm families** (as in rq1): a model is in the main family when at least
  `main_family_min_quarters` (3) scored quarters start at or after its scoring start
  (`src.benchmark.full_quarters`), else in the late family; Holm runs within each family
  over its scored cells with a test, once per family over all the run's cells (each cell
  pools the scored folds, d1-d4 or t1-t4). Q1 is answered by
  the main family; the late family gets its own verdict (`family_has_centre_skill` in
  `verdict.csv`, and a line in the answer's evidence). A late model
  has fewer quarters, so it may not meet the 3-quarter condition. With an empty main family,
  or a main family without any cell with a centre test, Q1 is inconclusive.
- **Centre share**: `centred_crps` is the CRPS of the forecast with every quantile moved by
  minus the median (what is left without the centre); `centre_share = 1 - crps /
  centred_crps` is the part of that CRPS the median removes (0: the median adds nothing,
  negative: it hurts).
- **Scale**: from the nine quantiles, the share of labels at or below each quantile
  (`below_q10` .. `below_q90`, the distribution function of the PIT at the nine levels, i.e.
  the PIT histogram in cumulative form). The 10-90 % coverage is `below_q90 - below_q10`; the
  PIT Kolmogorov-Smirnov distance is the largest gap between these shares and their levels
  (the PIT is known only at the nine levels, so this is the KS distance at those points).
- **Decision rule** (`centre_scale_verdict`, in `analyze.py`), per (zero-shot model, market,
  horizon) cell. Q1: `hit_difference` above 0, Holm-adjusted one-sided p below `alpha` (0.05)
  in the cell's family and a rank IC above zero in at least `min_quarters` (3) scored quarters.
  Q2: calibrated if the coverage error is at most `coverage_tolerance` (0.05) in at least 3
  quarters; better than EWMA if, in at least 3 quarters, both its coverage error and its KS
  distance are below the reference's in the same quarter; Q2 is yes if any cell is both
  (no test, so no Holm).

`run.py` writes `tables/cells.csv` (one row per model, market and horizon, all models of
`[models]`) and `tables/quarters.csv` (the same per fold); `analyze.py` writes
`tables/verdict.csv` (the zero-shot cells with `family`, the quarter counts, `adjusted_p` and
the verdicts), `tables/rq2.csv` and `figures/rq2.pdf` (`fig:rq2` of the paper, at
`figure_horizon` = 1 bar: EWMA-normal, then the zero-shot models in release order; (a) the hit
rate with its interval and climatology's hit rate on the same decisions as a black bar, line
at 0.5, (b) the coverage minus 0.80 with its interval, line at 0; one mark per market, marker
shape = market, filled = passes the centre rule or is calibrated; n per mark is
`signed_times` for (a) and `times` for (b)) and the results README.

Not coded (in the first plan, not in the shared cache or the decision rule): the reflection
check (the forecast of 1/P against that of P needs new forecasts on inverted prices) and the
cross-model agreement of interval widths. Both wait for the owner's decision.

## Data

The T1 universe of 01-universe, 1h bars; the post-release test windows of the shared design.

## Metrics

All on the vol-scaled label, per model, market, horizon and fold, and over all scored folds:

- Centre: hit rate of the sign of the median and climatology's on the same decisions, and
  their difference (means over decision times of the cross-asset shares), with block-bootstrap
  intervals and the one-sided p-value of the difference, Holm-adjusted within each family;
  rank IC (Spearman) of the median with the label, pooled over all decisions (assets and
  times) of the cell or quarter, not cross-sectional per time (`src.evaluate.rank_ic`); CRPS
  and CRPS with the median set to zero (`crps`, `centred_crps`, means over decision times,
  `src.evaluate.quantile_crps`) and the centre share.
- Scale: shares of labels at or below each quantile (cumulative PIT histogram), coverage of
  the 10-90 % interval with its block-bootstrap interval and its error |coverage - 0.80|, PIT
  Kolmogorov-Smirnov distance at the nine levels; the same two for the EWMA-normal baseline on
  the same decisions.

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the test window
(after each model's release, see the paper's shared design) runs once for the final numbers.
The folds are defined in [the README](../../README.md#folds). Each
question's own rule is under [Questions](#questions). Every test is one-sided at `alpha`
(H1: the model is better than its reference); "worse" is read from the interval only.

## Compute

Minutes on CPU once the shared forecasts exist. The 1/P reflection check is not coded (see
[Design](#design)); it would rerun the models on inverted prices, about a third of the shared
forecast cost on the local GPU.

## Status

Split out of 02-zero-shot on 2026-11-16. Plan locked by the owner on 2026-11-17, with the test
folds t1-t4 approved. The scoring and
`run.py` / `analyze.py` are coded and tested on a synthetic cache
(`tests/experiments/01-benchmark/04-rq2-center-vs-scale/`: `test_centre_scale.py` and
`test_rq2_center_vs_scale_study.py`: a model with centre skill but a too narrow spread is
classified as centre only, a calibrated model without sign skill as scale only, and an
always-positive median on a drifting label has no centre skill); no run yet, the shared
forecast cache does not exist yet (`forecast_cache` is empty). The reflection check and the
width agreement are not coded (see [Design](#design)).

Dev run (2026-11-17, `forecast_cache = "dev-2026-11-17"`, results in `results/`): Q1 no,
Q2 yes. Kronos's 10-90 % intervals cover about 0.53 of the labels (the lowest mean coverage
in the Q2 answer); the rq1 thread recomputed all 11,748 pooled Kronos rows of the cache and
found no bug, so Kronos is under-dispersed at its reference sampling settings (temperature 1,
top-p 0.9). The peak VRAM in the run table of a CPU run is the memory other processes held on
the GPU (`src.experiment` counts all of it).

### Plan changes

- 2026-11-16, after an independent code review, before any run: the shared decision rule of
  every study (valid forecasts, cells not scored reported), the bootstrap block from the
  spacing of the decisions, and the rank IC stated as pooled.
- 2026-11-17, after a second review against the paper text, before any run (owner's
  decisions for all scoring studies): the hit rate is tested against climatology's on the
  same decisions instead of 0.5 (a drifting target made an always-positive median pass);
  one-sided tests (H1: better); Holm within a main and a late family; the Politis-White block
  of the tested series (at least twice the overlap, four times since the owner's rule of
  2026-11-17) for every test and interval; a
  block-bootstrap interval of the coverage; the CRPS and the centred CRPS as means over
  decision times with the centre share; `run.py --models`; the figure `fig:rq2` and its table.
- 2026-11-17, owner's rule before any run: a bootstrap test or interval needs at least 5
  blocks, and the one-sided p reads the bootstrap t against Student's t with one degree of
  freedom per block less one; the bootstrap is the shared circular-block one of
  `src.stats` (`bootstrap_mean_test`, as in rq1), with t intervals.
