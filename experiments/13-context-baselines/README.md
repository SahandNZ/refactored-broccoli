# 13 Context baselines: rq1 against baselines fitted on the models' 512-bar context

Exploratory check of rq1 (03-rq1-null), planned after rq1's test results were seen (owner's
approval, 2026-11-18). It never changes rq1's answer; the paper reports it in the appendix as
exploratory.

## Questions

**Q1** (RQ-null). At 1 bar, does any zero-shot model of rq1's main family beat the best baseline when the random walk, climatology, GARCH(1,1) and the ridge regression are fitted only on the last 512 own bars before each decision?
Decision rule: rq1's rule (03-rq1-null Q1: lower mean CRPS in at least 3 of the 4 test quarters and a one-sided Diebold-Mariano test significant after Holm, adjusted p < 0.05) at horizon 1, the best baseline chosen per market on d1-d4 among the five baselines with the four fitted ones refitted on the context; one Holm family over the tested cells of the eight models; yes if any cell passes, no otherwise.
Primary metric: CRPS of the vol-scaled return.

## Hypothesis

- H1: the zero-shot models still lose; the longer fitting history of rq1's baselines is not
  what keeps them ahead.

## Design

- **Cache**: the test forecast cache of 02-forecasts named by `forecast_cache`; its labels of
  d1-d4 and t1-t4 at horizon 1 (one decision per own bar, in order) and the zero-shot models'
  and EWMA-normal's forecasts. EWMA-normal fits nothing, so it is the cache's. The rule is read
  from 03-rq1-null's `study.toml` (copied into the run as `rq1-config.toml`).
- **Context baselines** (`context_forecasts`), per asset on the `window` (512) decisions
  before each decision, whose labels all closed by then (the zero-shot models' context):
  - random walk: normal deciles with the root mean square of those labels;
  - climatology: their empirical deciles (linear interpolation);
  - ridge regression: refitted at every decision on those decisions' lags (the vol-scaled
    log returns over the last 1, 2, 4, 8 and 24 own bars, from the cache's horizon-1 log
    returns; a lag across a decision the cache left out at a fold's end is missing), penalty
    1, no intercept, plus the deciles of its residuals there;
  - GARCH(1,1): zero mean, normal errors, refitted once a day (UTC) on the 512 returns before
    the day's first decision, then filtered through the day; a day whose fit does not converge
    has no GARCH forecast (counted in `garch_not_fitted.csv`), as rq1's not-fitted rule.
  A decision with fewer than 512 earlier decisions gets no context forecast (the first months
  of d1 on markets that close).
- **Test** (`src.benchmark.null_tables` on `ContextCache`): rq1's cells at horizon 1 with the
  context baselines in place of the cache's, the best baseline chosen on d1-d4.

## Data

The development and test folds (2024-11-01 to 2026-10-31) of the 02-forecasts cache
`test-2026-11-17`, 72 assets, horizon 1.

## Metrics

`cells.csv` (rq1's columns), `verdict.csv` (with `adjusted_p` and `beats_baseline`),
`compared.csv` (each cell's skill against the context baseline next to its skill in rq1) and
`garch_not_fitted.csv`.

## Decision rule

Q1 as under [Questions](#questions). Nothing here changes rq1's answer.

## Compute

CPU only, minutes (the daily GARCH fits run in `workers` processes).

## Status

Q1 planned and run on 2026-11-18 (exploratory, after rq1's test results); the results cite
`test-2026-11-18-v2`.

### Plan changes

- 2026-11-18: the first run, `test-2026-11-18`, was started from `study.toml` before this
  README existed (an error against experiments rule 2; its run has no plan hash). A review
  then found that a ridge lag or a GARCH series could span a decision the cache leaves out at
  each fold's end (about 7 per asset on the markets that close); a ridge lag across such a gap
  is now left out. Rerun as `test-2026-11-18-v2` with this README in place, the run the
  results cite; `test-2026-11-18` stays as the first run (Q1 no, median skill -9.1 %).
