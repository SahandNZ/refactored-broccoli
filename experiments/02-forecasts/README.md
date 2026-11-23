# 02 Forecasts: one shared forecast cache for rq1-rq4

Every model and baseline forecasts once per decision time, asset and horizon; studies 03-06 only
read this cache, so the GPU cost is paid once and every study scores the same numbers.

## Questions

**Q1** (RQ-forecasts). Does every model give a valid forecast for every decision of the
benchmark windows?
Decision rule: per window, yes if every model with decisions after its release in that window
has 9 finite, ordered quantiles at every labelled horizon for at least 99 % of those decisions;
the answer is yes when every window is yes, no otherwise (the answer lists the windows and the
models and windows below 99 %). Listed exception (owner, 2026-11-17): a baseline fit that does
not converge (GARCH) gives no forecast for that asset and window and is listed in
`not_fitted.csv`; those decisions leave the count and the answer names them.
Primary metric: share of labelled decisions after the model's release with a valid forecast,
per model and window.

## Hypothesis

- H1: every zero-shot model and baseline covers at least 99 % of its labelled decisions after
  its release, in every window.

## Design

**Decisions.** The universe of 01-universe (run `universe_run` in `study.toml`: its assets with
`in_universe`). Each window is a fold (structure.md#folds) and every decision is labelled with its
fold name. `folds.use = "development"` builds d1-d4; `"test"` (locked plan only) builds every
window: d1-d4, t1-t4 and l1-l12 (the leakage period ends where d1 starts, so no decision is
in two windows; a decision whose label would end after its window's end is dropped, so no
label reaches into the next window). Decision times are the window's hourly grid from its
start (`step_hours`: development and test every hour, leakage every 4 hours); Kronos (sampled,
slow) has a grid every `kronos_every_hours` = 167 hours from each window start, and at each
grid point every asset takes its first labelled decision at or after it (before the next
point), whatever the window's step (`on_model_grid`); 167 hours is one week less one hour,
so a quarter's ~13 grid points cover about 14 hours of the day, the same hours in every
window (the grid restarts at 00:00 UTC in each).

**Contexts.** The context of `(t, asset)` is the asset's last 512 own 1h closes that closed by
`t` (`src.benchmark._contexts.context_ends`); it is valid when the asset has 512 closes by
`t`. Closed hours have no bars and are never filled. No age limit is needed: with one
decision per own close (below), the newest close of a context is at most one decision step
old.

**Labels** (`src.benchmark._build.decision_labels`), in the asset's own bars, the steps every
model forecasts. With `end` the asset's last close at or before `t`, for horizons `h` = 1, 4,
24 own bars: `label = log(P[end+h] / P[end]) / (sigma * sqrt(h))`, `sigma` the EWMA (span 96
bars) standard deviation of the own-bar log returns at `end` (known at `t`,
`src.panel.ewma_volatility`), and the label time is the close time of bar `end+h`. A decision
is labelled only when its context is valid, bar `end+h` exists, the label is finite and a new
own bar closed since the previous decision time of the window's grid: one decision per `end`,
the first after each own close. For a market that trades around the clock this is every
decision and `h` bars are `h` hours; a market that closes has no decisions while it is closed
(they would only repeat the label and the forecast of its last close). The forecasts follow
the labels, so every study scores the same decisions, and no two decisions of an asset share
a label.

**One call, three horizons.** Each zero-shot model forecasts 24 steps of the asset's own
series once per context, and a row is written only for each horizon the decision has a label
for (a decision near the end of the data may lack the longer ones) (`chunk` contexts per call, `moirai_chunk` for Moirai, which runs in a
worker process); the price quantiles `q` of steps 1, 4 and 24 become label quantiles with the
label's own transform `log(q / P[end]) / (sigma * sqrt(h))`
(`src.benchmark._build.label_quantiles`); a quantile at or below zero becomes NaN or -inf and
makes the forecast invalid. Kronos reads real candles (OHLC, mid prices for bid/ask sources;
volume, Dukascopy bid volume; amount = volume * close), samples `kronos_paths` = 20 paths per
seed of `study.toml` (seeds 0, 1, 2), sampling with `kronos_temperature` = 1 and `kronos_top_p`
= 0.9 (the values of its reference code) and the tokenizer `kronos_tokenizer`; its future
timestamps continue the last candle every hour, the bar length (`src.models.KronosSettings`),
since the real times of the next own bars are unknown at the decision. Python, NumPy and torch
are seeded once per model and seed (`src.experiment.seed_everything`), then every window is
forecast in order; Kronos's paths of all seeds are pooled (3 x 20 = 60) before the quantiles
are taken, so every model writes one forecast per decision, with seed 0, and every other model
is deterministic.

**Weights.** Each model is built from `options.zero_shot` (kind and Hugging Face repository
for `src.models.build_forecaster`) at the commit pinned in `options.revisions` (and
`kronos_tokenizer_revision` for the tokenizer): `src.models.require_revision` checks before
loading that the repository's `main` is that commit and stops the build otherwise. The release
date of a model (`options.release_dates`) is the later of the creation of its repository and
the first upload of its weights; `options.weight_dates` is the upload date of the pinned
weights, the earliest commit whose tensors equal them (a later format conversion with
identical tensors keeps the earlier date: TimesFM 2.5's 2025-10-01 re-upload only fuses the
query, key and value matrices, TiRex-2's 2026-10-06 safetensors equal its 2026-07-02
checkpoint). A model is scored only after the later of the two, its scoring start (the
`release_date` of `models.csv`): Toto-2.0-313m from 2026-05-12 and TiRex-2 from 2026-07-02,
whose pinned weights replaced the first ones; checked on 2026-11-17.

**Baselines** (`src.benchmark._baselines`), fitted per window, per asset and per horizon, only
on the training labels of the `train_days` = 730 days before the window start whose label time
is at or before the start (`src.benchmark._build.training_labels`, computed once per window for
every baseline); `z` is the vector of the nine standard normal quantiles:

| Baseline | Label quantiles |
|---|---|
| random-walk | `z * s`, zero centre, `s = sqrt(mean(label^2))` of the training labels (their spread around zero) |
| climatology | the empirical quantiles (linear interpolation) of the training labels |
| ewma-normal | `z`: the label is already divided by the EWMA volatility; nothing is fitted |
| garch-normal | `z * sqrt(V_h) / (sigma * sqrt(h))`: GARCH(1,1), zero mean, normal errors (`arch`), fitted on the own-bar log returns of the training days (divided by their standard deviation) and filtered with fixed parameters through the window; `V_h` is the variance of the sum of the next `h` own-bar returns given the returns up to `end`; a fit that does not converge (arch convergence flag other than 0) gives that asset no forecast in that window: it is logged and added to `not_fitted.csv` of the run (model, window, asset, convergence flag) before the model's forecasts are written, never filled (a rerun after a crash may list a fit twice; readers drop duplicates) |
| linear | `x w + r`: ridge regression without intercept (`linear_ridge` = 1) of the label on the lags `x_l = log(P[end] / P[end-l]) / (sigma * sqrt(l))`, `l` in `linear_lags` = 1, 2, 4, 8, 24 own bars, plus `r`, the empirical quantiles of the in-sample training residuals |

**Costs** (`assets.csv`): crypto pays the Binance USD-M fee `crypto_fee` (taker, 5 bps per
side); every other asset half its median Dukascopy bid-ask spread over the 1h bars of
`spread_window` (2020-01-01 .. 2024-01-01, the selection window of 01-universe, before every
fold; `src.benchmark._build.median_spread_bps`); US stocks
from Alpaca have no bid and ask and get the assumed half spread `us_stock_half_spread_bps` =
1 bp.

**Windows.** `run.py --windows W ...` (fold names) forecasts only the labels of those windows;
the labels of every window are still written. 06-rq4-leakage uses it to build l1-l12 into a
leakage run next to a test cache that lacks them.
Each built model gets one row in the run's `windows.csv` (`model`, `windows`: the windows
its call forecast, in the labels' order d, t, l; a record for people, read by no code; a model
rebuilt after a crash gets a second row, and the last row per model counts). Kronos saves each seed's paths in
`samples/<model>/seed<s>.npz` with one array per window, named by the window, and reads them
by name (`src.benchmark.saved_paths`; a seed lacking a window, or holding another number of
contexts than its window, stops the pooling, and the error names the file to delete before the
build is run again); files saved
before commit 3790fbb (2026-11-17 23:05), i.e. every Kronos file of `dev-2026-11-17`,
`test-2026-11-17` and `leakage-2026-11-18`, hold unnamed arrays (`arr_0`, ...) and are read
in order only when they hold as many windows, each with its window's number of contexts.

**Resuming.** `run.py --run-id ID --models NAME ...` with the id of an existing run builds the
listed models into it with the run's own `config.toml` (`src.experiment.resume_run`), and skips
every model whose `forecasts/<model>.parquet` exists; without `--models` it builds every model
of `study.toml`. A resume is refused when the git commit, the dirty flag of the work tree, the
plan's hash or the data version differs from the run's start (its `meta.json`), so every model
of a run comes from the same code, plan and data; each resumed pass has its own time budget
(twice the estimate) and adds its models' rows to `metrics.parquet`. The first build of a run
writes the labels, `models.csv` and `assets.csv`; later builds reuse them. A smoke run
(`--last-days N`, development folds only) keeps only the decisions of the last N days of each
fold; `metrics.parquet` records N in `last_days`. Models built in separate runs of the same
commit and configuration (for example on rented GPUs) have the same labels, so their
`forecasts/<model>.parquet` files can be copied into one run.

### Cache contract (read by 03-rq1 .. 06-rq4)

A cache is one run of this study, `runs/<run-id>/`:

| File | Columns | Notes |
|---|---|---|
| `forecasts/<model>.parquet` | `time`, `asset`, `horizon`, `seed`, `q10` .. `q90` | one file per model of `study.toml`; quantiles of the vol-scaled label (float32); one row per labelled decision and horizon, `seed` 0 (Kronos: its seeds' paths pooled) |
| `labels.parquet` | `time`, `asset`, `horizon`, `window`, `label_time`, `label`, `log_return`, `sigma` | one row per labelled decision and horizon; only finite labels; `label_time` is the close time of the label's last bar (`end + h`) |
| `models.csv` | `model`, `kind`, `release_date`, `pretraining_cutoff`, `every_hours` | `kind` is `zero-shot` or `baseline`; dates UTC, empty for baselines; `release_date` is the scoring start (the later of the release and the pinned weights' upload); `every_hours` is the hours between a grid model's grid points (Kronos 167), 0 for the others |
| `assets.csv` | `asset`, `market`, `cost_per_side_bps` | costs from `src.evaluate.cost_per_side_bps` (Costs above) |

- `time` is the decision time (UTC, tz-aware), `horizon` is 1, 4 or 24 (own bars), `window` is
  the fold name (`d1` .. `d4`, `t1` .. `t4`, `l1` .. `l12`).
- A (time, asset, horizon) row of a model exists only where the model gave a forecast; rows
  join on (`time`, `asset`, `horizon`).
- Studies read the cache only through `src.benchmark.read_cache` (which refuses a label that is
  not finite) and score it through `src.benchmark._decisions.scored_decisions`: the same
  validity rule as Q1 below (a decision with an invalid forecast is left out and counted per
  cell as `invalid`), one forecast per decision.

## Data

The universe of 01-universe (72 assets), 1h bars from `data/` (US stocks: regular session only), windows
d1-d4, t1-t4 and l1-l12 (structure.md#folds); the baselines are fitted on the 730 days before
each window (`train_days`).

## Metrics

Share of labelled decisions with a valid forecast per model and window
(the study's `lib.validity_table`, `tables/validity.csv`): a decision `(time, asset)` on the
model's grid after its release is valid when, at every labelled horizon, its nine quantiles
are finite and in increasing order (equal neighbours allowed;
`src.benchmark._decisions.valid_quantiles`, the rule every scoring study applies). Forecast rows,
time and peak GPU memory per model in `metrics.parquet` and the run log.

## Decision rule

Every choice is made on the development folds; the test folds run once, after every choice is
fixed ([the README](../../README.md#folds)). This study makes no choice: it
only computes forecasts. Each question's own rule is under [Questions](#questions).

## Compute

The GPU hours of the cache are estimated, and later measured, in one place: the paper README,
[Compute of the shared forecasts](../README.md#compute-of-the-shared-forecasts). A `--models`
option builds one model at a time into the same run, so the cache can grow model by model.

## Status

Plan written on 2026-11-16; the cache builder (`src.benchmark.build_cache`), `run.py` and
`analyze.py` (Q1, `q1_min_share` = 0.99) are written and tested on synthetic data only. Not run
on real data: waiting for the owner's approval.

Changed on 2026-11-16 after an independent code review, before any run: labels and every
forecast step are counted in the asset's own bars (with `label_time`), Kronos decides every
167 hours, a resumed run uses its own `config.toml` and needs the same commit and plan; one
decision per own close (no repeated closed-hour decisions); Kronos's future timestamps one
hour apart.

Changed on 2026-11-17 after a third review, before any full run (owner's decisions): Q1 counts
only decisions after each model's release and answers per window; the seed is set once per
model and seed, not per window; Kronos pools its 60 paths before the quantiles; no row for a
horizon without a label; pinned revisions, release and weight dates and the scoring start;
costs from the median spread of the selection window (it was the median over all data,
2020-2026, which includes the folds); `--last-days` for a smoke run.

Kronos in the development cache `dev-2026-11-17` (2026-11-17): Kronos-small and Kronos-base
were sampled on vast.ai in chunks, not in one pass per seed, to finish within the day
(scratch driver and chunk script in `.tmp/rq1/kronos2/`, code at commit 70fa6c1). A chunk is
one window of one seed (Kronos-small) or one contiguous half of one window of one seed
(Kronos-base, cut at row round(n/2) of the window's contexts in the cache's order); each chunk
restarts `seed_everything(seed)` at its start instead of continuing the seed's random stream
across windows. The decisions, contexts, seeds, 20 paths per seed and the pooling (3 x 20 = 60
paths) are those of an unsplit run: the chunks tile every window's contexts exactly (1004, 936,
1008 and 1008 per window for d1-d4, checked before pooling, no gap and no overlap), and the
pooled quantiles are written by the builder itself (`build_cache` on the merged
`samples/<model>/seed<s>.npz`). Only the random draws differ from an unsplit run (the first
chunk of each seed, window d1 or its first half, is identical); the test folds can use either
scheme, recorded with their run.

The test cache `test-2026-11-17` (2026-11-17, plan locked; RQ1's test run) is built for d1-d4
and t1-t4 only; the leakage windows l1-l12 are not built for most models (left for RQ4):
- `run.py` with `folds.use = "test"` builds every window, but GARCH(1,1) does not converge for
  NATGAS on l3 (training before 2022-07-01; arch exit code 8, also with 5,000 iterations), so
  the full build stopped after random-walk, climatology and ewma-normal (these three have all
  windows). The other local models were built by the builder with its window list cut to d and
  t (scratch `.tmp/rq1/test/build_dt.py`, `cache_windows` patched).
- timesfm-2.5, timesfm-3, tirex-1.1 and tirex-2 ran on vast.ai (`run.py` with the locked
  config; their own runs `test-2026-11-17-<model>` hold every window) and their
  `forecasts/<model>.parquet` was copied in after checking that their labels have the same keys
  and the same values within 1e-9 (float rounding of another CPU). TiRex needs a C compiler
  (torch.compile), which the vast image lacks: the jobs install gcc first.
- Kronos: d1-d4 reuse the development paths (the test run's d1-d4 labels equal the development
  run's exactly, so the contexts are the same); t1-t4 were sampled on vast.ai in units of one
  window (Kronos-small) or one window or its two halves (Kronos-base), each restarting
  `seed_everything(seed)` as for the development cache. The raw paths are kept unpooled, one
  file per model, seed and window: `.tmp/rq1/kronos_samples/<model>/seed<s>-<window>.npy`
  (shape (contexts, 20, 24), contexts in the builder's order). The forecasts are pooled by
  `.tmp/rq1/pool_kronos_windows.py <run-id> <model> <windows>`, which runs the builder with its
  window list cut to the given windows.
- l1-l12 for RQ4 are not added to this run: they are in a second run of the same
  configuration and data, `runs/leakage-2026-11-18`, built with `run.py --windows l1 .. l12`
  for the models whose l1-l12 this run lacks (chronos-bolt-small and -base, chronos-2,
  moirai-2-small, t0-gamma, toto-2-22m and -313m, garch-normal, linear). Kronos-small and
  Kronos-base were sampled on l1-l12 on vast.ai by the RQ4 thread and pooled into that run
  (state of 2026-11-17, commit 2d7432c; the RQ4 thread's scratch folder
  `a scratch folder`, outside the lab and outside git, holds the drivers and the
  raw paths, so these forecasts cannot be rebuilt from the repository alone). 06-rq4-leakage reads
  the two runs as one (`options.leakage_cache`); a window must come from one run only. The
  GARCH(1,1) fits that did not converge are left out and listed in the run's `not_fitted.csv`
  (owner, 2026-11-17): NATGAS in l3 and NFLX in l4 (arch convergence flag 8).

GPU numerics (2026-11-17): timesfm-3, tirex-1.1 and tirex-2 of `test-2026-11-17` were built on
RTX 4090s (vast.ai), the development cache on the local RTX 3500 Ada, with the same code, lock
file and package versions (torch 2.15.0, timesfm 3.0.2); these three models are reproducible
only up to GPU kernel numerics (TimesFM 2.5 runs with `float32_matmul_precision("high")`, i.e.
TF32, and TiRex compiles its blocks with `torch.compile`, whose kernels depend on the GPU). On
d1-d4
their quantiles differ from the development cache by a median of 0.0001-0.0027 label units (max
0.69, tirex-1.1); the other 15 models match exactly (timesfm-2.5 too, built on a 4090). Effect
on RQ1: 03 on the test folds uses d1-d4 only for the best-baseline choice (baselines, built
locally, identical) and scores t1-t4, so these d rows enter no test number; rerunning 03 on the
development cache with the 4090-built d rows swapped in changes tirex-1.1's skill by at most
0.005 percentage points and its p-values by at most 0.0004, and no status, quarter count, MDE,
family or verdict (timesfm-3 and tirex-2 have no scored development cell).
The TimesFM 2.5 wrapper set TF32 for the whole process (`float32_matmul_precision("high")`),
so a model built after it in the same process also ran in TF32. In `dev-2026-11-17` the only
such model is timesfm-3 (built after timesfm-2.5 in one process): its wrapper
(`src.models._new_models.TimesFM3Forecaster`) and the `timesfm3` package set no precision, so
dev timesfm-3 ran in TF32 only through that leak, while test timesfm-3 (its own vast.ai
process) ran in full float32. That is a possible second cause of timesfm-3's dev-test gap
on d1-d4, next to the GPU (not measured apart from it). In `test-2026-11-17` no model ran after TimesFM in one process (the local
models were built in processes without it). TiRex-2's package can also set the precision
process-wide (`matmul_precision`), which no wrapper here passes. Since 2026-11-17 the TimesFM
2.5 wrapper sets TF32 only inside its own load and predict calls
(`src.models._timesfm.TF32Precision`), which leaves its own numerics as before.
