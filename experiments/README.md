# Experiments

One folder per study, each named `NN-name` (NN is the order in the paper):

```
experiments/
  01-universe/ .. 13-context-baselines/   the studies of the benchmark paper (table below)
  00-template/    copy 00-template/01-rq1-smoke/ to start a study
```

Every study has one shape: a plan `README.md` (Questions, Hypothesis, Design, Data, Metrics,
Decision rule, Compute, Status), `study.toml`, `run.py`, `analyze.py`, optional `make_tab_*.py`
and `lib/`, checked by `tests/experiment/test_shape.py`.
Outputs go to `<study>/runs/<run-id>/` (every output of a run, not in git) and the final answers
in `<study>/results/` (the README and the files it cites, in git).

**Layout change (2026-11-18).** The studies moved from `experiments/01-benchmark/NN-name/` to
`experiments/NN-name/`, and this file took the content of `experiments/01-benchmark/README.md`.
Records written before that date (locked study plans, committed results, run `meta.json`) keep
the old path `experiments/01-benchmark/...`; read it as `experiments/...`.

**Commit ids (2026-11-18).** The history was rewritten that day (commit messages only; every
file is unchanged), so every commit id changed. Results READMEs and the paper cite the new ids.
The locked study plans and each run's `meta.json` (`git_commit`, `plan_commit`) keep the old
ids; `docs/history-sha-map.tsv` maps each old id to the new one (lab history only; the
anonymous release has no commit history and no map).

## The multi-market benchmark

Zero-shot time-series foundation models on five markets (crypto, forex, commodities, indices,
stocks), intraday, strictly after each model's release, with each market's costs. Each study below answers some of the research
questions of this paper; each keeps its results in its own `results/` folder.

### Studies

One study per question. Studies that answer a main research question of the paper are named
`NN-rqN-name`; supporting ones `NN-name` (NN is the order in the paper).

Status checked on 2026-11-19 against each study's `results/README.md` (its run table says
`completed` and it answers every question of the plan) and its committed `results/<run-id>/`
tables. Completed primary: the locked run whose answers the paper reports; completed
supporting: a run that fixes a choice or input of the primary studies (on the development folds
where it chooses); completed exploratory or descriptive: planned and locked after the test
results, reported as such. The `## Status` section of a locked plan is the record at its lock
and is not updated (the plan hash covers it); this table is the current status.

| Study | Answers | Status | Evidence (runs cited by `results/README.md`) |
|---|---|---|---|
| [01-universe](01-universe/README.md) | RQ-universe | completed, supporting (Q1 yes, Q3 yes) | `2026-11-17T0005-random-1000`; earlier runs kept in `results/` |
| [02-forecasts](02-forecasts/README.md) | RQ-forecasts | completed, supporting: development, test and leakage caches built (Q1 yes on the development cache) | `dev-2026-11-17`; the test cache `test-2026-11-17` is read by 01 and 03-08, 10, 12 and 13, the leakage cache `leakage-2026-11-18` by 06 (their `study.toml`); caches are not in git |
| [03-rq1-null](03-rq1-null/README.md) | RQ-null | completed, primary (Q1 no, Q2 no) | `test-2026-11-17`; development runs kept in `results/` |
| [04-rq2-center-vs-scale](04-rq2-center-vs-scale/README.md) | RQ-center-vs-scale | completed, primary (Q1 no, Q2 yes) | `test-2026-11-17` |
| [05-rq3-net](05-rq3-net/README.md) | RQ-net | completed, primary (Q1 no) | `test-2026-11-17` |
| [06-rq4-leakage](06-rq4-leakage/README.md) | RQ-leakage | completed, primary (Q1 no) | `test-2026-11-18` |
| [08-closures](08-closures/README.md) | RQ5 main answer (RQ-closures) | completed, primary, RQ5 case study (Q1 no, Q2 no) | `test-2026-11-17`; development run kept in `results/` |
| [07-rq5-off-hours](07-rq5-off-hours/README.md) | RQ5 secondary answer (RQ-off-hours) | completed, primary, RQ5 case study (Q1 no, Q2 yes) | `test-2026-11-17`; development run kept in `results/` |
| [09-mde-threshold](09-mde-threshold/README.md) | RQ-power | completed, supporting: chooses rq1's threshold on the development folds (Q1 yes) | `dev-2026-11-17` |
| [10-rq1-checks](10-rq1-checks/README.md) | RQ-null (exploratory checks, after the test results) | completed, exploratory (Q1 no) | `test-2026-11-18-v2` |
| [11-closures-length](11-closures-length/README.md) | RQ-closures (exploratory, after the test results) | completed, exploratory (Q1 yes) | `test-2026-11-18-v2` |
| [12-horizon-spans](12-horizon-spans/README.md) | RQ-forecasts (descriptive: the horizons' calendar spans) | completed, descriptive (Q1 yes) | `test-2026-11-18` |
| [13-context-baselines](13-context-baselines/README.md) | RQ-null (exploratory: baselines fitted on the 512-bar context) | completed, exploratory (Q1 no) | `test-2026-11-18-v2` |

Planned, not run: RQ-cross-market, RQ-timeframes and RQ-scaling (below) have no study, and
01-universe's former Q2 (the number of assets) moved to later work; nothing in `results/`
answers them.

The zero-shot plan that used to be one study (`02-zero-shot`, before that `02-benchmark`) was
split into rq1-rq4 on 2026-11-16 (owner's rule: one study per question); its shared design is
below.

### Shared design

rq1-rq4 read one shared forecast cache: every model and baseline forecasts once per decision
time, asset and horizon, and each study runs only its own analysis on that cache, so the GPU
cost is paid once. The cache lives in `experiments/02-forecasts/runs/<id>/` (not in git) with the
same metadata as a run.

Every model forecasts the close 1, 4 and 24 of the asset's own 1h bars ahead from its last
512 own 1h closes (horizons are counted in bars, not hours; for a stock 24 bars are about 3.4
trading days) (a market that closes has no bars while closed, so `h` bars can span more
than `h` hours); the label is the vol-scaled log return over the same `h` own bars, from the
last close at or before the decision time, and its label time is the close of the last of
those bars. A decision time is kept only when a new own bar closed since the previous one
(one decision per own close): crypto keeps every hour, a market that closes has no decisions
while it is closed. The price quantiles become quantiles of that label with its own transform, so every
model and baseline is scored on the same target (02-forecasts).

| Model | Kind | Decisions |
|---|---|---|
| 13 zero-shot checkpoints: Chronos-Bolt small and base, Chronos-2, TimesFM 2.5 (200M) and 3.0, TiRex 1.1 and TiRex-2, Toto 2.0 22m and 313m, t0-gamma, Moirai 2.0 small, Kronos-small and Kronos-base | zero-shot, each scored only after its own scoring start (below) | every own close; Kronos one per asset at each 167 h grid point (its first decision at or after it) |
| Kronos (real OHLCV candles, 20 sampled paths per seed, seeds 0-2 pooled to 60 paths; owner, 2026-11-16) | zero-shot | as above |
| Random walk: zero centre, spread of the asset's past labels | baseline | every own close |
| Climatology: quantiles of the asset's training labels | baseline | every own close |
| EWMA-normal: standard normal in label units | baseline | every own close |
| GARCH(1,1)-normal on own-bar returns, fitted on the 730 days before each window | baseline (scale) | every own close |
| Linear: ridge on the vol-scaled returns of the last 1, 2, 4, 8 and 24 own bars | baseline (centre) | every own close |

- Zero-shot models have no training; each fold's training period only fits the baselines.
- Deterministic models run once; Kronos samples 20 paths with each of seeds 0, 1 and 2, and
  the 60 paths are pooled before the quantiles are taken, so every study scores one forecast
  per decision.
- Weights are pinned: every checkpoint and Kronos's tokenizer is loaded at the commit recorded
  in 02-forecasts' `study.toml` (`options.revisions`), checked before loading.
- Every study scores the same decisions of a model: only valid forecasts (nine finite, ordered
  quantiles; the invalid ones are counted per cell), a zero-shot model only after its scoring
  start. Every (model, market, horizon) cell is reported; a cell without decisions, or with
  fewer decision points than its bootstrap block, is reported as not scored with the reason,
  and Holm's correction counts only the scored cells. A study in which no cell of any
  zero-shot model can be scored stops with an error that lists the reasons.
- Tests on a series of decision times count how its labels overlap (`cell_timing`): the
  overlap of a cell is the median number of later decision times of the cell inside a
  decision's label, counted directly, so a market that pools sessions of several exchanges is
  counted right (hourly crypto at 24 bars: 23; Kronos: 0). rq1's Diebold-Mariano test takes
  Newey-West lags max(overlap, Andrews' automatic lags), a t(n - 1) reference and a one-sided
  p-value (H1: the model is better), and its bootstrap block is the Politis-White block of the
  cell's loss difference with a floor of four times the overlap (03-rq1-null). Kronos's decisions
  of one 167 h grid point (the assets of a market that closes decide at different times
  after it) are one decision point.
- The best baseline of a cell is chosen on the development folds: for the test folds on
  d1-d4, while the plan is open each development fold on the other three.
- Trading (Q4): at each horizon's decision times the book holds the sign of each asset's median
  forecast, equal weight, for `h` of the asset's own bars (an asset opens a new position only
  after its last one ended). Costs per side: the Binance USD-M taker fee (5 bps, regular account) for crypto, half
  the median Dukascopy spread of the selection window 2020-01-01 .. 2024-01-01 (before every
  fold) for forex, commodities, indices and non-US stocks, and for US
  stocks (Alpaca, no bid/ask) an assumed half spread of 1 bp (a typical 2 bps quoted spread of
  large US stocks), reported as an assumption. The break-even cost is computed from gross
  returns and turnover.
- Chronos-2 can also take several series at once (group attention). That multivariate use is
  not part of this zero-shot benchmark; it is a possible baseline of later work.

#### Release dates and the post-release window

The benchmark has 13 checkpoints (the must-have list of the model catalogue, 2026-11-16).
A model's release date is the later of the creation of its Hugging Face repository and the
first upload of its weights; its scoring start is the later of that and the upload date of
the exact weights loaded (the pinned revision; a later format conversion with identical
tensors keeps the earlier date). Every model is scored only after its own scoring start, so
the models released during the test period are scored on part of it; rq1 puts the models with
fewer than 3 of the 4 test quarters wholly after their scoring start into a separate Holm
family. Checked on 2026-11-17 (02-forecasts `study.toml`: `options.revisions`,
`release_dates`, `weight_dates`):

| Model | Released | Weights loaded | Scoring start | Pinned revision |
|---|---|---|---|---|
| Chronos-Bolt small / base | 2024-11-25 | 2024-11-25 | 2024-11-25 | 772f3d2 / 5d9f166 |
| TiRex 1.1 | 2025-05-26 | 2025-05-26 | 2025-05-26 | 43ebaaf |
| Kronos small / base (tokenizer 0e01173) | 2025-06-30 | 2025-06-30 | 2025-06-30 | 901c26c / 2b55474 |
| Moirai 2.0 small | 2025-08-06 | 2025-08-06 | 2025-08-06 | 30f43ff |
| TimesFM 2.5 (200M) | 2025-09-02 | 2025-09-02 (the 2025-10-01 upload fuses q, k, v; identical tensors) | 2025-09-02 | 1d95242 |
| Chronos-2 | 2025-10-30 | 2025-10-30 | 2025-10-30 | 29ec376 |
| Toto 2.0 22m | 2026-04-14 | 2026-04-14 | 2026-04-14 | 685e4ae |
| Toto 2.0 313m | 2026-04-14 | 2026-05-12 (new weights) | 2026-05-12 | a7bab28 |
| TiRex-2 | 2026-06-16 | 2026-07-02 (the 2026-10-06 safetensors equal the 2026-07-02 checkpoint) | 2026-07-02 | 3526361 |
| TimesFM 3.0 | 2026-08-24 | 2026-08-24 | 2026-08-24 | 43046b8 |
| t0-gamma | 2026-10-15 (repository 2026-10-12) | 2026-10-15 | 2026-10-15 | fef67ab |

Financial market series in pretraining (paper or card): Chronos-Bolt none named (the Chronos
training set has no finance data; arXiv 2403.07815 Table 3); Kronos only market data (12.1B
K-line bars, about 96,600 assets, up to 2024-06; arXiv 2508.02739, Appendix D); TimesFM 2.5
GIFT-Eval Pretrain (its Econ/Fin part has FRED-MD and Bitcoin), Wikimedia to 2023-11, Google
Trends to 2022, synthetic; Chronos-2 none (Table 6 of arXiv 2510.15821).

#### Development, test and leakage periods

- **Development folds d1-d4** (2024-11 .. 2025-10) serve only the benchmark's own choices
  (the MDE threshold of rq1, the trading rule); the context length (512) and Kronos's paths
  (20 per seed) are fixed, not chosen. They overlap some releases, so no headline number
  comes from them.
- **Then the plan is locked** (`[plan] locked = true`) and the **test folds t1-t4** run once
  for Q1-Q4.
- **Leakage (Q5):** the same locked run covers the quarters l1-l12 (2022-01 .. 2024-10; l12 is
  the one month 2024-10, so the leakage period ends where d1 starts). For
  each model, "before" is its decisions in l1-l12 whose label time is before its pretraining
  cutoff and its release, and "after" is its decisions in t1-t4 after its release; the control
  is the best baseline (06-rq4-leakage).

#### Compute of the shared forecasts

The one place for the GPU hours of the shared forecast cache (02-forecasts); measured numbers
replace the estimates as each part is built.

- Measured on 2026-11-17 (smoke run, last 2 days of each development fold, local RTX 3500):
  the 5 baselines and Chronos-Bolt small take 27 s together for about 6,300 contexts.
  Kronos is the cost: it samples 60 paths of 24 steps per decision, autoregressively.
- Estimate for d1-d4 (about 290,000 contexts per hourly model; Kronos about 3,700 decisions):
  the hourly models take minutes to about an hour each on one GPU; Kronos-small several GPU
  hours and Kronos-base about four times that, so Kronos runs in seed shards (`--seeds`) on
  rented GPUs. The test folds and the leakage quarters (t1-t4, l1-l12) are about four times
  the development folds.
- `--dry-run` first; the runner stops a run at twice its estimate.

### Research questions

Every study question names the one it serves, so each result maps to the paper.

Main questions (the benchmark):

| Id | Research question |
|---|---|
| RQ-null | Does the zero-shot null result hold across five markets (crypto, forex, commodities, indices, stocks), intraday, strictly after every model's release, against the best of five simple baselines (random walk, climatology, EWMA-normal, GARCH-normal, linear)? |
| RQ-center-vs-scale | Is any skill in the centre of the forecast (sign, mean) or in its scale (volatility, calibration), per market and horizon? |
| RQ-net | What is the net result after each market's costs, and the break-even cost per market and horizon? |
| RQ-leakage | Does each model do better before its own release than after it, beyond the best baseline's change between the same periods (difference in differences)? |

Analysis questions:

| Id | Research question |
|---|---|
| RQ-cross-market | Which markets inform the forecasts of which others? |
| RQ-off-hours | When a market is closed, does weekend crypto fed to a zero-shot model as a covariate improve its forecast of the reopening (secondary answer to RQ5, with RQ-closures)? |

Supporting and appendix questions:

| Id | Research question |
|---|---|
| RQ-universe | Which assets, and how many, should the model see, and why? |
| RQ-forecasts | Does every model give a valid forecast for every decision of the benchmark windows (the shared forecast cache)? |
| RQ-closures | How should closed market hours be shown to a foundation model? |
| RQ-power | Which minimum detectable effect can the benchmark's tests resolve, and which threshold should rq1 use? |
| RQ-timeframes | Which input timeframes and which decision frequency, and why? |
| RQ-scaling | Does forecasting improve with more assets, and do models transfer to unseen assets? (future work) |
