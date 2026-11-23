# Results: 07-rq5-off-hours

Plan: [../README.md](../README.md). Runs: test-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-17 | 2026-11-17 20:23 / 21:23 | 131.0 s | completed | local | NVIDIA GeForce RTX 4090 | 17.91 / 3.32 GB | - | None |

## Q1 (RQ-off-hours)

Does feeding the weekend crypto move to a zero-shot model as a covariate improve its forecast of the first bar after a weekend reopening?

Primary metric: CRPS of the first bar after the reopening, in the benchmark's label units.

**Q1**: no. the covariate arm beats all four other arms in 0 of 9 tested market and model cells: none. One-sided tests (adjusted p below 0.05): covariate vs calendar: 3 of 9 cells, median skill -0.9 % (cell intervals from -61.5 to +22.4 %); covariate vs event: 0 of 9 cells, median skill -48.7 % (cell intervals from -399.4 to -19.9 %); covariate vs correction: 0 of 9 cells, median skill -64.0 % (cell intervals from -405.8 to -22.8 %); covariate vs model-free: 0 of 9 cells, median skill -62.8 % (cell intervals from -416.8 to -22.7 %); correction vs event: 3 of 9 cells, median skill +7.2 % (cell intervals from -2.8 to +14.4 %); model-free vs event: 3 of 9 cells, median skill +6.6 % (cell intervals from -3.4 to +14.0 %). Median share of the move's crypto slope a model passes on from its covariate (descriptive): chronos-2 -0.10; t0-gamma 0.89; tirex-2 0.09. 0 cells not tested.

Limitations: weekends from 2024-01-01, scored in the t1-t4 folds, about one weekly point per weekend and cell (low power); no cut at a model's release: some or all scored weekends precede a model's release (on d1-d4 all of them, for every model; on t1-t4 none for Chronos-2, until 2026-07-02 for TiRex-2 and almost all for t0, released 2026-10-15), so a model that memorised a reopening could favour its own arms over the correction and the model-free arm; one forecast per weekend and model; the label is the close of the first bar, not the gap of Q2; the correction and the model-free arm are refitted before every weekend on all earlier weekends since 2024-01-01 whose label was known (at least 26), also inside the scored quarter (past labels only); the covariate use is descriptive (no interval)

Files: [test-2026-11-17/tables/reopening_cells.csv](test-2026-11-17/tables/reopening_cells.csv), [test-2026-11-17/tables/reopening_tests.csv](test-2026-11-17/tables/reopening_tests.csv), [test-2026-11-17/tables/covariate_use.csv](test-2026-11-17/tables/covariate_use.csv)

## Q2 (RQ-off-hours)

Does the weekend crypto move explain the reopening gap of closed-market assets beyond Friday's information?

Primary metric: the crypto coefficient's two-sided Holm-adjusted p-value per asset (its
sign, HAC interval, rank correlation and R^2 increment are reported next to it).

**Q2**: yes. 10 of 85 tested assets: COPPER beta 0.03 [0.01, 0.04] (positive, adjusted p 0.01), trimmed rank corr 0.18, R^2 +0.07; USDILS beta -0.02 [-0.02, -0.01] (negative, adjusted p 0.00056), trimmed rank corr -0.21, R^2 +0.05; AUS200 beta 0.03 [0.02, 0.04] (positive, adjusted p 0.0065), trimmed rank corr 0.26, R^2 +0.13; CHE20 beta 0.04 [0.03, 0.06] (positive, adjusted p 6.5e-05), trimmed rank corr 0.30, R^2 +0.17; ESP35 beta 0.04 [0.03, 0.06] (positive, adjusted p 0.00016), trimmed rank corr 0.33, R^2 +0.14; EUS50 beta 0.05 [0.03, 0.08] (positive, adjusted p 0.00041), trimmed rank corr 0.30, R^2 +0.15; FRA40 beta 0.05 [0.03, 0.07] (positive, adjusted p 4.2e-05), trimmed rank corr 0.29, R^2 +0.16; NLD25 beta 0.05 [0.03, 0.07] (positive, adjusted p 0.0013), trimmed rank corr 0.30, R^2 +0.18; PLN20 beta 0.06 [0.04, 0.09] (positive, adjusted p 0.00019), trimmed rank corr 0.36, R^2 +0.17; USA500 beta 0.04 [0.02, 0.06] (positive, adjusted p 0.013), trimmed rank corr 0.33, R^2 +0.15. Median rank correlation 0.07; on crypto-down weekends 0.04, on crypto-up weekends 0.07; positive in 578 of 930 tested asset half-years. 0 assets not tested.

Limitations: weekends 2020-01-01 .. 2026-11-01, in effect from the listing of the last basket asset on Binance USD-M, before which the basket move is missing (weekends counted as incomplete); assets that passed 01-universe's data checks of 2020-2023, but from candidate lists of 2026; Friday's information is only the last 6 hours of the asset and the basket; two-sided HAC t, Holm over 85 assets, a little liberal with 4 Bartlett lags (about 7 % at 5 % with 4 lags on 100 synthetic null weekends); 95% HAC intervals, not Holm-adjusted; the gap is a mid quote of a CFD, whose spread is wide at the reopening; an in-sample regression over all weekends up to the end of the scored folds, so a test run includes the development months

Files: [test-2026-11-17/tables/gap_assets.csv](test-2026-11-17/tables/gap_assets.csv), [test-2026-11-17/tables/gap_half_years.csv](test-2026-11-17/tables/gap_half_years.csv), [test-2026-11-17/tables/gap_directions.csv](test-2026-11-17/tables/gap_directions.csv)

## Provenance (added after review)

Both runs ran on rented RTX 4090s (vast.ai) on 2026-11-17 without a `.git` folder, so their
`meta.json` records no commit ("None" above) and `location = "local"`. The code was the lab
tree at commit bfa9b10, inferred, not recorded: the copy to the instance excluded `.git`
and recorded no file hashes. The evidence: both runs' `plan_hash` is that of the plan README
at bfa9b10, their `config.toml` equal that commit's `study.toml` (test run, plan locked, test
folds) and the `study.toml` of commit 36ba4fe (development run, development folds), the data
version matches, the only commit in between touched 08 only, and an independent review
regenerated `weekends.csv` exactly from that code and the local data. Each run folder holds a
`provenance.md` with these facts. The development run's answers (the same
code and plan on d1-d4) are kept in [dev-2026-11-17/README.md](dev-2026-11-17/README.md) with
its tables.

## Reading the answers (added after review)

An independent review on 2026-11-17 recomputed every number of both runs from
`forecasts.parquet` and `weekends.csv` (CRPS per cell and arm, skills, intervals, the
Diebold-Mariano tests, Holm, the Q2 regressions) and found them exact; it found no look-ahead
in the correction and model-free fits and no implementation error in the contexts. What the
numbers mean:

- **Why the covariate arm loses.** The covariate is given only in calendar time, whose context
  ends in about 49 missing closed hours, so every model forecasts as if about 50 hours ahead:
  its 10-90 % interval is 5 to 12 times wider than the event forecast's, with a coverage of
  84-98 %, and its median drifts by 1-2.5 label units. Covariate against event time, the
  correction and the model-free forecast therefore mostly measures calendar time against
  event time; only covariate against calendar isolates the covariate. 08-closures' "missing"
  arm shows the same loss on its development run. Q1's "no" holds under its rule, but it does
  not show that a zero-shot model cannot use the weekend crypto move: giving the move in
  event time, as a known covariate at the forecast step (known at `reopen_time`), is not
  tested.
- **The covariate-against-calendar wins are Chronos-2's only** (3 of 9 cells on both runs), and
  Chronos-2 does not pass on the crypto direction (`use_share` about 0 or negative): its gain
  comes from a less drifting centre of the forecast, not from following crypto.
- **The event forecast is too narrow at a reopening.** It is worse than the naive
  `ewma-normal` in every cell, with a 10-90 % coverage of 40-63 %, so "correction and
  model-free beat event time" largely repairs that width; the model-free forecast barely
  beats `ewma-normal` on forex.
- **`use_share` is not meaningful for forex**, whose move has no crypto slope (`move_slope`
  about 0): its forex values (e.g. -29.8 for t0 on the test run) enter the per-model medians
  above; read the commodity and index cells.
