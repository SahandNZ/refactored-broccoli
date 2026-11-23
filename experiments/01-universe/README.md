# 01 Universe: which assets, and how many

## Questions

**Q1** (RQ-universe). Does the rule-based universe cover more independent market behaviour
than simple alternatives of the same size?
Decision rule: yes if, on the held-out window 2024-01-01 to 2024-10-31, its effective rank is
at or above the 90th percentile of 1000 random universes with the same number of assets per pool
and above the Liquid universe, all measured on that window; no otherwise. The same numbers on
the selection window, where the universe is picked, are reported as in-sample.
Primary metric: effective rank of the weekly-return correlation matrix on the held-out window.

**Q3** (RQ-universe). Does the ranking of the models by mean CRPS stay the same across parts of the universe?
Decision rule: yes if the median Kendall tau between the rankings on the two halves of 1000 random splits of the universe (half of every pool in each half), on the hourly models of the test period, is at least 0.7; no otherwise.
Primary metric: median Kendall tau of the split halves' rankings by mean CRPS.

## Hypothesis

- H1: the rule-based universe has a higher effective rank and a lower mean absolute
  correlation than the Liquid universe and than the median of 1000 random universes, also on
  the held-out window it was not picked on.
- H3: the ranking of models is stable across parts of the universe (random halves, markets) and across test quarters.

## Design

### The universe

The selection produces one universe (rule fix 2026-11-16: T2 and T3 are gone and the
remaining T1 is called "the universe"). The stocks are two pools picked separately: US
stocks and non-US stocks (all non-US exchanges together).

| Pool | crypto | forex | commodities | indices | US stocks | non-US stocks | total |
|---|---|---|---|---|---|---|---|
| Quota | 8 | 16 | 8 | 8 | 16 | 16 | up to 72 |

- A pool with fewer candidates than its quota takes all it can (for example the commodities
  when fewer than 8 pass, a pool whose candidates are dropped as near-duplicates, or a stock
  pool whose sectors run out under the sector cap); the real per-pool counts are recorded in
  `universe.csv`, in the `size` column of `diversity.csv` and in the run log.
- Membership is the `in_universe` column (true or false) of `runs/<run-id>/universe.csv`,
  with the pick order in `order`; it is not part of `data/`.
- The quotas are `[options.quotas]` of `study.toml`, one number per pool.
- No survival condition: an asset is never required to trade until the end of the test
  period (that would be survivorship bias), and no data after the selection window decides
  anything. An asset that stops trading (for example a delisted coin) is scored up to its last
  bar and reported; `universe.csv` records each candidate's `last_bar` from the manifest.

### Selection protocol

Fixed before any test period. Every filter, score and correlation of the pick uses only the
selection window, 2020-01-01 to 2023-12-31 (`start`, `end`), so the selection never sees the
development folds (d1 starts 2024-11-01). The candidates are the instruments of
`data/manifest.json`; its `passed` and `reason` (computed over 2020-2026) are not used.

1. **Data filter.** A forex pair that contains a pegged currency (`pegged_currencies`: HKD
   and DKK; CNH stays) is dropped. Then each candidate needs bars in the window and a first
   bar by its start date: `start` plus `start_tolerance_days` (7; most markets first opened on
   2 January 2020, some after 00:00 UTC), or for crypto and commodities the end of the day in
   `[options.listed_by]` (crypto 2020-12-31, commodities 2021-12-31, so platinum and palladium
   are candidates; no tolerance is added). Each is masked before its first bar.
2. **Data quality in the window.** The lab's standard quality check (`src.data.check_quality`,
   the limits of `[quality]` in `configs/data.toml`) runs on each candidate's own base bars
   (15m) cut to the selection window: coverage of the expected days at least `min_coverage`
   (0.95), no gap between two bars longer than `max_gap_days` (10 days), almost no ask below
   bid, no one-bar jump above `max_abs_return` outside crypto, and for stocks with a Yahoo
   reference the daily close-to-close log returns against Yahoo's returns of the same window
   on their common days (median and 99th percentile of the absolute difference at most
   `yahoo_return_median_bps` (50) and `yahoo_return_p99_bps` (250); returns, not price levels,
   so a level offset does not count, and the 99th percentile leaves out the few days of a
   corporate action; `no_reference` stocks skip it), and at most `max_flat_share` (0.25) of bars with high equal to low (frozen filler
   bars). Expected days: the sessions of the exchange's official calendar
   (`exchange_calendars`, named in `configs/data.toml`) for stocks (from the ticker suffix,
   `[stock_calendars]`), indices and the softs (COCOA, COFFEE, COTTON, OJUICE and SOYBEAN on
   New York, SUGAR on London), so an exchange holiday is no missing day; every weekday for
   forex, metals and energy (a Sunday-evening session is no extra day; the few days around
   Christmas and New Year count as missing, which the 0.95 coverage allows for); every day for
   crypto. Coverage runs from the candidate's first bar in the window to the window's end, so
   bars that stop early count as missing days, and the time from the last bar to the end
   counts as a gap. The study uses the lab's limits unchanged.
3. **US stocks: regular session.** A US stock keeps only its bars of 09:30-16:00
   America/New_York on weekdays (Alpaca also has pre-market, after-market and some Saturday
   bars), for the quality check, the dollar volume and the daily closes, as in the panel.
4. **Liquidity filter.** Crypto: median daily quote volume at least `min_daily_amount`.
   Assets with a bid/ask spread: median relative spread of their 1h bars in the window at most
   `max_spread_ratio` (3) times the median of their market's assets with a spread. US stocks
   (Alpaca, no spread) have no liquidity filter. Liquidity ranks (pools, ties, the Liquid
   universe) use the median daily dollar volume for crypto and US stocks (`vwap` times
   `volume` of the regular-session bars, summed per UTC day) and the spread elsewhere.
5. **Anchors.** Each pool starts from assets chosen for their economic role, stated before
   selection: BTC and ETH; EURUSD and USDJPY; gold, copper and Brent; S&P 500; for the two
   stock pools none (the greedy pick starts from the most liquid).
6. **Near-duplicates, across all pools.** A candidate with an absolute correlation above
   `duplicate_threshold` (0.85) to an asset already kept in any pool is dropped (for example
   WTI against Brent, or a US stock against the S&P 500). The anchors of every pool count as
   kept from the start (they are never dropped), and the pools are picked one after the other
   in a fixed order (`POOL_ORDER` in `lib/_settings.py`): crypto, forex, commodities, indices,
   US stocks, non-US stocks; so a candidate is checked against the anchors of all pools, the
   picks of the pools before its own and the picks of its own pool so far.
7. **Liquid pool.** The greedy pick of a pool chooses only among its `liquid_pool_factor`
   times quota most liquid candidates that passed the filters (all of them when fewer); the
   factor is set per pool (`[options.liquid_pool_factor]`): 4 for every pool except forex, 2.
   Crypto 4 x 8 = 32; forex 2 x 16 = 32 (the pairs with the lowest spread; 63 pairs pass the
   filters, so a factor of 4 would take them all); US stocks 64 (all that pass); non-US stocks
   64; commodities and indices 32 (all that pass). The anchors are always in it and count
   toward its size.
8. **Sector cap.** In each stock pool, the universe holds at most `max_per_sector` (3) stocks
   of one GICS sector (`lib/sectors.csv`).
9. **Greedy pick within each pool.** Inside the liquid pool, repeatedly add the candidate with
   the lowest mean absolute correlation to the assets already kept in this pool (with no
   anchors, the most liquid candidate left starts), until the pool's quota is filled or no
   candidate is left. Ties go to the more liquid asset. A candidate whose sector
   is full is skipped (not dropped) and the pick continues with the others. The pick order is
   the `order` column.
10. **Correlation.** Pearson correlation of weekly log returns, pair by pair over the weeks
    both assets have a return. The weekly close is the asset's last close (mid for bid/ask
    sources) of each ISO week, Monday 00:00 to Sunday 24:00 UTC, inside the window; a week
    without a close has no return, and neither has the week after it (a return across a
    closed week is missing, never a jump). A pair needs at least `min_overlap` (100) common
    weekly returns in the selection window and `holdout_min_overlap` (20) in the held-out
    window; a pair with fewer stops the run (no pair is ever set to 0), and so does an asset
    whose returns never change (zero variance, a frozen price). One weekly matrix per window
    serves the near-duplicate step, the greedy pick and every diversity measure.

### Comparison universes (Q1)

- **Liquid:** the most liquid assets of each pool (US and non-US stocks separately), as many
  as the universe holds in that pool, with the same sector cap; no correlation step, so no
  near-duplicate step either.
- **Random:** 1000 random universes (`random_universes`) from the filtered candidates with the
  universe's number of assets per pool and the sector cap (one generator of the run's seed, 0),
  without a near-duplicate step, like Liquid. Every draw has its rows in `diversity.csv` and
  its members in `comparison_universes.csv` (about 2 MB for 1000 draws of 72 assets, so all
  members are written). The noise level is measured only for the universe, Liquid and the
  first 20 draws (`BOOTSTRAP_RANDOM_UNIVERSES`; NaN for the others), and the bootstrap of the
  gap uses the same 20 (Metrics). The global near-duplicate rule (protocol step 6) applies
  to the universe only; Liquid and Random take the universe's real count per pool, so the
  rule changes their sizes, never their own choice.
- The Q1 comparison refuses universes with a different number of assets in any pool (the
  code raises).

### Held-out diversity (the Q1 decision)

The diversity of the same universes (the universe, Liquid, every Random draw) is measured on
the held-out window 2024-01-01 to 2024-10-31 (`end` to `holdout_end`, which ends where d1
starts), with weekly correlations from that window only (`holdout_min_overlap` = 20 common
weekly returns). Q1 is decided there, on data the selection never saw; the same comparison on
the selection window, where the universe was picked, is reported next to it as in-sample.

### Number of assets (moved)

The former Q2 (does a panel model forecast better with more assets: 8, 16, 32, T1, T2, T3?) is
moved to later work (rule fix 2026-11-16): it needed T2 and T3, which are gone, and
subsets of the universe "with the quotas in proportion" have no stated rule. Q1 and Q3 keep their
numbers.

### Robustness (Q3)

Exploratory, added after the plan was locked (Plan changes, 2026-11-18). Q3 reads the test
forecast cache of 02-forecasts (`[options.ranking]`: `forecast_cache`, the test quarters
t1-t4, horizon 1); no forecast is made. `[models]` lists the models released before the test
period starts (2025-11-01): the five baselines and the eight zero-shot models of RQ1's main
family, so every model is scored on the same period, after every release.

- Losses: the CRPS of the volatility-scaled label (RQ1's loss, `src.benchmark.decision_losses`)
  on the decisions every compared model scored (`common_losses`). A set of assets is scored
  by the mean over its assets of each asset's mean CRPS (every asset weighs the same); the
  ranking is the order of these scores, compared by Kendall's tau-b.
- Two groups: the hourly models (11: the five baselines and the six hourly zero-shot models)
  on every hour they all scored, and all 13 models on Kronos's own decision points (every 167
  hours, where Kronos S and B have forecasts and every hourly model has one too).
- Random halves (the decision): 1000 splits (`splits`) of the 72 assets into two halves with
  half of every pool in each (`random_half`, one generator of the study's seed 0); tau between
  the two halves' rankings, which share no asset. Reported: the median and the 2.5 and 97.5
  percentiles of the 1000 taus.
- Markets and quarters (reported, not the decision): each market against the other four, and
  each test quarter against the other three (`part_taus`); neither pair shares a decision.
- Sharper agreement (reported, not the decision): the share of splits (and of markets and
  quarters) whose two parts rank the same model first, and the same three first in the same
  order (`agreement`). The threshold 0.7 is more lenient than planned: with the plan's four
  models it needed the same order, with 11 models it lets 8 of the 55 pairs swap, and most
  pairs differ by far more than the top ones.

### Limitations

- The candidate lists were drawn up in 2026: crypto perpetuals still trading in 2026 and
  today's large-cap stocks. This is a survivorship bias of the candidate set; it mostly
  affects return-based results (assets that failed before 2026 are missing), less the
  correlation structure.
- Coverage is measured to the end of the selection window, so an asset that stopped trading
  inside the window is dropped. This uses only data available at selection time (no
  look-ahead), but it removes assets that died inside 2020-2023.
- The held-out window is short (10 months, about 43 weekly returns), so its correlations are
  noisier than those of the selection window; the bootstrap interval of the gap is that of the
  selection window, and it is in-sample (the selection is not redone inside a draw). It is
  the normal interval around the point gap, because the percentile interval is shifted below
  the point gap by the in-sample optimism and the repeated weeks (Metrics).
- Effective-rank levels are compared only within a window: with 43 weeks and about 72 assets
  the held-out matrix has rank at most 43 (noise level about 31 there, about 60 on the
  selection window; `noise_effective_rank`).
- Pairs with a short held-out overlap (WAVESUSDT and OCEANUSDT, 20-25 weeks) have correlation
  standard errors near 0.22 and produce the largest negative eigenvalues. They enter only
  Random draws, and the decision is unaffected: the review's check (2026-11-16) repaired each
  held-out matrix to positive semi-definite (negative eigenvalues set to 0, diagonal rescaled
  to 1) and the Random 90th percentile stays at 20.27.
- `holdout_min_overlap` = 20 equals the shortest held-out pair today (WAVESUSDT with
  MAERSK-B.CO), so a candidate delisted earlier in 2024 would stop the run (accepted risk).

### Plan changes

- 2026-11-16: the first Q1 run uses partial stock data. The owner approved running Q1 as soon
  as all 15 commodities and 21 indices were complete, with whatever non-US stocks were
  complete by then, because the Dukascopy download is limited to about 90 files per hour.
  At run time the stock candidates are the 47 US stocks that pass the quality checks (Alpaca;
  IBM fails the Yahoo check) and no non-US stocks (Europe 0, Japan 0, Hong Kong 0 of 80
  listed; still downloading). The plan itself is unchanged and stays locked. Q1 is rerun with
  a new run id when all data is in, and the stock tier of this first run is reported as
  US-only.
- 2026-11-16, second run (all data in, as approved): the non-US stocks are now built from 1h
  Dukascopy bars (their 15m download is too slow; the universe uses 1h closes only), 70 of the
  80 pass the data checks. Q1 is rerun with a new run id on the full candidate set; the first
  run stays as the US-only record. The selection code now lives in this folder
  (`universe/`) and each run keeps the manifest it read.
- 2026-11-16, implementation fix (no rule change): the data filter's "data from 2020-01-01"
  allowed a first bar at most one day late, so instruments whose market first opened on
  2 January 2020 after 00:00 UTC (Brent, gasoil, the European indices) were dropped and the
  Brent anchor was missing. The tolerance is now 7 days (`START_TOLERANCE`, with a test). The
  first attempt failed on the missing anchor before selecting anything.
- 2026-11-16, rule fix (exchange calendar; owner's approval 2026-11-16): coverage and the gap
  rule were measured against every weekday (the manifest's numbers over the whole history),
  so a market holiday counted as missing data. All 13 Japanese stocks were dropped for a
  "gap 7.79 days" (New Year and Golden Week), the six softs (COCOA, COFFEE, COTTON, OJUICE,
  SOYBEAN, SUGAR; coverage 0.960-0.968, about 10 US exchange holidays a year) and CHE20 and
  PLN20 (0.964, 0.960) for coverage. Now both rules use the exchange's own calendar from the
  data, over the selection window only (protocol steps 1-2); `max_gap_days` stays 7, now
  counted in missed open days. Venues: stock ticker suffix, commodity and index groups in
  `[options.venues]`, one venue for crypto and one for forex. The drop reasons are reordered
  (pegged, quality, start, coverage, gap) so a late start is reported as such.
- 2026-11-16, rule fix (pegged currencies; owner's approval 2026-11-16): USDHKD and EURDKK
  were picked into T1, but a pegged pair barely moves and adds no market behaviour. Forex
  pairs with a currency of `pegged_currencies = ["HKD", "DKK"]` are dropped before selection
  with the reason "pegged currency" in `universe.csv`; CNH stays.
- 2026-11-16, rule fix (liquid pool; owner's approval 2026-11-16): the greedy pick favoured
  small fringe coins (BLZ, FLM, RLC, OMG in T1) because they are the least correlated. It now
  chooses only among the `liquid_pool_factor = 4` times T1-quota most liquid candidates of
  each pool (all of them when fewer), anchors included; the rest follow in liquidity order
  (protocol steps 6-7). The crypto volume threshold and the spread-ratio filter are unchanged
  and still remove very illiquid assets first.
- 2026-11-16, rule fix (selection window; owner's approval 2026-11-16): the statistics used
  2020-01-01 to 2025-07-31, which overlaps the development folds d1-d3 (2024-11 .. 2025-07),
  so the universe was chosen with data those folds evaluate. The window now ends 2023-12-31
  (`end = "2024-01-01"`, exclusive).
- 2026-11-16, rule fix (crypto listing date; owner's approval 2026-11-16): crypto must be
  listed by 2020-12-31 (was 2020-09-30), so the ten perpetuals listed in October and November
  2020 (AAVE, FIL, NEAR, KSM, ...) become candidates; the other markets still need data from
  2020-01-01.
- 2026-11-16, sectors (no selection rule; owner's choice 2026-11-16): there are no sector
  quotas. `universe.csv` has a `sector` column for stocks from `universe/sectors.csv` (the
  GICS sector of each of the 128 stock candidates, written in by hand from the companies'
  published GICS classifications, MSCI and S&P Global), and each run writes
  `tables/sector_mix.csv`, the number of kept stocks per sector in each tier.
- 2026-11-16, rule fix (commodity start; owner's approval 2026-11-16): commodities may start by
  2021-12-31 (`[options.listed_by]`), so platinum (first bar 2021-11-01) and palladium
  (2021-07-04), the only Dukascopy commodities left out by the 2020 start, join the commodity
  pool; their returns are missing before their first bar. No commodity ETF is added (owner).
- 2026-11-16, rule fix (quotas and stock pools; owner's approval 2026-11-16): T1 quotas
  crypto 8, forex 16, commodities 8, indices 8, stocks 32 as two pools picked separately, US
  16 and non-US 16; T2 and T3 in the same proportions (both removed later the same day); no
  sector quotas (replaced by the sector cap below). The commodity T1 quota stays 8; fewer
  passing commodities give all they have and the real count is recorded. T1 is now up to 72
  assets instead of 64. The Liquid and Random universes use the same quotas and pools. The Q1
  decision rule is unchanged. Q1 must be rerun with a new run id; the run
  `2026-11-16T1410-all-data` stays as the record of the old rules.
- 2026-11-16, rule fix (US stock liquidity; owner's approval 2026-11-16): Alpaca bars have no
  bid/ask, so every US stock got its market's median spread score; the ties made the Liquid
  universe's US stocks and the first US pick follow the manifest order. A US stock's liquidity
  is now its median daily dollar volume, `vwap` times `volume` of its Alpaca 1d bars (columns
  checked on AAPL: open, high, low, close, volume, trades, vwap). It ranks the US pool, its
  ties and the Liquid universe; a pool never mixes dollar volumes and spreads (the code raises).
  US stocks have no spread, so the spread filter does not apply to them, and the stock
  market's median spread now uses only the stocks with a spread. No score is filled with a
  market median any more: an asset without a volume or spread stops the run.
- 2026-11-16, index calendars approved by the owner (2026-11-16): the index venues americas,
  emea and asia_pacific of `[options.venues.index]`.
- 2026-11-16, wording fix (no rule change; owner's choice 2026-11-16): the title, Q1, Q2, H2
  and the Q2 design said "64" for the main universe; they now say "the T1 universe", since its
  size follows from the quotas and the passing candidates.
- 2026-11-16, rule fix (sector cap; owner's approval 2026-11-16): in each stock pool, T1
  takes at most `max_per_sector = 3` stocks of one GICS sector, so a pool cannot fill up with
  one cluster (for example banks). The greedy pick skips a candidate whose sector is full and
  continues; the Liquid and Random comparison universes follow the same cap so the comparison
  stays fair. This replaces "no sector quotas" in the sectors entry above.
- 2026-11-16, rule fix (T1 only; owner's approval 2026-11-16): the paper needs only T1, so T2
  and T3 are removed: the quotas are one number per pool, the greedy pick stops at the quota
  (the candidates outside the liquid pool are no longer appended in liquidity order),
  `universe.csv` has `tier` 1 for T1 and nothing else, and `sector_mix.csv` counts T1 only.
  The former Q2 (number of assets, which needed T2 and T3) moves to later work; Q1 and Q3 keep their numbers.
- 2026-11-16, no survival condition (owner's approval 2026-11-16): no asset is required to
  trade until the end of the test period; an asset that stops trading is scored up to its
  last bar and reported, and `universe.csv` records each candidate's `last_bar` from the
  manifest. The benchmark code is unchanged (labels simply stop; a test in
  `tests/benchmark/test_build.py` checks it).

- 2026-11-16, data (no rule change): commit 551f364 rebuilt the non-US stocks from their 15m
  Dukascopy bars like every other Dukascopy group (all 1120 yearly 15m parts are on disk); the
  1h build of the second-run entry above is gone. 1h bars resampled from 15m match Dukascopy's
  own 1h bars in 99.8 % of bars on average. ROG.SW still has no Yahoo reference.
- 2026-11-16, rule fix (data quality inside the selection window; owner's approval
  2026-11-16): the study's own venue calendar (open days taken from the candidates' data,
  `[options.venues]`, `min_coverage = 0.98`, a gap of 7 missed open days) counted
  Sunday-evening sessions as open days (Brent coverage about 0.83, so the Brent anchor was
  missing), mixed holiday calendars inside a venue (softs, CHE20, PLN20) and gave a
  single-member exchange (.MI, .HE, .BR, .CO) coverage 1 always; and the manifest's
  `passed`/`reason`, computed over 2020-2026, removed candidates for problems after the
  selection window (MAERSK-B.CO for a gap in 2024), a hidden look-ahead and survival
  condition. Now the lab's standard check, `src.data.check_quality` with the limits of
  `configs/data.toml` (official exchange calendars for stocks, weekdays for forex,
  commodities and indices, every day for crypto; Yahoo agreement for stocks with a
  reference), runs on each candidate's bars cut to the selection window (protocol step 2).
  The manifest only lists the candidates. `venues`, `min_coverage` and `max_gap_days` leave
  `study.toml`.
- 2026-11-16, rule fix (one daily correlation matrix; owner's approval 2026-11-16): hourly
  correlations needed 500 common hours, and a pair with fewer (15 % of non-US stock pairs and
  8 % of US x non-US pairs never share an hour) counted as correlation 0, so the greedy pick
  preferred assets that never trade at the same time. Every correlation (near-duplicates,
  greedy pick, diversity) is now the Pearson correlation of daily close-to-close log returns
  (protocol step 10), one consistent matrix; `min_overlap` is now 100 common daily returns
  (the 10-month held-out window has about 140 for two weekday markets of different regions;
  100 gives a correlation a standard error of about 0.1). A pair below it stops the run
  instead of counting as 0. Daily closes of different time zones are not synchronous, which
  lowers cross-region correlations a little (a limitation of the results). A matrix built pair
  by pair can have a negative eigenvalue: it is reported, not hidden (Metrics).
- 2026-11-16, rule fix (held-out diversity; owner's approval 2026-11-16): Q1 judges the
  universe on the same window it was picked on, so its answer is in-sample. The diversity of
  the same universes is now also reported on the held-out window 2024-01-01 to 2024-10-31
  (`holdout_end = "2024-11-01"`), as a robustness line in the results README and its
  limitations; the decision rule is unchanged and stays on the selection window.
- 2026-11-16, rule fix (comparison sizes; owner's approval 2026-11-16): Liquid and Random were
  built with the quotas while the universe can be shorter (a pool cut short by near-duplicates
  or the sector cap; the old run had 61 against 62). They now take the universe's real count
  in each pool, and the Q1 comparison raises when the sizes differ; Q1's decision rule says
  "the same number of assets per pool" instead of "the same quotas".
- 2026-11-16, rule fix (US regular session; owner's approval 2026-11-16): Alpaca's US stock
  bars include pre- and after-market and some Saturday bars, which entered the closes and the
  dollar volume. A US stock now keeps only its weekday bars of 09:30-16:00 America/New_York
  (`src.utils.in_session`), as in the panel (protocol step 3).
- 2026-11-16, rule fix (liquidity and start; owner's approval 2026-11-16): the spread was one
  end-of-day value per day; it is now the median relative spread of the 1h bars in the window.
  The US dollar volume is `vwap` times `volume` of each regular-session 15m bar summed per UTC
  day (the stored 1d `vwap` is a sum of 15m `vwap` values, not a price). The start tolerance
  moved from the code to `study.toml` (`start_tolerance_days = 7`) and is no longer added to a
  `listed_by` date.
- 2026-11-16, rule fix (Global greedy removed; owner's approval 2026-11-16): the "Global
  greedy" comparison universe needed a disabled near-duplicate step and sector cap and was
  not part of any decision rule; it is gone, and Q1 compares the universe with Liquid and
  Random only.
- 2026-11-16, names (no rule change; owner's choice 2026-11-16): "T1" is now "the universe"
  everywhere in the plan and the code (earlier entries above keep the old name); the
  `universe.csv` column `tier` (1 for T1) is now `in_universe` (true or false), read by the
  benchmark (`src.benchmark`); the universe's heatmap is `universe_correlation`. The study's
  code moved from `universe/` to `lib/` in this folder (the lab's rule for study-owned code).

- 2026-11-16, rule fix (index calendars; owner's approval 2026-11-16, commit ad6c8ef):
  indices were counted on every weekday, so an exchange holiday counted as a missing day (HKG
  coverage 0.944 on 2020-2023). Each index now has its exchange's calendar in the `calendars`
  table of the index group in `configs/data.toml` (New York for the US indices, DOLLAR and VIX;
  Singapore for CHIA50 and SGDBLUE; the home exchange otherwise); HKG covers 0.998, and every
  index passes except ITA40 (23 days of real gaps). Protocol step 2 says which markets use
  exchange sessions: stocks, indices and (entry below) the softs; forex, metals and energy use
  weekdays, crypto every day.
  Later the same day (data rebuild): CHIA50 and SGDBLUE (Singapore, XSES) are counted on
  weekdays again, because `exchange_calendars` 4.11 has no XSES sessions after 2025-12-31 and
  the manifest refresh stopped on them (`configs/data.toml`).
- 2026-11-16, rule fix (review of 2026-11-16, code only; the data rebuild and the Q1 rerun wait
  for the owner):
  - Delisted coins: Binance writes frozen bars (constant price, volume 0, 0 trades) after a
    delisting (WAVESUSDT from 2024-06-11, OCEANUSDT from 2024-06-26, CVCUSDT from 2022-11-30)
    and during outages. A bar without a trade is now left out when the bars are built (a closed
    market is masked, never filled). In the selection window only CVCUSDT has such bars (38178
    of them), so after the rebuild its bars stop on 2022-11-30 and it fails coverage.
  - New quality limit `max_flat_share = 0.25` in `configs/data.toml` (the flat share was
    measured but never used): on 2020-2023 the real data of every candidate stays at or below
    0.150 (VIX; median 0.0001, 99th percentile 0.047), the frozen CVCUSDT series had 0.347.
  - Coverage and the gap rule run to the end of the window, not to the candidate's last bar,
    so data that stops early inside the window is seen.
  - An asset with zero variance stops the run with its own error ("zero variance") instead of
    being reported as a pair with too few common days.
  - Rule 13: the universe's effective rank on the selection window gets a moving-block
    bootstrap interval (Metrics; new options `bootstrap_block_days`, `bootstrap_resamples`,
    `interval_level`), reported in `diversity.csv` and the Q1 answer.
  - The Q1 comparison checks the number of assets in every pool, as the decision rule says,
    not only the total.
  - `universe.csv` says why every filtered candidate is not in the universe (`outside liquid
    pool`, `not picked`, `not picked: sector cap (... full)`).
- 2026-11-16, rule fix (softs calendars; review of 2026-11-16): the softs were counted on every
  weekday, so their exchange holidays counted as missing (coverage 0.964-0.967 on 2020-2023).
  On 2020-2023 COCOA, COFFEE, COTTON, OJUICE and SOYBEAN miss 36 of the 37 New York exchange
  holidays that fall on weekdays and almost no other day, so they use New York (XNYS); SUGAR
  misses 33 of the 35 London holidays and only 13 New York ones, so its Dukascopy series
  follows London (XLON), not ICE US. Their coverage is now 0.998-1.000. Metals and energy miss
  only about 7 weekdays in four years (Christmas, New Year), no exchange's holiday pattern, and
  stay on weekdays. The stock calendars (by ticker suffix) moved from the code into
  `[stock_calendars]` of `configs/data.toml`, so every calendar is decided in one place.

- 2026-11-16, rule fix (weekly correlations; owner's approval 2026-11-16): every correlation
  (near-duplicates, greedy pick, diversity of both windows, the bootstrap interval) is now
  that of weekly log returns: each asset's last close of each ISO week (UTC), a week without a
  close is missing, returns between consecutive weekly closes (protocol step 10; replaces the
  daily matrix of the entry above). Daily closes of different time zones are not synchronous
  (Tokyo and Hong Kong close before New York opens, so one day's news shows up a day apart),
  and a daily return across a closed weekend was missing, so Monday's move of every
  weekday market was dropped; weekly closes remove both. `min_overlap` now counts weekly
  returns: 100 in the selection window (208 weekly returns; on the data of 2026-11-16 the
  shortest pair of the filtered candidates has 112, platinum with the other commodities), a
  new `holdout_min_overlap = 20` for the held-out window (43 weekly returns; coins delisted in
  it, like WAVESUSDT from 2024-06-11, have about 23, and the shortest pair has 20, WAVESUSDT
  with MAERSK-B.CO) and a new `bootstrap_min_overlap = 20` for each bootstrap draw (a draw
  holds about 112 +- 14 weeks of platinum, so 100 would often stop the run). The bootstrap
  blocks are `bootstrap_block_weeks = 4` (about a month, as the approved 30 days; the settings
  1000 draws and 95 % are approved unchanged).
- 2026-11-16, rule fix (Q1 on the held-out window; owner's approval 2026-11-16): Q1 is now
  decided on the held-out window 2024-01-01 to 2024-10-31 (the universe's effective rank there
  at or above the 90th percentile of the 20 Random universes and above Liquid, all on that
  window); the selection-window numbers are reported as in-sample. The clause "the rules are
  then revised on the same pre-test data" is removed from the rule. This replaces the
  held-out entry above, where the held-out window was robustness only.
- 2026-11-16, rule fix (Yahoo check on returns; owner's approval 2026-11-16): the lab-wide
  Yahoo check (`src.data.check_quality`) compared price levels, so an offset of the level (a
  different adjustment of past prices) counted on every day. It now compares daily close-to-close log
  returns of our session closes with Yahoo's on common days; the limits are
  `yahoo_return_median_bps = 50` and `yahoo_return_p99_bps = 250` in `configs/data.toml`
  (renamed from `yahoo_median_bps` and `yahoo_p99_bps`). On 2020-2023 (data of 2026-11-16) the
  US stocks have a median of 1.5-4 bps and a 99th percentile of 11-41, the non-US stocks
  (Dukascopy's last quote against the closing auction) 8-27 and 40-170; on 2020-2026 the
  largest are 30 (NESTE.HE) and 183 (8035.T). A corporate-action day moves one or two returns
  (GSK.L 2094 bps on 2022-07-18, the Haleon demerger with a 0.8 share consolidation that Yahoo
  dates 2022-07-19; SIE.DE 970 bps on 2020-09-28, the Siemens Energy spin-off), and the 99th
  percentile leaves those out (four years of
  about 1000 returns absorb about 5 such days), so no special rule for split days is needed.
- 2026-11-16, rule fix (near-duplicates across all pools; owner's approval 2026-11-16): a
  candidate is now dropped when its absolute correlation is above 0.85 to an asset kept in
  any pool, not only its own: the anchors of every pool count from the start, then the pools
  are picked in the fixed order crypto, forex, commodities, indices, US stocks, non-US stocks
  (protocol step 6). The greedy pick's mean correlation still uses the pool's own kept
  assets. Liquid and Random have no near-duplicate step, so they stay unchanged apart from
  the universe's count per pool.
- 2026-11-16, Q3 seeds (owner's choice 2026-11-16): Q3 uses three Random universes drawn with
  seeds 0, 1 and 2 (Robustness (Q3)).
- 2026-11-16, limitations (owner's approval 2026-11-16): the survivorship bias of the 2026
  candidate lists and the dropping of assets that stopped inside the selection window are
  stated under Limitations and in the results' limitations line (`analyze.py`).
- 2026-11-16, rule fix (1000 Random universes; owner's approval 2026-11-16): Q1 compares the
  universe with 1000 Random universes instead of 20 (`random_universes = 1000`), because the
  90th percentile of 20 draws is too noisy. Q3's three Random universes (seeds 0, 1 and 2)
  are unchanged. Every draw keeps its rows in `diversity.csv` and its members in
  `comparison_universes.csv` (about 2 MB); the point gap uses the median of all draws, while
  the gap's bootstrap and the noise levels use the first 20 draws (all 1000 would take about
  20 minutes and 80 seconds; Metrics).
- 2026-11-16, rule fix (normal interval of the gap; owner's approval 2026-11-16): the gap's
  bootstrap interval is now the normal interval `gap +- z * SE`, SE the standard deviation of
  the paired bootstrap gaps, `z` from `interval_level`; the paired draws are unchanged. The
  percentile interval did not hold its own point gap (9.77 above 6.59 to 9.28), because it is
  shifted by the in-sample optimism and the repeated weeks (Metrics, Limitations).
- 2026-11-16, rule fix (forex liquid pool; owner's approval 2026-11-16): `liquid_pool_factor`
  is set per pool (`[options.liquid_pool_factor]`): crypto, commodities, indices and both stock
  pools 4, forex 2, so the forex liquid pool is the 2 x 16 = 32 pairs with the lowest spread.
  Only 63 forex pairs pass the filters, fewer than 4 x 16, so a factor of 4 took every pair
  and illiquid pairs such as USDTRY, EURHUF and USDTHB were picked (protocol step 7).
- 2026-11-16, review fixes (metrics only, decision rule unchanged): `diversity.csv` gains the
  noise level `noise_effective_rank`; the selection window's bootstrap interval of the
  universe's effective rank (shifted down by repeated weeks) is replaced by the paired interval
  of its gap to the Random median (`gap`, `gap_low`, `gap_high`); the members of every compared
  universe are saved in `tables/comparison_universes.csv`; the negative-eigenvalue warning
  ignores rounding above -1e-8 (Metrics, Limitations).
- 2026-11-17, output column (no rule change; owner's approval 2026-11-17): `universe.csv` gains
  `passed_data_filter` (true when the candidate passed the data filter of the selection window,
  in the universe or not), which 07-rq5-off-hours reads for its Q2 assets instead of matching
  the text of `dropped`. Rebuilt as `2026-11-17T1235-data-filter-column`: the same 72 assets,
  every other column of `universe.csv` and `diversity.csv`, `sector_mix.csv` and
  `comparison_universes.csv` identical to `2026-11-17T0005-random-1000`; 274 of the 292
  candidates pass the data filter, the same set the reason match gives. 02-forecasts keeps
  `2026-11-17T0005-random-1000`.
- 2026-11-18, Q3 amended (owner's approval 2026-11-18; exploratory, after the plan lock and
  after the test results of RQ1 were seen): the plan compared the model rankings on the
  Liquid and three Random universes, which needs new zero-shot forecasts for 115 assets
  (on the local GPU, mostly for Kronos) that the paper's budget does not allow. Q3 now tests the
  stability of the ranking inside the universe instead (random halves of the pools, the
  decision; markets and test quarters, reported) on the existing test forecasts, with the
  same threshold 0.7 (more lenient than planned with 11 models; the first-model and
  first-three agreement is reported next to it), all models on the same test period after every compared model's
  release (Robustness (Q3)). The comparison on other universes is left to later work and
  stated as a limitation. `[models]` lists the compared models and `[options.ranking]` the
  settings.

## Data

Candidates are the instruments of `data/manifest.json`, listed in `configs/data.toml`. The lists
were set when the quotas were crypto 16, forex 16, commodities 8, indices 8 and stocks 16: four
times each quota where the source has enough, eight times for stocks. Against today's quotas
crypto has 8 times its quota (64 = 8 x 8), forex 4 times, the US stocks 3 times and the non-US
stocks 5 times:

| Pool | Quota | Candidates | Source |
|---|---|---|---|
| Crypto | 8 | 64: the 54 perpetuals listed by 2020-09-30, then the next 10 (Oct-Nov 2020); all 64 are listed by 2020-12-31 | Binance USD-M |
| Forex | 16 | 64: majors, crosses, Scandinavian and emerging-market pairs (4 with HKD or DKK are dropped as pegged) | Dukascopy |
| Commodities | 8 | 15: everything Dukascopy has | Dukascopy |
| Indices | 8 | 21: everything Dukascopy has | Dukascopy |
| US stocks | 16 | 48 | Alpaca |
| Non-US stocks | 16 | 80 from Europe, Japan and Hong Kong | Dukascopy |

The selection reads each candidate's base bars once, `data/<market>/<symbol>/ohlcv/15m.pq`
(every group is built from 15m bars, the non-US stocks too since commit 551f364), from
`start` to `holdout_end`; from them it takes the quality check, the liquidity (15m traded value
summed per UTC day; the spread of the 1h bars built from 15m) and the daily closes (mid for
bid/ask sources; the weekly closes of the correlations are the last of these daily closes in
each week). Binance bars without a trade (volume and trades 0: the frozen bars Binance
writes during an outage and for months after a delisting) are left out when the bars are built
(`src.data`), so a delisted coin's data stops at its last trade. Stocks with a Yahoo reference
also read
`data/stock/<symbol>/reference/yahoo/1d.pq`, cut to the selection window. The quality limits
are `[quality]` of `configs/data.toml`. The GICS sector of each stock candidate is in
`lib/sectors.csv` (for the sector cap and the sector mix).

## Metrics

- Diversity of each universe, on the selection window and again on the held-out window:
  mean absolute correlation, effective rank (the exponential of the entropy of the normalised
  eigenvalues), the number of principal components that explain 80 % of the variance, the
  share of the first component, and the smallest eigenvalue (a matrix built pair by pair can
  have small negative eigenvalues; they count as 0 in the other measures, the smallest is
  reported in `min_eigenvalue` and the run log warns about those below -1e-8,
  `NEGATIVE_EIGENVALUE_TOLERANCE`: a rank-deficient matrix, such as the held-out one, has
  zero eigenvalues that rounding puts near -1e-15, which are no finding).
- Noise level (`noise_effective_rank` in `diversity.csv`, quoted in the Q1 answer): the
  effective rank of independent Gaussian series with each universe's own missing data in that
  window, the mean of 10 draws (`NOISE_DRAWS`) with the run's seed; measured for the universe,
  Liquid and the first 20 Random draws only (about 40 ms per row and window, so 80 s for all
  1000 draws). Levels are compared only
  within a window: with 43 weeks and about 72 assets the held-out matrix has rank at most 43.
- Interval of the gap: on the selection window, the universe's effective rank minus the
  median of all 1000 Random universes (`gap`) comes with a 95 % (`interval_level`) paired
  moving-block bootstrap interval over weeks (`src.stats.block_indices`; blocks of
  `bootstrap_block_weeks` = 4 weeks, about a month like the approved 30 days, so volatility
  clusters stay together; `bootstrap_resamples` = 1000 draws; the run's seed). Each draw
  rebuilds the correlations from the resampled weeks and measures the universe and the first
  20 Random universes (`BOOTSTRAP_RANDOM_UNIVERSES`) on the same weeks; the bootstrap's Random
  median uses these 20 because all 1000 would take about 20 minutes (1000 resamples times
  1000 effective ranks), with 20 it takes under a minute. The interval is the normal interval
  `gap +- z * SE` (`gap_low`, `gap_high` in `diversity.csv`): SE is the standard deviation of
  the paired bootstrap gaps and `z` the normal quantile of `interval_level` (1.96 for 95 %).
  It is not the percentile interval of the bootstrap gaps, because that one is shifted: the
  universe was picked on these weeks (in-sample optimism) and repeated weeks lower its tuned
  rank more than a Random one's, so the point gap lay above the percentile interval on the
  real data (gap 9.77, percentile interval 6.59 to 9.28 in review). The normal interval
  keeps the bootstrap's spread and holds the point gap by construction. A pair needs
  `bootstrap_min_overlap` (20) common weekly returns in a draw (a draw holds about 112 +- 14
  weeks of platinum, the latest start, so the window's 100 would stop the run). The interval
  is in-sample (the selection is not redone inside a draw) and is quoted in the Q1 answer; it
  does not change the decision rule. The held-out window (43 weeks, some assets with about
  20) is too short for it.
- `diversity.csv` has each universe's number of assets in every pool (`size_<pool>`).
- Q3: Kendall tau-b between the rankings by mean CRPS of two disjoint parts (Robustness (Q3)).

## Decision rule

Every choice here is made on four development folds whose validation quarters are
2024-11..2025-01, 2025-02..04, 2025-05..07 and 2025-08..10 (training always ends before each
quarter). These months are never test data. After the choices are fixed, the four test folds
(2025-11 to 2026-10) run once for the final numbers. The folds are defined in
[the README](../../README.md#folds). Each question's own rule is under
[Questions](#questions).

Outputs (`runs/<run-id>/`): `tables/universe.csv` (every candidate: market, source, last bar,
the quality numbers of the selection window and its first bar there, `passed_data_filter`
(true when it passed the data filter of the selection window, in the universe or not), pool,
pick order, `in_universe`, dollar volume, spread, liquidity, why it is not in the universe (the filter that
dropped it, `near-duplicate of ...`, `outside liquid pool`, `not picked: sector cap (... full)`
or `not picked`; empty for the universe's assets), sector of a stock),
`correlation.parquet` and `holdout_correlation.parquet` (weekly correlations of the filtered
candidates in the two windows), `tables/diversity.csv` (one row per universe and window, with
the size of each pool, the noise level and the universe's gap to the Random median with its
interval; every Random draw has its rows), `tables/comparison_universes.csv` (the members of
the universe, Liquid and all 1000 Random draws: `universe`, `symbol`, `pool`; about 2 MB), `tables/sector_mix.csv` (stocks of the universe per
sector and stock pool), the copy of
`manifest.json`, and from `analyze.py` `tables/universe_correlation.csv` with its figure and
Q3's `tables/ranking_scores.csv` (each group's model scores and ranks), `tables/ranking_parts.csv`
(each market and test quarter against the rest) and `tables/ranking_splits.csv` (each random
split's tau and agreement). In
the paper: one sentence per rule in the main text, the full table in the appendix.

## Compute

- Selection and diversity (Q1): CPU only, minutes (reading the 15m bars of 292 candidates).
- Robustness (Q3): CPU only, minutes (reads the test forecast cache; no forecast is made).

## Status

Plan approved and locked by the owner (2026-11-15) for the development data; rule fixes of
2026-11-16 under Plan changes. Q1 code is written (`lib/` in this folder) with weekly
correlations, the held-out decision and the cross-pool near-duplicate rule; the data rebuild
(crypto without Binance's frozen bars, US stocks) is done. After the rule fixes of 2026-11-16
(1000 Random universes, the normal interval of the gap, the forex liquid pool) Q1 ran as
`2026-11-17T0005-random-1000`: yes on the held-out window (effective rank 20.70 against the
random 90th percentile 20.35 and Liquid 17.29; [results/README.md](results/README.md));
`2026-11-16T2215-rules-final` stays as the record of the earlier rules. 02-forecasts reads
the new run. Q3 (amended 2026-11-18, exploratory) is answered by `analyze.py` of the same run from
the test forecast cache. The former Q2 moved to the next paper.
