# Results: 05-rq3-net

Plan: [../README.md](../README.md). Runs: test-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-17 | 2026-11-17 19:12 / 20:12 | 178.6 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 2.74 / 0.0 GB | - | c7f490f830f9452fa400cf195c94bbb305898b0e |

## Q1 (RQ-net)

Does trading the sign of a model's median earn money after each market's costs, and up to which cost?

Primary metric: break-even cost per side in bps (the cost at which the mean net return is
zero).

**Q1**: no. Main family: 0 of 120 scored cells earn money after costs: none. Late-release family (own verdict): 0 of 71 scored cells earn money after costs: none. The break-even cost per side is above the cost paid in 28 of 191 scored cells (every zero-shot cell's break-even and its interval are in verdict.csv, the baselines' cells in cells.csv; per-quarter means in quarters.csv; the one-bar delay check in delayed.csv). Highest break-even per market at 1h: commodity kronos-small (main family) 1.5 bps [-1.1, 4.4] vs cost 7.2 bps; crypto moirai-2-small (main family) 0.5 bps [-2.4, 3.5] vs cost 5.0 bps; forex kronos-small (main family) 0.4 bps [0.1, 0.7] vs cost 0.8 bps; index timesfm-3 (late family) 1.3 bps [0.4, 2.2] vs cost 2.3 bps; stock t0-gamma (late family) 3.8 bps [-0.9, 9.5] vs cost 2.0 bps. linear (reference), break-even at 1h: commodity -0.5 bps vs cost 8.3 bps (net -2.4 bps per book time); crypto -0.4 bps vs cost 5.0 bps (net -4.0 bps per book time); forex 0.2 bps vs cost 0.8 bps (net -0.4 bps per book time); index 0.2 bps vs cost 2.7 bps (net -0.6 bps per book time); stock 0.1 bps vs cost 2.1 bps (net -0.4 bps per book time). buy-and-hold (reference), break-even at 1h: commodity 247.3 bps vs cost 7.4 bps (net 0.3 bps per book time); crypto -708.1 bps vs cost 5.0 bps (net -0.2 bps per book time); forex -0.3 bps vs cost 0.8 bps (net -0.0 bps per book time); index 153.7 bps vs cost 2.8 bps (net 0.2 bps per book time); stock 148.6 bps vs cost 2.0 bps (net 0.2 bps per book time). One-bar delay (robustness, delayed.csv): in 161 scored cells with a delayed book, the mean net return is lower with the delay in 86, and the delayed break-even is above the on-time cell's cost in 26. 16 cells have fewer than 10 bootstrap blocks (has_few_blocks); 0 cells trade less than 0.05 per effective time (is_low_turnover, close to buy and hold). Cells not scored (4, in the table with their status): 1 not scored: 18 effective book times, fewer than 5 blocks of 16; 1 not scored: 41 effective book times, fewer than 5 blocks of 12; 1 not scored: 45 effective book times, fewer than 5 blocks of 16; 1 not scored: 92 effective book times, fewer than 5 blocks of 20.

Limitations: test folds; only decisions after each model's release; equal weight 1/n over each market's assets (an asset without a position is cash; the stock market pools US and non-US stocks), each position held for the horizon in the asset's own bars; fixed costs per side, the final closing trade charged, no impact, funding or delay (delay only in the robustness check); one-sided tests at 0.05, Holm per family over all horizons and markets; the intervals are 95% unadjusted; US stock costs are an assumed 1 bp half spread

Files: [test-2026-11-17/tables/verdict.csv](test-2026-11-17/tables/verdict.csv), [test-2026-11-17/tables/cells.csv](test-2026-11-17/tables/cells.csv), [test-2026-11-17/tables/quarters.csv](test-2026-11-17/tables/quarters.csv), [test-2026-11-17/tables/reference.csv](test-2026-11-17/tables/reference.csv), [test-2026-11-17/tables/delayed.csv](test-2026-11-17/tables/delayed.csv), [test-2026-11-17/tables/costs.csv](test-2026-11-17/tables/costs.csv)
