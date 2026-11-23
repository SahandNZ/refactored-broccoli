# Results: 13-context-baselines

Plan: [../README.md](../README.md). Runs: test-2026-11-18-v2.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-18-v2 | 2026-11-18 04:02 / 05:02 | 293.0 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 24.79 / 0.0 GB | - | cf5f4334178d7b26e3dd7a77246b68a33cc2cf44 |

## Q1 (RQ-null)

At 1 bar, does any zero-shot model of rq1's main family beat the best baseline when the random walk, climatology, GARCH(1,1) and the ridge regression are fitted only on the last 512 own bars before each decision?

Primary metric: CRPS of the vol-scaled return.

**Q1**: no. Test quarters, 1 bar, rq1's rule with the random walk, climatology, GARCH(1,1) and the ridge regression refitted on the last 512 own bars: 0 of 40 tested cells beat the best baseline. Best baseline per market: commodity climatology, crypto climatology, forex climatology, index climatology, stock climatology. Skill against it: median -9.1 % (rq1: -9.2 %); 0 positive (rq1: 0). GARCH daily fits that did not converge: 116 (their days have no GARCH forecast).

Limitations: exploratory, planned after rq1's test results were seen; one horizon (1 bar); a decision needs 512 earlier decisions of the cache, so the first months of d1 have no context baseline on the markets that close and the best baseline is chosen on fewer development decisions; GARCH is refitted daily, not at every decision

Files: [test-2026-11-18-v2/tables/verdict.csv](test-2026-11-18-v2/tables/verdict.csv), [test-2026-11-18-v2/tables/compared.csv](test-2026-11-18-v2/tables/compared.csv), [test-2026-11-18-v2/tables/garch_not_fitted.csv](test-2026-11-18-v2/tables/garch_not_fitted.csv)
