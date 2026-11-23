# Results: 02-forecasts

Plan: [../README.md](../README.md). Runs: dev-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| dev-2026-11-17 | 2026-11-17 10:48 / 11:48 | 1714.2 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 6.0 / 2.87 GB | - | fe46d5be139f6cd83eab29023bbbd44723d6df70 |

## Q1 (RQ-forecasts)

Does every model give a valid forecast for every decision of the benchmark windows?

Primary metric: share of labelled decisions after the model's release with a valid forecast,
per model and window.

**Q1**: yes. 13 models with decisions after their release; lowest share of labelled decisions with a valid forecast 0.9990 (chronos-2); per window: d1 yes, d2 yes, d3 yes, d4 yes.

Limitations: development folds only; only decisions after each model's release count; valid means finite, ordered quantiles, not accurate

Files: [dev-2026-11-17/tables/validity.csv](dev-2026-11-17/tables/validity.csv)

## Kronos interval dispersion (diagnostic, not a plan question)

Kronos samples with its reference settings (temperature 1, top_p 0.9, top_k off; 20 paths per
seed, 3 seeds pooled to 60). Its 10-90 % interval is too narrow: on all development decisions
the 60 pooled paths cover the label in 0.62 / 0.59 / 0.47 of the decisions at 1 / 4 / 24 bars
(Kronos-small) and 0.63 / 0.61 / 0.45 (Kronos-base), against 0.80 for a calibrated forecast
(median width 1.6-1.8 label units, about 2.6 when calibrated); seed 0 alone (20 paths) covers
0.41-0.58. Without the nucleus cut (top_p 1.0; Kronos-small, seed 0, 20 paths, 150 evenly spaced
contexts of d4) the interval widens by 15-35 % but covers only 0.65 / 0.63 / 0.41 (0.59 / 0.52 /
0.42 at top_p 0.9 on the same contexts), so the narrowness is Kronos's own, not the cut's.
Reproduce with `scripts/check_kronos_dispersion.py --run-id dev-2026-11-17 --top-p` (local GPU, about
10 min).

Files: [dev-2026-11-17/tables/kronos_dispersion.csv](dev-2026-11-17/tables/kronos_dispersion.csv)

On the test cache `test-2026-11-17` (d1-d4 and t1-t4, 7,903 decisions), the same check
(`scripts/check_kronos_dispersion.py --run-id test-2026-11-17`, without `--top-p`): the 60
pooled paths cover 0.62 / 0.60 / 0.48 (Kronos-small) and 0.63 / 0.63 / 0.45 (Kronos-base) at
1 / 4 / 24 bars, seed 0 alone 0.42-0.59.

Files: [test-2026-11-17/tables/kronos_dispersion.csv](test-2026-11-17/tables/kronos_dispersion.csv)

