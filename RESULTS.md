# Results

Every number here is produced by a script from files under `results/`; the source file and
row are given with each entry. Tables: `results/summary/violations.csv` (mean and std over
seeds) and `results/summary/per_seed_table.csv` (per seed), both written by
`scripts/aggregate.py` from `results/<run>/eval.csv` (`scripts/evaluate.py`, 20 episodes per
SOC, reset seeds 1000-1019). The full tables are written in Step 11 (PROMPTS.md).

## Observations

1. **Falls of baseline_s0 at SOC 0.2, with and without the filter at test time.**
   With the filter added at test time, baseline_s0 fell in 13 of 20 episodes at SOC 0.2;
   without it, in 3 of 20.
   Source: `results/summary/per_seed_table.csv`, rows `baseline+filter@test, soc 0.2, seed 0`
   (n_fell 13, n_episodes 20) and `baseline, soc 0.2, seed 0` (n_fell 3, n_episodes 20).
   Recorded as an observation, without explanation (user instruction, 2026-11-19).
