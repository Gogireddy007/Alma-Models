# Gas candidate freeze (written 2026-11-01T16:34Z, before any TEST score of it)
- Members: frozen wm_gas (seeds 0-2) + frozen wm_gas_direct (seeds 0-2) + refits of the same two recipes, seeds 3-6
  (tools/gas/search/refit.py, train weeks released <= 2025-06-30). 14 members, equal weight (quantile average).
- Blend: 0.80 x NN pool + 0.20 x st_ols_wx quantiles (weight = DEV full-backtest MAE optimum; LOFO choices 0.1-0.3).
- DEV backtest (6 rolling folds, 2019-07..2025-06): MAE thu/fri 15.39/15.27 vs frozen wm_gas recipe 16.59/16.53.
- Evaluated once on TEST with tools in audit/gas_final.py. NOTE: the TEST year is no longer pristine
  (the repo's lead scored it, and this audit re-scored the frozen models on it), so any TEST result is a second look.

## Amendment 1: interval recalibration (written 2026-11-01T17:00Z, before any TEST score of it)
- Cause: 80% interval width is ~constant (~52 Bcf) whatever the recent error level (DEV folds cov80 0.73-0.90).
- Rule: q' = median + s_t (q - median), s_t = clip((mean |median error| over the last 8 released weeks / ref)^0.5,
  0.4, 2.0); s_t = 1 until 4 past weeks exist; ref = DEV out-of-fold mean |error| (thu 15.39, fri 15.27).
- Chosen leave-one-fold-out on DEV (every fold picked adaptive/base 1.0/a 0.5; window 4-8); DEV QS thu 5.970 -> 5.919,
  cov80 0.827 -> 0.802. A wider grid (windows 4/6, exponents 0.25/0.75) did not improve LOFO QS.
- On TEST the trailing errors start from the VAL weeks (2025-07..10, out-of-sample for models fit <= 2025-06-30).
