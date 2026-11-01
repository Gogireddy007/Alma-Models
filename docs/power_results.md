# Alma-Power results

Protocol: `docs/power_prereg.md` (commit 5730a54; amendment A1 in 4085a95). Harness: `tools/power/`.

## Data summary

- Source: Energy-Charts API, downloaded 2026-10-31 (73 raw yearly JSON files; urls, times, sha256 in
  `data/power/manifest.json`). Panel `data/power/hourly.csv`, sha256 `79a2c96e…6dea` (pinned in `run_eval.DATA_SHA256`).
- 4,322 delivery days, 2015-01-01 .. 2026-10-31. Zone DE-AT-LU until 2018-09-30, DE-LU after. Hourly day-ahead
  products until 2025-09-30, 15-min from 2025-10-01 (hourly = mean of 4 quarter-hours).
- Missing: only the 12 spring-forward slots (structural), 2015-01-01 00:00 local (before the first API value), and
  the last UTC hour of 2026-10-31 for the covariates (never used). No forward fill.
- Yearly mean price (EUR/MWh): 2015 31.6, 2016 29.0, 2017 34.2, 2018 44.5, 2019 37.7, 2020 30.5, 2021 96.8,
  2022 235.5, 2023 95.2, 2024 78.5, 2025 89.3, 2026 (Jan-Oct) 114.0. Share of negative hours: 1-2% before 2019,
  5.2% in 2024, 6.6% in 2025.
- Splits: DEV backtest 365 days (2024-07-01..2025-06-30), VAL 123 days, TEST 365 days (2025-11-01..2026-10-31).
- Information-set tests (`tools/power/test_infoset.py`): pass.

## VAL (2025-07-01 .. 2025-10-31, 123 days): baselines, competitors, foundation models, Alma zero-shot

Debugging only, not a claim. These are the strict-mode rehearsal numbers (`runs/power/rehearsal_val/`, RUN.json
status complete). They are identical to the separate VAL run: LEAR differs by less than 1e-10, every other model
by 0. DEV majority up-rates: P1 0.457, P2 0.483, P3 0.456. VAL base rate for P1: 0.472.

| model | P1 acc [95% CI] | P2 acc [day-block CI] | P3 acc | QS skill vs naive_epf | MAE | rMAE | cov80 | cov95 |
|---|---|---|---|---|---|---|---|---|
| majority | 0.528 [0.441, 0.614] | 0.522 [0.487, 0.554] | 0.528 | - | - | - | - | - |
| dow_majority | 0.667 [0.579, 0.744] | 0.608 [0.563, 0.648] | 0.667 | - | - | - | - | - |
| naive_d | 0.528 [0.441, 0.614] | 0.522 [0.487, 0.554] | 0.528 | 0.086 | 24.83 | 0.925 | 0.833 | 0.954 |
| naive_w | 0.724 [0.639, 0.795] | 0.651 [0.611, 0.692] | 0.740 | -0.142 | 30.77 | 1.146 | 0.822 | 0.945 |
| naive_epf | 0.634 [0.546, 0.714] | 0.566 [0.529, 0.603] | 0.634 | 0.000 | 26.84 | 1.000 | 0.828 | 0.946 |
| arx | 0.756 [0.673, 0.823] | 0.741 [0.712, 0.769] | 0.683 | 0.370 | 18.07 | 0.673 | 0.826 | 0.946 |
| lear | 0.789 [0.708, 0.851] | 0.748 [0.720, 0.774] | 0.740 | 0.398 | 17.24 | 0.642 | 0.783 | 0.929 |
| lear_nocov | 0.780 [0.699, 0.845] | 0.757 [0.726, 0.786] | 0.764 | 0.402 | 17.09 | 0.637 | 0.797 | 0.938 |
| chronos2 | 0.780 [0.699, 0.845] | 0.759 [0.735, 0.783] | 0.780 | 0.420 | 16.63 | 0.620 | 0.765 | 0.954 |
| chronos2_cov | 0.732 [0.647, 0.802] | 0.759 [0.734, 0.783] | 0.724 | 0.403 | 17.20 | 0.641 | 0.790 | 0.959 |
| chronos_bolt_base | 0.724 [0.639, 0.795] | 0.722 [0.693, 0.749] | 0.724 | 0.354 | 18.55 | 0.691 | 0.809 | - |
| tirex | 0.732 [0.647, 0.802] | 0.733 [0.701, 0.762] | 0.756 | 0.384 | 17.73 | 0.660 | 0.773 | - |
| kronos_small (every 5th day, n=25) | 0.640 [0.445, 0.798] | 0.623 [0.557, 0.697] | 0.640 | -0.332 | 35.94 | 1.353 | 0.482 | 0.617 |
| timestral_5m (CONTAMINATED) | 0.772 [0.691, 0.838] | 0.743 [0.717, 0.768] | 0.732 | 0.389 | 17.75 | 0.661 | 0.784 | - |
| alma_s0 | 0.650 [0.563, 0.729] | 0.663 [0.639, 0.688] | 0.642 | 0.220 | 21.98 | 0.819 | 0.748 | - |
| alma_direct_s0 | 0.667 [0.579, 0.744] | 0.631 [0.601, 0.663] | 0.626 | 0.200 | 22.68 | 0.845 | 0.740 | - |

Reading (descriptive):
- The bars to beat are high. Weekday seasonality alone gives P1 0.667 (`dow_majority`) and 0.724 (`naive_w`).
  LEAR and Chronos-2 reach about 0.78-0.79 on P1 and 0.75-0.76 on P2.
- LEAR vs naive_w on P1: +0.065, CI [-0.024, 0.147], not significant on 123 days. On P2, LEAR vs naive_w is +0.097,
  CI [0.058, 0.136]. Chronos-2 vs LEAR: no significant difference on any target (P4 QS diff -0.25 EUR/MWh,
  CI [-0.53, 0.06]).
- Lagged fundamentals (from D-2 and earlier) add nothing measurable: lear_nocov ≈ lear, and chronos2_cov is slightly
  worse than chronos2 (QS +0.19, p = 0.05). This is expected, because the information set excludes day-D forecasts
  of wind, solar and load, which are the main drivers in the EPF literature.
- Alma's general zero-shot SSM is clearly below the EPF baselines (rMAE 0.82, P1 0.65). The rollout and the direct
  twin are not significantly different.
- LEAR's in-sample residual intervals undercover a little (80% interval: 0.78). Chronos-2's 80% interval is at
  0.77. Kronos is poor on hourly prices (rMAE 1.35, 80% coverage 0.48).
- The P1 and P3 accuracies of majority / dow_majority / naive_epf coincide. The labels agree on 80% of VAL days
  (DEV: 72%) and have the same VAL base rate (58 of 123 up). This was checked and is not a bug.

TEST: not run. It requires `runs/power/FROZEN.json` (architect), a committed clean `tools/power/**`, and the
auditor's GO relayed by the manager.
