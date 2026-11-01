# Alma-Fin results (VAL, debugging only)

Author: Alma-Fin lead (agent), 2026-11-01. Protocol: `docs/fin_prereg.md` (36d2232, committed before any score).
Harness: `tools/fin/run_eval.py`. Raw outputs: `runs/fin/val/` (`metrics.json`, `report.md`, one npz per model and
market). TEST has not been run; no TEST value has been computed or inspected.

VAL = decision days 2025-07-01..2025-10-31 (123 crypto days x 24 coins = 2,952 coin-days; 87 ETF sessions x 13).
VAL is for debugging and for the single confirmation of a frozen candidate. No claim is made from VAL (prereg
section 7). The day-level tests below are the pre-registered ones; with 123 days and rho = 0.52 they can only see
large effects.

## 1. P1: next-day realised-vol direction (crypto, all 24 coins; base rate 0.504, DEV majority 0.453 -> "down")

| model | acc | 95% CI (7-day blocks) | P3 QS (log RV) | MAE median | cov80 |
|---|---|---|---|---|---|
| lgbm_dow | 0.736 | 0.701-0.771 | 0.1590 | 0.390 | 0.884 |
| har_intra_dow | 0.729 | 0.696-0.766 | 0.1647 | 0.402 | 0.891 |
| har_dow | 0.714 | 0.672-0.757 | 0.1706 | 0.418 | 0.884 |
| tirex | 0.704 | 0.657-0.752 | 0.1714 | 0.425 | 0.802 |
| chronos2 | 0.699 | 0.658-0.742 | 0.1699 | 0.422 | 0.786 |
| timestral_5m (CONTAMINATED) | 0.693 | 0.651-0.736 | 0.1732 | 0.433 | 0.798 |
| chronos_bolt_base | 0.690 | 0.641-0.738 | 0.1747 | 0.433 | 0.806 |
| lgbm_asset | 0.685 | 0.645-0.729 | 0.1841 | 0.454 | 0.808 |
| rv_persist | 0.681 | 0.640-0.720 | 0.2214 | 0.548 | 0.814 |
| lgbm | 0.675 | 0.628-0.724 | 0.1802 | 0.441 | 0.850 |
| harx | 0.666 | 0.619-0.715 | 0.1931 | 0.470 | 0.851 |
| har_roll | 0.666 | 0.619-0.716 | 0.1920 | 0.473 | 0.826 |
| har | 0.662 | 0.611-0.711 | 0.1913 | 0.464 | 0.856 |
| kronos_small | 0.574 | 0.519-0.630 | 0.3495 | 0.818 | 0.449 |
| garch | 0.571 | 0.510-0.634 | 0.2389 | 0.605 | 0.787 |
| majority | 0.496 | 0.409-0.587 | - | - | - |

By group (accuracy, training 16 / held-out 8): chronos2 0.703/0.691; chronos_bolt_base 0.691/0.689; garch 0.569/0.575; har 0.667/0.651; har_dow 0.717/0.707; har_intra_dow 0.734/0.721; har_roll 0.672/0.654; harx 0.665/0.670; kronos_small 0.588/0.545; lgbm 0.674/0.677; lgbm_asset 0.696/0.664; lgbm_dow 0.740/0.727; majority 0.492/0.504; rv_persist 0.684/0.674; timestral_5m 0.696/0.686; tirex 0.706/0.701.

Pairwise on VAL (DM-HLN on daily accuracy, two-sided; 7-day block CI of the mean difference):
- P1 lgbm_dow vs har: diff +0.0742, p = 0.000, CI +0.0366..+0.1128
- P3 lgbm_dow vs har: diff -0.0323, p = 0.000, CI -0.0415..-0.0237
- P1 lgbm_dow vs har_roll: diff +0.0698, p = 0.001, CI +0.0332..+0.1098
- P3 lgbm_dow vs har_roll: diff -0.0329, p = 0.000, CI -0.0437..-0.0232
- P1 lgbm_dow vs rv_persist: diff +0.0549, p = 0.002, CI +0.0291..+0.0813
- P3 lgbm_dow vs rv_persist: diff -0.0624, p = 0.000, CI -0.0894..-0.0429
- P1 lgbm_dow vs lgbm: diff +0.0606, p = 0.001, CI +0.0308..+0.0911
- P3 lgbm_dow vs lgbm: diff -0.0212, p = 0.000, CI -0.0281..-0.0146
- P1 lgbm_dow vs lgbm_asset: diff +0.0508, p = 0.002, CI +0.0234..+0.0799
- P3 lgbm_dow vs lgbm_asset: diff -0.0251, p = 0.000, CI -0.0341..-0.0167
- P1 lgbm_dow vs chronos2: diff +0.0366, p = 0.020, CI +0.0088..+0.0647
- P3 lgbm_dow vs chronos2: diff -0.0108, p = 0.013, CI -0.0227..-0.0025
- P1 lgbm_dow vs tirex: diff +0.0315, p = 0.043, CI +0.0010..+0.0613
- P3 lgbm_dow vs tirex: diff -0.0123, p = 0.000, CI -0.0201..-0.0061
- P1 lgbm_dow vs kronos_small: diff +0.1619, p = 0.000, CI +0.1023..+0.2263
- P3 lgbm_dow vs kronos_small: diff -0.1904, p = 0.000, CI -0.2213..-0.1588

Reading. (1) Calendar matters most: the three calendar-aware baselines registered by amendment A5 (lgbm_dow 0.736,
har_intra_dow 0.729, har_dow 0.714) lead, ahead of every zero-shot FM (TiRex 0.704, Chronos-2 0.699, Chronos-Bolt
0.690). Weekend days have a far lower up-rate than weekdays. (2) Models without calendar input cluster at 0.66-0.70;
the HAR family and LightGBM without weekday sit at 0.66-0.69. (3) On P3 (distribution), lgbm_dow and
har_intra_dow are best; Kronos-small (daily K-lines, Parkinson mapping) is poorly calibrated (cov80 0.45). (4) The
pairwise p-values above are VAL debugging numbers, not claims; with the best of these as the bar, a TEST P1 claim
needs roughly +5 pp over lgbm_dow / har_intra_dow (MDE ~5.4 pp, prereg section 12), which is unlikely.
Coverage is 1.0 for every model (A4).

## 2. E1: ETF transfer arm (13 ETFs, Garman-Klass RV; base rate 0.502)

| model | acc | 95% CI (7-day blocks) | P3 QS | MAE median | cov80 |
|---|---|---|---|---|---|
| chronos2 | 0.596 | 0.545-0.645 | 0.2165 | 0.556 | 0.758 |
| garch | 0.592 | 0.546-0.634 | 0.2202 | 0.568 | 0.820 |
| har_roll | 0.592 | 0.544-0.636 | 0.2185 | 0.564 | 0.792 |
| timestral_5m (CONTAMINATED) | 0.587 | 0.534-0.637 | 0.2170 | 0.559 | 0.774 |
| chronos_bolt_base | 0.579 | 0.531-0.625 | 0.2163 | 0.556 | 0.756 |
| tirex | 0.578 | 0.522-0.633 | 0.2178 | 0.562 | 0.785 |
| lgbm_asset | 0.569 | 0.532-0.607 | 0.2196 | 0.569 | 0.767 |
| har | 0.563 | 0.517-0.607 | 0.2435 | 0.626 | 0.705 |
| lgbm | 0.546 | 0.500-0.595 | 0.2387 | 0.608 | 0.666 |
| har_dow | 0.541 | 0.493-0.591 | 0.2656 | 0.675 | 0.631 |
| rv_persist | 0.538 | 0.500-0.577 | 0.2826 | 0.716 | 0.653 |
| kronos_small | 0.535 | 0.489-0.578 | 0.3648 | 0.891 | 0.531 |
| lgbm_dow | 0.529 | 0.477-0.579 | 0.2704 | 0.679 | 0.584 |
| majority | 0.498 | 0.437-0.559 | - | - | - |

Crypto-fitted pooled models (har, lgbm and the weekday variants) transfer worse than per-asset rolling ones on ETFs:
their level, persistence and weekday pattern (crypto weekends) are calibrated on crypto. Chronos-2, rolling HAR and
GARCH are the ETF bar (~0.59). har_intra_dow is not run on ETFs (no hourly data).

## 3. P2: trading baselines, net of costs (Sharpe annualised; 7-day block CI)

Held-out 8 coins (the P2 population):

| model | Sharpe | 95% CI | PSR(0) | ann vol | maxDD | turnover | Sharpe @2x cost |
|---|---|---|---|---|---|---|---|
| tsmom_1 | 1.45 | -1.88..4.97 | 0.804 | 0.632 | 0.281 | 0.974 | 0.89 |
| tsmom_7 | 1.20 | -1.85..4.35 | 0.758 | 0.642 | 0.285 | 0.352 | 1.00 |
| buy_hold | 0.57 | -2.21..3.98 | 0.627 | 0.781 | 0.339 | 0.008 | 0.56 |
| tsmom_30 | -1.34 | -4.39..1.63 | 0.213 | 0.535 | 0.426 | 0.230 | -1.50 |
| voltgt_tsmom | -1.52 | -4.42..1.30 | 0.186 | 0.265 | 0.208 | 0.204 | -1.80 |
| lgbm_policy | -2.55 | -5.59..0.49 | 0.060 | 0.248 | 0.258 | 0.534 | -3.33 |
| meanrev_1 | -2.57 | -5.98..0.64 | 0.063 | 0.635 | 0.521 | 0.974 | -3.12 |

All 24 coins:

| model | Sharpe | 95% CI | PSR(0) | ann vol | maxDD | turnover | Sharpe @2x cost |
|---|---|---|---|---|---|---|---|
| tsmom_1 | 1.60 | -1.50..4.69 | 0.830 | 0.593 | 0.267 | 0.976 | 1.01 |
| tsmom_7 | 1.16 | -1.95..4.49 | 0.746 | 0.570 | 0.260 | 0.367 | 0.93 |
| buy_hold | 1.09 | -1.76..4.55 | 0.730 | 0.760 | 0.237 | 0.008 | 1.08 |
| tsmom_30 | -0.45 | -3.75..2.75 | 0.396 | 0.444 | 0.290 | 0.206 | -0.62 |
| voltgt_tsmom | -0.95 | -3.99..2.17 | 0.289 | 0.215 | 0.142 | 0.183 | -1.25 |
| lgbm_policy | -2.52 | -5.31..0.45 | 0.050 | 0.218 | 0.215 | 0.530 | -3.36 |
| meanrev_1 | -2.76 | -5.71..0.19 | 0.048 | 0.596 | 0.506 | 0.976 | -3.33 |

ETFs:

| model | Sharpe | 95% CI | PSR(0) | ann vol | maxDD | turnover | Sharpe @2x cost |
|---|---|---|---|---|---|---|---|
| buy_hold | 2.17 | -0.24..5.04 | 0.892 | 0.084 | 0.023 | 0.011 | 2.16 |
| lgbm_policy | 1.98 | -0.43..4.84 | 0.871 | 0.070 | 0.019 | 0.122 | 1.89 |
| tsmom_30 | 1.05 | -1.41..3.75 | 0.726 | 0.059 | 0.026 | 0.195 | 0.88 |
| meanrev_1 | 1.03 | -2.05..4.86 | 0.720 | 0.061 | 0.034 | 1.035 | 0.17 |
| voltgt_tsmom | 0.74 | -1.80..3.49 | 0.663 | 0.050 | 0.025 | 0.257 | 0.48 |
| tsmom_7 | -1.98 | -4.75..0.76 | 0.116 | 0.060 | 0.061 | 0.372 | -2.29 |
| tsmom_1 | -2.73 | -6.91..0.56 | 0.070 | 0.061 | 0.071 | 1.035 | -3.58 |

123 days give Sharpe CIs about +-3 wide: nothing here is distinguishable from zero. Buy & hold and short-horizon
TSMOM made money in this window; mean reversion and the
LightGBM return policy lost after costs; vol targeting cut volatility (0.20-0.27 annualised vs 0.76 for buy & hold)
without adding return. P2 is co-primary but under-powered (prereg section 8).

## 4. Status of the competitor list

| competitor | VAL status |
|---|---|
| buy & hold, TSMOM 1/7/30, mean reversion, vol-targeted TSMOM, LightGBM policy | run (crypto + ETF) |
| HAR, HAR-X, rolling HAR, GARCH(1,1), RV persistence, majority | run (HAR-X crypto only) |
| LightGBM pooled / per-asset | run |
| har_dow, lgbm_dow, har_intra_dow (A5) | run (har_intra_dow crypto only); also on DEV folds E1-E3, F1 (`runs/fin/fold_refs.json`) |
| Chronos-2, Chronos-Bolt-base, TiRex | run, zero-shot, offline |
| Timestral-5M | run, CONTAMINATED (no claims) |
| Kronos-small | run on every VAL day (daily K-lines, 10 samples; ~35 min crypto) |
| PPO / DQN (model-free twin) | architect (`tools/fin/agents/`), not yet in the harness run |

## 5. Flags for the auditor

- DEV is not survivorship-free (universe chosen 2025-10-31); TEST is.
- P2 TEST window ends 2026-09-29 (no 2026-10 funding archive; REST endpoint geo-restricted). 2026-10-28 is missing in
  the archive (NaN row).
- Chronos-2's corpus cut-off is unknown and may overlap VAL; Kronos has seen DEV; Timestral is contaminated.
- ETF data come from Yahoo's unofficial endpoint (snapshot pinned).
- The VAL ranking of baselines is not used for any selection by the lead; the architect may read it (VAL is not
  sealed) but every look counts toward K.
