# Alma-Gas results: data summary and VAL (debug) tables

Alma-Gas lead, 2026-10-31. Protocol: `docs/gas_prereg.md` (A1 data pins, A2 DEV holiday-week disclosure). Leakage:
`docs/gas_leakage.md`. Full tables: `runs/gas/val/report.md`, `runs/gas/val/metrics.json`; DEV storage baselines
`runs/gas/dev/report.md`. **VAL is for debugging only.** It has 17 scored storage weeks, all in summer and autumn
2025, and 82-87 price origins. Its CIs are wide, and no VAL number supports a claim (prereg sections 3 and 6).
**TEST has not been run, and no TEST value has been inspected.**

## 1. Data (snapshot 2026-11-01 01:16-01:42 UTC, pinned in `run_eval.DATA_PINS`)

- Henry Hub spot (EIA RNGWHHD): 7,485 days, 1997-01-07 .. 2026-10-27. Split: DEV 7,155, VAL 87, TEST 243 rows.
  FRED DHHNGSP matches it on all 5,474 common days <= 2025-10-31; FRED has about 1,900 blanks before 2007. No
  non-positive price through VAL. EIA NYMEX futures stopped in 2024-04, so they are not used.
- EIA weekly storage, L48 + 5 regions: 878 weeks, 2010-01-01 .. 2026-10-23. First print is used for 565 weeks
  (from 2015-06-19). Per-week release dates come from the EIA schedule; the conservative rule is applied to pre-2025
  holiday weeks. Storage weeks with avg5: 617. VAL has 18 target weeks (17 scored, 1 exact tie); TEST has 52.
- CPC observed degree days (gas-weighted HDD, population CDD; CONUS) 1997-01-01 .. 2026-10-29. CPC archived 7-day
  forecasts are on disk for the 2,333 issue days the storage origins can use; 95% of issues have a same-day
  Last-Modified.
- Information-set tests pass (`runs/gas/test_infoset.txt`). Alma TRAIN scan: zero hits for L48 storage
  (`runs/gas/leakage_scan.txt`).

## 2. Storage deviation direction (PRIMARY G1/G2), Lower 48, VAL (17 weeks, base rate up 0.529)

Origin `thu`: close of the previous release day, about 6.8 days before the target release; CPC forecasts are used
for the Thursday and Friday of the target week.

| model | kind | acc | Wilson 95% | p binom | MAE Bcf | QS | cov80 |
|---|---|---|---|---|---|---|---|
| tirex | FM | 0.824 | [0.590, 0.938] | 0.013 | 14.5 | 5.67 | 1.00 |
| timestral_5m | FM, CONTAMINATED | 0.765 | [0.527, 0.904] | 0.049 | 14.2 | 5.63 | 0.94 |
| chronos_bolt_base | FM | 0.706 | [0.469, 0.867] | 0.143 | 14.4 | 5.70 | 0.89 |
| chronos2 | FM | 0.647 | [0.413, 0.827] | 0.332 | 15.1 | 5.88 | 0.72 |
| alma_s0 | Alma (rollout) | 0.647 | [0.413, 0.827] | 0.332 | 19.3 | 7.81 | 1.00 |
| st_ols_wx_nolag | baseline | 0.647 | [0.413, 0.827] | 0.332 | 16.6 | 6.26 | 0.89 |
| st_ols_wx | baseline (the bar) | 0.588 | [0.360, 0.784] | 0.629 | 16.2 | 6.05 | 0.89 |
| st_climatology | baseline | 0.588 | [0.360, 0.784] | 0.629 | 14.7 | 6.13 | 0.72 |
| chronos2_cov | FM | 0.588 | [0.360, 0.784] | 0.629 | 19.1 | 7.45 | 0.61 |
| alma_direct_s0 | Alma (direct twin) | 0.588 | [0.360, 0.784] | 0.629 | 20.6 | 8.01 | 0.89 |
| st_persistence / st_seasonal_naive / st_arima_dev / majority | baselines | 0.529 | [0.310, 0.738] | 1.000 | 24.8 / 21.0 / 16.6 / - | 9.5 / 8.9 / 7.6 / - | |

Origin `fri` (the nowcast): the univariate models are identical to `thu`, because both origins see the same storage
history. st_ols_wx is 0.588 (MAE 16.4) and st_ols_wx_nolag is 0.588.

No pairwise test is significant after correction. On McNemar, tirex vs st_persistence gives p = 0.125 and tirex vs
st_ols_wx gives p = 0.289. On MAE (DM-HLN), tirex and chronos_bolt_base beat st_persistence (p = 0.003 / 0.006,
unadjusted) but not st_ols_wx (p ~ 0.5). Alma's WM test on VAL, alma_s0 vs alma_direct_s0 at thu: MAE -1.2 Bcf
(p = 0.29), QS -0.20 (p = 0.63), accuracy +1 week (McNemar p = 1.0). That is no significant difference.

Regions pooled (G3, 5 x 17-18 labels, thu): timestral 0.807, tirex 0.761, chronos2 0.705, st_ols_wx 0.670,
alma_direct_s0 0.648, alma_s0 0.614, st_persistence 0.602, majority 0.489.

DEV reference for the storage baselines (313 weeks released 2019-07 .. 2025-06; A2 caveat: 60 of them are
holiday-week nowcasts):

| origin | st_ols_wx | st_ols_wx_nolag | st_persistence | st_seasonal_naive | st_climatology | majority |
|---|---|---|---|---|---|---|
| thu | 0.744 | 0.722 | 0.633 | 0.527 | 0.489 | 0.511 |
| fri | 0.757 | 0.719 | 0.633 | 0.527 | 0.489 | 0.511 |

On DEV, st_ols_wx vs st_persistence gives McNemar p = 0.003 at thu. This reproduces the research DEV check (0.76).
On VAL, the weather OLS falls to 0.59. That is consistent with the research finding that summer is the hard season
(DEV summer 0.67) on 17 weeks; the CI includes the DEV value.

## 3. Henry Hub spot (negative control P1 and secondary P2/P3), VAL

- P1 direction, h = 1 (82 pairs, base rate 0.549). Every model is within 0.44-0.56; the best is timestral 0.561
  (contaminated). Nothing differs from 0.5 or from majority. h = 5 (17 non-overlapping pairs): best auto_arima 0.651
  on all origins (non-overlap 0.588). h = 20 rests on 4 non-overlapping windows. All of this is noise, as
  pre-registered.
- P2, 5-day vol direction (83 origins, base 0.494). The FMs and Alma reach 0.69-0.75 and GARCH 0.663 (9
  non-overlapping windows). Descriptive only.
- P3, log-price QS skill vs rw_gauss at h = 1 / 5 / 20: tirex 0.27 / 0.31 / 0.52, px_fund_ols 0.29 / 0.31 / 0.55,
  chronos2_cov 0.24 / 0.31 / 0.57, alma_s0 0.18 / 0.30 / 0.52. rw_gauss intervals are too wide in calm summer 2025
  (sigma from 250 days spanning the winter, cov80 = 1.00). auto_ets is unstable at h = 20.

## 4. Notes for the auditor and the architect

- Univariate FMs on the weekly dS series look strong on VAL, roughly at the level of the weather OLS. This is 17
  summer weeks; the TEST winter, when weather drives storage, is the real test. The bar for the world model on TEST
  is st_ols_wx (G1 McNemar) and the best FM/OLS on G2 MAE.
- Kronos runs on every 5th origin (18 at h = 1). Its h = 20 block-bootstrap CI is degenerate because n = 14 is less
  than the block length.
- The strict TEST procedure was rehearsed on VAL into `runs/gas/rehearsal_val/` (see the RUN.json there). The TEST
  command refuses without `--i-have-auditor-go` and without a committed FROZEN.json.
- Rehearsal: `run_eval.py --split val --rehearse` (strict path, all 13 storage and 12 price pre-registered models,
  preflight, data pins, RUN.json, hashes.json) finished with status complete. Every metric in
  `runs/gas/rehearsal_val/metrics.json` equals `runs/gas/val/metrics.json` (0 differences), so the run is
  deterministic and reproducible. Its RUN.json shows `tools_gas_dirty` caused by the architect's uncommitted
  `tools/gas/search/*`. On TEST this blocks the run until those files are committed, which is intended.
