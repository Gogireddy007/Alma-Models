# wm_gas_ens: US gas storage ensemble (audit follow-up, 2026-11-01)

## What it is
- **Members:** the frozen `wm_gas` (rollout, seeds 0-2) and `wm_gas_direct` (no rollout, seeds 0-2), plus refits of
  the same two recipes with seeds 3-6 (`checkpoints/x_wm_s3456.pt`, `checkpoints/x_dir_s3456.pt`). 14 members, all
  trained by `tools/gas/search/refit.py` on weeks released <= 2025-06-30.
- **Blend:** 0.8 x member quantile average + 0.2 x `st_ols_wx` (the pre-registered OLS baseline).
- **Intervals (L48):** width scaled by the model's own recent accuracy:
  `s = clip((mean |error| of the last 8 released weeks / ref)^0.5, 0.4, 2.0)`, ref thu 15.39 / fri 15.27 Bcf.
  Regions get the blended quantiles without this step.
- **Code:** `tools/gas/wm_gas_ens.py` (registers `wm_gas_ens`; checks every checkpoint's sha256 at load).

## How the choices were made (no TEST data used)
| Choice | Evidence (DEV rolling backtest, 6 yearly folds, release 2019-07..2025-06, each fold trained before it) |
|---|---|
| Pool rollout + direct seeds | 3 seeds MAE 16.59 → 6 seeds 16.10 → 14 seeds 15.73 (thu) |
| 0.2 OLS blend | leave-one-fold-out weights 0.1-0.3 in every fold; full-DEV optimum 0.20; MAE 15.39 |
| Adaptive intervals | every fold picked the same rule (window 4-8, exponent 0.5); DEV QS 5.97 → 5.92, cov80 0.83 → 0.80 |
The freeze record with timestamps is `audit/GAS_FREEZE.md` (outside this repo).

## TEST (52 weeks, 2025-10-31 .. 2026-10-23, Lower 48)
| model | MAE thu / fri (Bcf) | QS thu / fri | cov80 thu / fri | direction acc thu / fri |
|---|---|---|---|---|
| wm_gas (frozen) | 11.53 / 11.56 | 4.59 / 4.56 | 0.87 / 0.89 | 0.788 / 0.808 |
| st_ols_wx | 14.35 / 14.46 | 6.14 / 6.09 | 0.83 / 0.85 | 0.808 / 0.808 |
| **wm_gas_ens** | **10.63 / 10.46** | **4.14 / 4.07** | 0.89 / 0.90 | 0.808 / 0.788 |

- vs `st_ols_wx`: MAE −3.7 / −4.0 Bcf (bootstrap 95% CI excludes 0); QS −2.0 (CI excludes 0).
- vs frozen `wm_gas`: MAE −0.9 / −1.1 (p 0.13 / 0.08, **not significant** on 52 weeks).
- Adaptive intervals vs fixed intervals: QS −0.16 (CI [−0.25, −0.07] / [−0.24, −0.08]).
- Leak test (`audit/gas_ens_leak.py`): scrambling every storage number from the target week on, and all
  weather after the origin, in the global data store changes 0 of 22 forecasts.

## Caveats
- **TEST is a second look:** the year was scored by the original project and re-scored in the audit before this
  model existed. Only forward data after 2026-11-01 is a clean test.
- Intervals still over-cover a little on TEST (0.89-0.90 for a nominal 80%).
- No direction gain: accuracy equals the OLS baseline.
- The "world model" (rollout) members add nothing over the direct members; the gain is from ensembling and blending.
