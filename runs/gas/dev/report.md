# Alma-Gas dev report

commit 36d2232f502c9c66166fdd3118ee4147ef395282, scored 2026-10-31T23:40:27-0400, rehearsal=False

## Storage (PRIMARY): sign(dS_W - avg5_W), 313 target weeks 2019-06-28 .. 2025-06-20

### origin thu, Lower 48 (base rate up 0.511; DEV majority up-frac 0.505)

| model | kind | n | acc | Wilson 95% | p binom | p PT | MAE Bcf | QS | cov80 | McNemar p vs st_persistence | vs st_ols_wx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wm_gas | model | 313 | 0.955 | [0.926, 0.973] | 0.000 | 0.000 | 6.6 | 3.11 | 0.98 | 0.000 | 0.000 |
| wm_gas_direct | model | 313 | 0.949 | [0.919, 0.968] | 0.000 | 0.000 | 6.2 | 3.21 | 0.99 | 0.000 | 0.000 |
| st_ols_wx | baseline | 313 | 0.744 | [0.693, 0.790] | 0.000 | 0.000 | 20.4 | 7.87 | 0.80 | 0.003 | - |
| st_ols_wx_nolag | baseline | 313 | 0.722 | [0.670, 0.769] | 0.000 | 0.000 | 21.4 | 8.24 | 0.79 | 0.027 | 0.167 |
| st_persistence | baseline | 313 | 0.633 | [0.578, 0.684] | 0.000 | 0.000 | 35.8 | 14.78 | 0.80 | - | 0.003 |
| st_seasonal_naive | baseline | 313 | 0.527 | [0.472, 0.582] | 0.366 | 0.178 | 37.9 | 15.58 | 0.82 | 0.011 | 0.000 |
| majority | baseline | 313 | 0.511 | [0.456, 0.566] | 0.735 | - | - | - | - | 0.003 | 0.000 |
| st_climatology | baseline | 313 | 0.489 | [0.434, 0.544] | 0.735 | 0.665 | 32.4 | 12.51 | 0.70 | 0.000 | 0.000 |

Regions pooled (secondary; 5 regions x weeks, week-clustered bootstrap):

| model | n labels | acc | 95% CI |
|---|---|---|---|
| wm_gas_direct | 1540 | 0.830 | [0.812, 0.848] |
| wm_gas | 1540 | 0.829 | [0.810, 0.847] |
| st_ols_wx | 1540 | 0.693 | [0.664, 0.720] |
| st_ols_wx_nolag | 1540 | 0.649 | [0.616, 0.680] |
| st_persistence | 1540 | 0.627 | [0.600, 0.657] |
| st_seasonal_naive | 1540 | 0.519 | [0.488, 0.550] |
| majority | 1540 | 0.518 | [0.498, 0.538] |
| st_climatology | 1540 | 0.475 | [0.449, 0.502] |

### origin fri, Lower 48 (base rate up 0.511; DEV majority up-frac 0.505)

| model | kind | n | acc | Wilson 95% | p binom | p PT | MAE Bcf | QS | cov80 | McNemar p vs st_persistence | vs st_ols_wx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wm_gas_direct | model | 313 | 0.952 | [0.922, 0.971] | 0.000 | 0.000 | 6.0 | 3.14 | 0.99 | 0.000 | 0.000 |
| wm_gas | model | 313 | 0.949 | [0.919, 0.968] | 0.000 | 0.000 | 6.4 | 3.07 | 0.98 | 0.000 | 0.000 |
| st_ols_wx | baseline | 313 | 0.757 | [0.707, 0.801] | 0.000 | 0.000 | 20.4 | 7.86 | 0.80 | 0.001 | - |
| st_ols_wx_nolag | baseline | 313 | 0.719 | [0.667, 0.766] | 0.000 | 0.000 | 21.4 | 8.25 | 0.80 | 0.034 | 0.017 |
| st_persistence | baseline | 313 | 0.633 | [0.578, 0.684] | 0.000 | 0.000 | 35.8 | 14.78 | 0.80 | - | 0.001 |
| st_seasonal_naive | baseline | 313 | 0.527 | [0.472, 0.582] | 0.366 | 0.178 | 37.9 | 15.58 | 0.82 | 0.011 | 0.000 |
| majority | baseline | 313 | 0.511 | [0.456, 0.566] | 0.735 | - | - | - | - | 0.003 | 0.000 |
| st_climatology | baseline | 313 | 0.489 | [0.434, 0.544] | 0.735 | 0.665 | 32.4 | 12.51 | 0.70 | 0.000 | 0.000 |

Regions pooled (secondary; 5 regions x weeks, week-clustered bootstrap):

| model | n labels | acc | 95% CI |
|---|---|---|---|
| wm_gas_direct | 1540 | 0.832 | [0.813, 0.849] |
| wm_gas | 1540 | 0.828 | [0.809, 0.847] |
| st_ols_wx | 1540 | 0.695 | [0.666, 0.722] |
| st_ols_wx_nolag | 1540 | 0.662 | [0.630, 0.692] |
| st_persistence | 1540 | 0.627 | [0.600, 0.657] |
| st_seasonal_naive | 1540 | 0.519 | [0.488, 0.550] |
| majority | 1540 | 0.518 | [0.498, 0.538] |
| st_climatology | 1540 | 0.475 | [0.449, 0.502] |

