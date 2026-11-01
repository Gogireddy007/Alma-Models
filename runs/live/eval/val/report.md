# Alma-Live Gas harness: val

Score days ['2024-07-01', '2025-10-24', 332]; decision 15:30 ET (ctx = asof), fill 16:00 ET close. Data pins tag d58ec4ac2752. History is NOT the claim test (the forward paper test is).

## Primary targets (quantile score = mean pinball over levels 0.05..0.95; lower is better)

ret = UNG log return close(d)->close(d+1); rv5 = log sum of squared UNG returns d+1..d+5; stor = storage deviation vs naive 5-yr expectation (Bcf), origin = last decision before the first print.

| model | ret_n | ret_qs | ret_qs_lo | ret_qs_hi | ret_skill_vs_fc_hist | ret_dqs_p | ret_cov90 | rv5_n | rv5_qs | rv5_qs_lo | rv5_qs_hi | rv5_skill_vs_fc_hist | rv5_dqs_p | rv5_cov90 | stor_n | stor_qs | stor_qs_lo | stor_qs_hi | stor_skill_vs_fc_hist | stor_dqs_p | stor_cov90 | stor_dir_acc | stor_dir_lo | stor_dir_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fc_hist | 332 | 0.008159 | 0.007447 | 0.009001 | 0 |  | 0.9036 | 332 | 0.158 | 0.1384 | 0.1825 | 0 |  | 0.8976 | 69 | 9.468 | 6.652 | 13.06 | 0 |  | 0.942 | 0.4559 | 0.343 | 0.5734 |
| fc_ewma | 332 | 0.00802 | 0.007313 | 0.008824 | 0.01712 | 0.025 | 0.9066 | 332 | 0.1495 | 0.133 | 0.1694 | 0.05415 | 0.294 | 0.9096 | 69 | 9.685 | 6.027 | 14.39 | -0.02296 | 0.805 | 0.9275 | 0.6912 | 0.5736 | 0.7883 |

## Secondary: trading (UNG, 10 bp/side; Sharpe at 1x/2x/3x costs; DSR k = 8)

| model | n_active | dir_acc | dir_lo | dir_hi | sharpe_1x | sharpe_lo_1x | sharpe_hi_1x | psr_1x | dsr_1x | sharpe_2x | sharpe_3x | mdd_1x | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | 330.000 | 0.491 | 0.437 | 0.545 | -0.055 | -1.664 | 1.472 | 0.475 | 0.158 | -0.056 | -0.057 | 0.514 | 0.003 |
| short_ung_carry | 330.000 | 0.509 | 0.455 | 0.563 | 0.052 | -1.472 | 1.662 | 0.524 | 0.189 | 0.051 | 0.049 | 0.573 | 0.003 |
| tsmom_5 | 330.000 | 0.491 | 0.437 | 0.545 | -0.985 | -2.445 | 0.331 | 0.130 | 0.019 | -1.134 | -1.283 | 0.675 | 0.334 |
| tsmom_20 | 329.000 | 0.505 | 0.451 | 0.558 | -0.304 | -1.781 | 1.182 | 0.364 | 0.099 | -0.417 | -0.529 | 0.523 | 0.256 |
| tsmom_60 | 329.000 | 0.495 | 0.442 | 0.549 | -0.882 | -2.374 | 0.677 | 0.155 | 0.025 | -0.962 | -1.042 | 0.732 | 0.178 |
| seasonal | 163.000 | 0.546 | 0.469 | 0.621 | 0.540 | -0.892 | 1.964 | 0.732 | 0.374 | 0.530 | 0.520 | 0.252 | 0.015 |
| weather_rev | 252.000 | 0.528 | 0.466 | 0.589 | 0.077 | -1.636 | 1.681 | 0.535 | 0.197 | -0.343 | -0.763 | 0.589 | 0.870 |
| storage_surprise | 315.000 | 0.498 | 0.444 | 0.553 | 0.480 | -1.316 | 2.234 | 0.710 | 0.347 | 0.419 | 0.358 | 0.398 | 0.136 |