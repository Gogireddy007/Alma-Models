# Alma-Live Gas harness: dev

Score days ['2017-01-03', '2024-06-21', 1879]; decision 15:30 ET (ctx = asof), fill 16:00 ET close. Data pins tag d58ec4ac2752. History is NOT the claim test (the forward paper test is). (Report re-rendered from the run's CSVs after a markdown-dependency failure; numbers unchanged.)

## Primary targets (quantile score = mean pinball over levels 0.05..0.95; lower is better)

| model | ret_n | ret_qs | ret_qs_lo | ret_qs_hi | ret_skill_vs_fc_hist | ret_dqs_p | ret_cov90 | rv5_n | rv5_qs | rv5_qs_lo | rv5_qs_hi | rv5_skill_vs_fc_hist | rv5_dqs_p | rv5_cov90 | stor_n | stor_qs | stor_qs_lo | stor_qs_hi | stor_skill_vs_fc_hist | stor_dqs_p | stor_cov90 | stor_dir_acc | stor_dir_lo | stor_dir_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fc_hist | 1879 | 0.007456 | 0.006969 | 0.00798 | 0 |  | 0.884 | 1879 | 0.2304 | 0.2151 | 0.2462 | 0 |  | 0.8834 | 390 | 10.08 | 8.349 | 11.96 | 0 |  | 0.8744 | 0.509 | 0.4594 | 0.5585 |
| fc_ewma | 1879 | 0.007214 | 0.006758 | 0.007701 | 0.03248 | 0 | 0.8968 | 1879 | 0.1868 | 0.1742 | 0.2001 | 0.1892 | 0 | 0.8946 | 390 | 10.02 | 8.272 | 11.93 | 0.005849 | 0.889 | 0.8769 | 0.6211 | 0.5719 | 0.668 |

## Secondary: trading (UNG, 10 bp/side; Sharpe at 1x/2x/3x costs; DSR k = 8)

| model | n_active | dir_acc | dir_lo | dir_hi | sharpe_1x | sharpe_lo_1x | sharpe_hi_1x | psr_1x | dsr_1x | sharpe_2x | sharpe_3x | mdd_1x | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | 1850.000 | 0.490 | 0.468 | 0.513 | -0.206 | -0.947 | 0.521 | 0.287 | 0.074 | -0.206 | -0.207 | 0.911 | 0.001 |
| short_ung_carry | 1850.000 | 0.510 | 0.487 | 0.532 | 0.206 | -0.521 | 0.946 | 0.713 | 0.373 | 0.205 | 0.205 | 0.885 | 0.001 |
| tsmom_5 | 1840.000 | 0.486 | 0.464 | 0.509 | -0.221 | -0.914 | 0.480 | 0.273 | 0.068 | -0.396 | -0.571 | 0.948 | 0.375 |
| tsmom_20 | 1845.000 | 0.502 | 0.479 | 0.525 | 0.228 | -0.481 | 0.964 | 0.734 | 0.397 | 0.144 | 0.059 | 0.689 | 0.181 |
| tsmom_60 | 1845.000 | 0.496 | 0.473 | 0.519 | -0.055 | -0.742 | 0.629 | 0.440 | 0.150 | -0.109 | -0.162 | 0.895 | 0.114 |
| seasonal | 940.000 | 0.485 | 0.453 | 0.517 | -0.319 | -0.998 | 0.280 | 0.192 | 0.040 | -0.329 | -0.340 | 0.865 | 0.016 |
| weather_rev | 1508.000 | 0.495 | 0.470 | 0.520 | -0.444 | -1.159 | 0.295 | 0.114 | 0.018 | -0.863 | -1.280 | 0.941 | 0.823 |
| storage_surprise | 1779.000 | 0.515 | 0.492 | 0.538 | 0.188 | -0.480 | 0.936 | 0.696 | 0.356 | 0.113 | 0.037 | 0.636 | 0.160 |
