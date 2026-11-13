# Deployment audit: 58/58 checks passed

| Section | Check | Status | Detail |
|---|---|---|---|
| A | all winning_models checkpoints match SHA256SUMS and FREEZE.json | **PASS** | 18 files |
| A | all FREEZE.json files (code + checkpoints) unchanged | **PASS** | 29 files, changed: [] |
| A | git: no edits to frozen model code / checkpoints / forward freeze | **PASS** | [] |
| A | oil wrapper pins == oil SHA256SUMS | **PASS** | 6 pins |
| B | deploy scripts contain no optimizer/backward/train() call | **PASS** | [] |
| B | oil/results.json: every registered model ran (no competitor silently skipped) | **PASS** | 9 models; skipped: {} |
| B | oil/results.json: torch parameters unchanged after deployment | **PASS** | 7 modules hashed |
| B | power/results.json: every registered model ran (no competitor silently skipped) | **PASS** | 13 models; skipped: {} |
| B | power/results.json: torch parameters unchanged after deployment | **PASS** | 10 modules hashed |
| B | gas/results.json: every registered model ran (no competitor silently skipped) | **PASS** | 13 models; skipped: {} |
| B | gas/results.json: torch parameters unchanged after deployment | **PASS** | 21 modules hashed |
| B | fin/results_corrected.json: every registered model ran (no competitor silently skipped) | **PASS** | 15 models; skipped: {} |
| B | fin/results_corrected.json: torch parameters unchanged after deployment | **PASS** | 10 modules hashed |
| B | fin/results_corrected.json: zero online updates | **PASS** | {'wm_fin': 0, 'wm_direct': 0, 'wm_fin_blend': 0, 'har_intra_dow': 0, 'har_dow': 0, 'lgbm_dow': 0, 'har': 0, 'har_roll': 0, 'lgbm': 0, 'lgbm_asset': 0, 'rv_persist': 0, 'garch': 0, 'chronos2': 0, 'chronos_bolt_base': 0, 'tirex': 0} |
| G | oil preds sha256 matches results | **PASS** |  |
| C | oil: origin dates re-derived from raw FRED csv | **PASS** |  |
| C | oil oct_2026 h1: n targets from raw csv | **PASS** | n=22 |
| C | oil oct_2026 h5: n targets from raw csv | **PASS** | n=22 |
| C | oil oct_2026 h20: n targets from raw csv | **PASS** | n=22 |
| C | oil nov_2026 h1: n targets from raw csv | **PASS** | n=8 |
| C | oil nov_2026 h5: n targets from raw csv | **PASS** | n=8 |
| C | oil nov_2026 h20: n targets from raw csv | **PASS** | n=8 |
| D | oil: every QS recomputed from saved forecasts | **PASS** | 54/54 match |
| F | oil: every significance verdict rests on >= 10 independent units | **PASS** | 48 verdicts, 0 significant, bad=[] |
| E | oil: future-scramble leak test (wm_sv, wm_sv_direct) | **PASS** | 0/34 forecasts changed |
| R | oil: saved wm_sv forecasts = fresh re-run of the frozen model | **PASS** | 12 origins, max /diff/ 0.0e+00 |
| R | oil: saved wm_sv_direct forecasts = fresh re-run of the frozen model | **PASS** | 12 origins, max /diff/ 0.0e+00 |
| G | power preds sha256 matches results | **PASS** |  |
| C | power: realised hourly prices re-derived from raw Energy-Charts JSON | **PASS** | 1056 hours, max diff 5.68e-14 |
| C | power oct_2026: n delivery days | **PASS** | n=31 |
| C | power nov_2026: n delivery days | **PASS** | n=13 |
| D | power: every QS recomputed from saved forecasts | **PASS** | 52/52 match |
| F | power: every significance verdict rests on >= 10 independent units | **PASS** | 96 verdicts, 0 significant, bad=[] |
| E | power: future-scramble leak test (prices >= D, fundamentals >= D-1) | **PASS** | 0/16 forecasts changed |
| R | power: saved wm_power forecasts = fresh re-run of the frozen model | **PASS** | 11 delivery days, max /diff/ 6.1e-05 |
| R | power: saved wm_power_direct forecasts = fresh re-run of the frozen model | **PASS** | 11 delivery days, max /diff/ 3.1e-05 |
| G | gas preds sha256 matches results | **PASS** |  |
| C | gas: TEST forecasts reproduce the audit cache (inputs identical) | **PASS** | max diff 5.1e-05 Bcf over 10 series x 52 weeks |
| C | gas oct_2026: storage changes re-derived from raw EIA ngshistory.xls | **PASS** | [64.0, 85.0, 67.0, 77.0, 80.0] vs [64.0, 85.0, 67.0, 77.0, 80.0] |
| C | gas nov_2026: storage changes re-derived from raw EIA ngshistory.xls | **PASS** | [77.0] vs [77.0] |
| D | gas: every MAE and QS recomputed from saved forecasts | **PASS** | 208/208 match |
| F | gas: every significance verdict rests on >= 10 independent units | **PASS** | 192 verdicts, 0 significant, bad=[] |
| R | gas: saved wm_gas (oct_2026, thu) forecasts = fresh re-run of the frozen model | **PASS** | 5 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas_ens (oct_2026, thu) forecasts = fresh re-run of the frozen model | **PASS** | 5 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas (oct_2026, fri) forecasts = fresh re-run of the frozen model | **PASS** | 5 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas_ens (oct_2026, fri) forecasts = fresh re-run of the frozen model | **PASS** | 5 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas (nov_2026, thu) forecasts = fresh re-run of the frozen model | **PASS** | 1 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas_ens (nov_2026, thu) forecasts = fresh re-run of the frozen model | **PASS** | 1 weeks, max /diff/ 2.0e-07 |
| R | gas: saved wm_gas (nov_2026, fri) forecasts = fresh re-run of the frozen model | **PASS** | 1 weeks, max /diff/ 0.0e+00 |
| R | gas: saved wm_gas_ens (nov_2026, fri) forecasts = fresh re-run of the frozen model | **PASS** | 1 weeks, max /diff/ 1.8e-07 |
| E | gas: future-scramble leak test (storage >= target week, weather after origin) | **PASS** | 0/36 forecasts changed |
| G | fin corrected preds sha256 matches results | **PASS** |  |
| C | fin corrected: log-RV labels re-derived from raw Binance zips (60 random coin-days) | **PASS** | 60 labelled, max /diff/ 8.9e-16 |
| D | fin corrected: every QS and direction accuracy recomputed | **PASS** | 120/120 match |
| F | fin corrected: every significance verdict rests on >= 10 independent units | **PASS** | 112 verdicts, 0 significant, bad=[] |
| R | fin: saved wm_fin forecasts = fresh re-run of the frozen model | **PASS** | 8 days x 24 coins, max /diff/ 0.0e+00 |
| R | fin: saved wm_direct forecasts = fresh re-run of the frozen model | **PASS** | 8 days x 24 coins, max /diff/ 0.0e+00 |
| E | fin: future-scramble leak test (daily rows >= D, hourly rows >= D 00:00) | **PASS** | 0/6 forecasts changed |
