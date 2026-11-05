# Deployment audit: 40/40 checks passed

| Section | Check | Status | Detail |
|---|---|---|---|
| A | all winning_models checkpoints match SHA256SUMS and FREEZE.json | **PASS** | 18 files |
| A | all FREEZE.json files (code + checkpoints) unchanged | **PASS** | 29 files, changed: [] |
| A | git: no edits to frozen model code / checkpoints / forward freeze | **PASS** | [] |
| A | oil wrapper pins == oil SHA256SUMS | **PASS** | 6 pins |
| B | deploy scripts contain no optimizer/backward/train() call | **PASS** | [] |
| B | oil/results.json: torch parameters unchanged after deployment | **PASS** | 7 modules hashed |
| B | power/results.json: torch parameters unchanged after deployment | **PASS** | 10 modules hashed |
| B | gas/results.json: torch parameters unchanged after deployment | **PASS** | 21 modules hashed |
| B | fin/results_corrected.json: torch parameters unchanged after deployment | **PASS** | 10 modules hashed |
| B | fin/results_corrected.json: zero online updates | **PASS** | {'wm_fin': 0, 'wm_direct': 0, 'wm_fin_blend': 0, 'har_intra_dow': 0, 'har_dow': 0, 'lgbm_dow': 0, 'har': 0, 'har_roll': 0, 'lgbm': 0, 'lgbm_asset': 0, 'rv_persist': 0, 'garch': 0, 'chronos2': 0, 'chronos_bolt_base': 0, 'tirex': 0} |
| G | oil preds sha256 matches results | **PASS** |  |
| C | oil: origin dates re-derived from raw FRED csv | **PASS** |  |
| C | oil oct_2026 h1: n targets from raw csv | **PASS** | n=22 |
| C | oil oct_2026 h5: n targets from raw csv | **PASS** | n=22 |
| C | oil oct_2026 h20: n targets from raw csv | **PASS** | n=22 |
| C | oil nov_2026 h1: n targets from raw csv | **PASS** | n=2 |
| C | oil nov_2026 h5: n targets from raw csv | **PASS** | n=2 |
| C | oil nov_2026 h20: n targets from raw csv | **PASS** | n=2 |
| D | oil: every QS recomputed from saved forecasts | **PASS** | 54/54 match |
| F | oil: every significance verdict rests on >= 10 independent units | **PASS** | 48 verdicts, 0 significant, bad=[] |
| E | oil: future-scramble leak test (wm_sv, wm_sv_direct) | **PASS** | 0/30 forecasts changed |
| G | power preds sha256 matches results | **PASS** |  |
| C | power: realised hourly prices re-derived from raw Energy-Charts JSON | **PASS** | 864 hours, max diff 5.68e-14 |
| C | power oct_2026: n delivery days | **PASS** | n=31 |
| C | power nov_2026: n delivery days | **PASS** | n=5 |
| D | power: every QS recomputed from saved forecasts | **PASS** | 52/52 match |
| F | power: every significance verdict rests on >= 10 independent units | **PASS** | 96 verdicts, 0 significant, bad=[] |
| E | power: future-scramble leak test (prices >= D, fundamentals >= D-1) | **PASS** | 0/12 forecasts changed |
| G | gas preds sha256 matches results | **PASS** |  |
| C | gas: TEST forecasts reproduce the audit cache (inputs identical) | **PASS** | max diff 5.1e-05 Bcf over 10 series x 52 weeks |
| C | gas oct_2026: storage changes re-derived from raw EIA ngshistory.xls | **PASS** | [64.0, 85.0, 67.0, 77.0, 80.0] vs [64.0, 85.0, 67.0, 77.0, 80.0] |
| G | gas nov_2026: no EIA release yet -> scored n = 0, ex-ante forecast sealed | **PASS** | 2026-11-05 |
| D | gas: every MAE and QS recomputed from saved forecasts | **PASS** | 104/104 match |
| F | gas: every significance verdict rests on >= 10 independent units | **PASS** | 96 verdicts, 0 significant, bad=[] |
| E | gas: future-scramble leak test (storage >= target week, weather after origin) | **PASS** | 0/30 forecasts changed |
| G | fin corrected preds sha256 matches results | **PASS** |  |
| C | fin corrected: log-RV labels re-derived from raw Binance zips (60 random coin-days) | **PASS** | 59 labelled, max /diff/ 8.9e-16 |
| D | fin corrected: every QS and direction accuracy recomputed | **PASS** | 116/116 match |
| F | fin corrected: every significance verdict rests on >= 10 independent units | **PASS** | 112 verdicts, 0 significant, bad=[] |
| E | fin: future-scramble leak test (daily rows >= D, hourly rows >= D 00:00) | **PASS** | 0/6 forecasts changed |
