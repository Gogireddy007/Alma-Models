# V5 — Calibration, property tests, placebo

## Property tests (frozen models, October 2026 inputs)

| project | property | cases | violations | detail |
|---|---|---|---|---|
| power | less wind never lowers any price quantile (merit order) | 48360 | 0 | mean median price by wind change {"-0.5": 185.16, "-0.3": 181.32, "-0.1": 177.13, "0.0": 174.96, "0.1": 172.78, "0.3": 168.51} |
| gas | colder forecast (+HDD) never raises predicted injection | 30 | 0 | mean change in the weekly median for +20 HDD/day on forecast days: -34.2 Bcf |
| oil | twice-as-volatile recent returns widen the 80% interval | 125 | 0 | mean width ratio 1.50x |
| crypto | higher recent realised vol raises the predicted median vol | 192 | 0 | mean rise in median log-RV for +1 log-RV over 3 days: 0.443 |

## Calibration, Jul 2025 - Nov 2026 (share of outcomes below each quantile; ideal = the level)

| model | n | max |share - level| | 80% coverage |
|---|---|---|---|
| oil/wm_sv | 337 | 0.038 | 0.766 |
| oil/wm_sv_direct | 337 | 0.031 | 0.766 |
| oil/rw_gauss | 337 | 0.034 | 0.801 |
| oil/chronos2 | 337 | 0.077 | 0.825 |
| oil/tirex | 337 | 0.037 | 0.772 |
| power/wm_power | 11831 | 0.063 | 0.758 |
| power/wm_power_blend | 11831 | 0.017 | 0.784 |
| power/wm_power_direct | 11831 | 0.070 | 0.750 |
| power/lear | 11831 | 0.048 | 0.750 |
| power/chronos2 | 11831 | 0.050 | 0.738 |
| power/tirex | 11831 | 0.051 | 0.767 |
| gas/wm_gas_ens | 70 | 0.086 | 0.843 |
| gas/wm_gas | 70 | 0.143 | 0.857 |
| gas/wm_gas_direct | 70 | 0.129 | 0.871 |
| gas/st_ols_wx | 70 | 0.114 | 0.843 |
| gas/chronos2 | 70 | 0.143 | 0.757 |
| crypto/wm_fin | 11736 | 0.040 | 0.840 |
| crypto/wm_fin_blend | 11616 | 0.053 | 0.870 |
| crypto/wm_direct | 11736 | 0.050 | 0.842 |
| crypto/har_intra_dow | 11616 | 0.081 | 0.889 |
| crypto/chronos2 | 11760 | 0.031 | 0.795 |
| crypto/tirex | 11760 | 0.038 | 0.811 |

## Placebo: true vs shuffled outcomes (500 shuffles within month)

| model | n | QS true | QS shuffled | improvement | p |
|---|---|---|---|---|---|
| oil/wm_sv | 337 | 0.009165 | 0.02405 | 61.9% | 0.0020 |
| oil/wm_sv_direct | 337 | 0.009208 | 0.02425 | 62.0% | 0.0020 |
| oil/rw_gauss | 337 | 0.0092 | 0.0241 | 61.8% | 0.0020 |
| oil/chronos2 | 337 | 0.009523 | 0.0237 | 59.8% | 0.0020 |
| oil/tirex | 337 | 0.009435 | 0.02407 | 60.8% | 0.0020 |
| power/wm_power | 11831 | 8.601 | 24.32 | 64.6% | 0.0020 |
| power/wm_power_blend | 11831 | 8.171 | 23.81 | 65.7% | 0.0020 |
| power/wm_power_direct | 11831 | 8.63 | 24.27 | 64.4% | 0.0020 |
| power/lear | 11831 | 8.689 | 23.73 | 63.4% | 0.0020 |
| power/chronos2 | 11831 | 8.151 | 24.17 | 66.3% | 0.0020 |
| power/tirex | 11831 | 8.363 | 23.73 | 64.8% | 0.0020 |
| gas/wm_gas_ens | 70 | 4.277 | 42.88 | 90.0% | 0.0020 |
| gas/wm_gas | 70 | 4.545 | 42.85 | 89.4% | 0.0020 |
| gas/wm_gas_direct | 70 | 4.65 | 42.44 | 89.0% | 0.0020 |
| gas/st_ols_wx | 70 | 6.114 | 42.02 | 85.4% | 0.0020 |
| gas/chronos2 | 70 | 10.15 | 39.47 | 74.3% | 0.0020 |
| crypto/wm_fin | 11736 | 0.1491 | 0.4007 | 62.8% | 0.0020 |
| crypto/wm_fin_blend | 11616 | 0.148 | 0.3926 | 62.3% | 0.0020 |
| crypto/wm_direct | 11736 | 0.1483 | 0.4005 | 63.0% | 0.0020 |
| crypto/har_intra_dow | 11616 | 0.1568 | 0.386 | 59.4% | 0.0020 |
| crypto/chronos2 | 11760 | 0.1593 | 0.397 | 59.9% | 0.0020 |
| crypto/tirex | 11760 | 0.1612 | 0.3935 | 59.0% | 0.0020 |
