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

## Placebo: true vs shuffled outcomes (500 shuffles within month)

| model | n | QS true | QS shuffled | improvement | p |
|---|---|---|---|---|---|
