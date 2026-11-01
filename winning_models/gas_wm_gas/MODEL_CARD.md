# wm_gas: US natural-gas storage world model

## Architecture
A daily **storage-balance state-space world model**:
1. **Encoder.** An encoder reads the released history and produces a latent state z.
2. **Daily rollout.** A GRUCell rolls z forward **one day at a time**. It first runs over week W-1, using observed weather and writing the innovation against the released ΔS for W-1 into the state, which acts as data assimilation. It then runs over the 7 days of target week W.
3. **Regional readout.** The readout is monotone in HDD/CDD for each region and anchored at the 5-year average, with a 3-regime mixture. The Lower-48 total is the sum of the 5 regions.
4. **Quantiles.** Quantiles are the median plus holdout residual quantiles.

**Training.**
- Model: d = 32, about 23.5K params per member.
- Optimiser: AdamW, lr 3e-3 OneCycle, wd 0.05, dropout 0.3, auxiliary loss weight 0.3, L48 loss weight 3.
- Schedule: 1,200 steps, batch 64.
- Ensemble: seeds 0, 1, 2, averaged by quantile.
- Data: trained on 547 weeks (2015-01 to 2025-06).

**Matched twin `wm_gas_direct`.** Identical config except `arch=direct`: same modules, parameter count, data, steps and seeds. Each day's state is one GRUCell step from the origin state, with no state carried over.

**Code and checkpoints.**
- Code: `tools/gas/wm_gas.py` and `tools/gas/wm_scenario.py`, which runs scenario simulations (`runs/gas/scenario_demo.json`).
- Checkpoints: `checkpoints/wm_gas.pt` and `wm_gas_direct.pt`.

## How it was tested
- **Protocol.** Pre-registered in `docs/gas_prereg.md` (A1–A3). Leakage design in `docs/gas_leakage.md`. Weather inputs are vintage-checked. TEST covered 52 weeks from 2025-11 and was run once offline.
- **Targets.** The weekly EIA storage change, forecast from Thursday and Friday origins. Measures: MAE in Bcf, quantile score, direction (G1) and coverage.
- **Competitors.**
  - Statistical: OLS with weather (with and without lag), persistence, seasonal naive, climatology and ARIMA.
  - Foundation models: Chronos-2, Chronos-2 with covariates, Chronos-Bolt, TiRex and Kronos-small.
  - Zero-shot Alma.
  - Timestral, which is contaminated.
- **Audit.** `docs/audit_gas.md` re-derived the results with 0 differences. Amendment A3 adds a shutdown-week robustness check, which left every result unchanged.

## Audited TEST results
| Metric | wm_gas | Competitors | Verdict |
|---|---|---|---|
| Quantile score | **Best of all external models** | OLS next | **Significant vs OLS** (Thursday Holm 0.030, Friday 0.039) and vs every foundation model |
| MAE (Bcf) | **11.5** | OLS 14.4, Chronos-2-cov 19.8, TiRex 29.1 | **Significant vs every clean foundation model** (Chronos-2-cov Holm 0.0016); vs OLS p 0.13, not significant |
| Thursday direction G1 | 0.788 | twin 0.865, OLS 0.808 | No significant accuracy win |
| World-model test vs twin | n/a | n/a | Not passed: the twin is numerically better everywhere |

## Caveats
- Intervals are conservative: 80% coverage is 0.87–0.89.
- The protocol was amended mid-search (Am1/Am2) after early networks overfit.
- The Henry Hub price task has its TEST period already exposed.
