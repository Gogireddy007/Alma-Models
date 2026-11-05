# V1 — Mutation tests of the deployment audit

Each row plants one fake in a copy of the sealed 4 Nov snapshot; the audit must fail on it. The control copy must pass.

| mutant | planted fake | expected to trip | tripped | result |
|---|---|---|---|---|
| control_unmutated | no change | — (must pass) | — | **PASS (no false alarm)** |
| oil_bluff_metric | Report a better score than the model achieved (wm_sv October 1-day QS improved by 5%). | D | D | **CAUGHT** |
| oil_edit_sealed_forecast | Edit a sealed forecast after the fact without updating its seal. | G | D, G, R | **CAUGHT** |
| oil_fabricated_consistent | Sophisticated fake: change the forecasts, re-seal them AND recompute every reported QS consistently. | R | R | **CAUGHT** |
| gas_truth_tamper | Change a realised value (the EIA storage change of the first October week) to flatter the model. | C | C, D | **CAUGHT** |
| gas_overclaim_significance | Claim significance on 5 weeks of data. | F | F | **CAUGHT** |
| gas_fabricated_consistent | Replace the sealed wm_gas_ens October forecasts with better ones (closer to the truth), re-seal, and
    recompute MAE and QS consistently. | R | R | **CAUGHT** |
| power_label_tamper | Change realised prices (the labels) for one delivery day. | C | C, D | **CAUGHT** |
| power_weights_changed | Report that a network's parameters changed during deployment (i.e. it was trained). | B | B | **CAUGHT** |
| fin_label_tamper | Shift realised log-RV labels of three days. | C | C, D | **CAUGHT** |
| fin_fabricated_resealed | Edit the wm_fin forecasts and re-seal them, but leave the metrics. | D, R | D, R | **CAUGHT** |
| fin_online_update | Report a continual-learning update during deployment. | B | B | **CAUGHT** |

**12/12 as expected.**
