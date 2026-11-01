# Winning models

There is one frozen model per project. For each one this folder holds:
- the exact checkpoints that were scored on the sealed TEST year (plus the matched "direct" twin used for the world-model test);
- `SHA256SUMS`, which match the pins in `FROZEN.json` and in the model wrappers;
- the `FROZEN.json` freeze record (recipe, seeds, K, pre-declared claims);
- a `MODEL_CARD.md` covering the architecture, the models it was tested against, how it was tested, and its audited scores and caveats.

"Winning" means each model was the best entry in its project's search, chosen on DEV/VAL only. The model cards state where each one beats the competition significantly and where it does not.

| Folder | Project | Model | Audited significant wins |
|---|---|---|---|
| [`oil_wm_sv/`](oil_wm_sv/MODEL_CARD.md) | WTI crude | Stochastic-volatility world model Ties the random walk on the 1-day probabilistic forecast (numerically first of 9, +0.45% skill, p 0.48); beats ARIMA, Chronos-Bolt, Kronos |
| [`power_wm_power/`](power_wm_power/MODEL_CARD.md) | German day-ahead power | Physics-informed merit-order world model | #1 numerically on hourly direction; beats naive baselines, Kronos, zero-shot Alma |
| [`gas_wm_gas/`](gas_wm_gas/MODEL_CARD.md) | US gas storage (EIA) | Storage-balance state-space world model | Beats all external models on quantile score; beats foundation models on MAE |
| [`fin_wm_fin/`](fin_wm_fin/MODEL_CARD.md) | Crypto volatility | RSSM world model | Beats Chronos-2, TiRex, Chronos-Bolt and Kronos on quantile score; on held-out coins it ties the weekday-aware HAR baseline |

### Follow-up models (audit, 2026-11-01)

Built after an independent re-audit. Every choice was fixed on DEV or VAL; their TEST numbers are a **second look**
at a year that had already been scored, so treat them as indicative until forward data confirms them.

| Folder | Model | TEST vs the frozen model | Significant? |
|---|---|---|---|
| [`gas_wm_gas_ens/`](gas_wm_gas_ens/MODEL_CARD.md) | 14 seeds + 0.2 OLS + adaptive intervals | MAE 10.6 vs 11.5 Bcf; QS 4.14 vs 4.59 | Beats OLS on MAE and QS; not significant vs wm_gas |
| [`fin_wm_fin_blend/`](fin_wm_fin_blend/MODEL_CARD.md) | 0.6 wm_fin + 0.4 har_intra_dow | Direction 0.740 vs 0.728; QS 0.1459 vs 0.1471 | +1.7 pp vs wm_fin; not significant vs har_intra_dow on direction |
| [`power_wm_power_blend/`](power_wm_power_blend/MODEL_CARD.md) | 0.5 wm_power + 0.5 LEAR | QS 8.60 vs 9.03; P2 0.755 vs 0.746 | QS vs both members; P2 not significant vs wm_power |

**Loading:** the model code lives in `tools/<project>/` (wrappers named `wm_*.py`). Each wrapper asserts the checkpoint sha256 at load time and exposes the project harness interface. See each card for the entry point.
