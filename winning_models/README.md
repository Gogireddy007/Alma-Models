# Winning models

There is one frozen model per project. For each one this folder holds:
- the exact checkpoints that were scored on the sealed TEST year (plus the matched "direct" twin used for the world-model test);
- `SHA256SUMS`, which match the pins in `FROZEN.json` and in the model wrappers;
- the `FROZEN.json` freeze record (recipe, seeds, K, pre-declared claims);
- a `MODEL_CARD.md` covering the architecture, the models it was tested against, how it was tested, and its audited scores and caveats.

"Winning" means each model was the best entry in its project's search, chosen on DEV/VAL only. The model cards state where each one beats the competition significantly and where it does not.

| Folder | Project | Model | Audited significant wins |
|---|---|---|---|
| [`oil_wm_sv/`](oil_wm_sv/MODEL_CARD.md) | WTI crude | Stochastic-volatility world model | #1 of 9 on the 1-day probabilistic forecast; beats ARIMA, Chronos-Bolt, Kronos |
| [`power_wm_power/`](power_wm_power/MODEL_CARD.md) | German day-ahead power | Physics-informed merit-order world model | #1 numerically on hourly direction; beats naive baselines, Kronos, zero-shot Alma |
| [`gas_wm_gas/`](gas_wm_gas/MODEL_CARD.md) | US gas storage (EIA) | Storage-balance state-space world model | Beats all external models on quantile score; beats foundation models on MAE |
| [`fin_wm_fin/`](fin_wm_fin/MODEL_CARD.md) | Crypto volatility | RSSM world model | Beats Chronos-2, TiRex, Chronos-Bolt and Kronos on quantile score; transfers to unseen coins |

**Loading:** the model code lives in `tools/<project>/` (wrappers named `wm_*.py`). Each wrapper asserts the checkpoint sha256 at load time and exposes the project harness interface. See each card for the entry point.
