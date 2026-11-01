"""wm_power_blend: equal quantile blend of the frozen wm_power world model and the LEAR competitor (audit 2026-11-01).

q = sort(0.5 * q_wm_power + 0.5 * q_lear) per slot (weight chosen on VAL 2025-07..10 only, before the TEST look).
Both members are harness models and see only the harness arguments (prices <= D-1, covariates <= D-2).
LEAR refits a lasso per slot per day, so this model takes ~12 min per year of days.
"""
import numpy as np

W_LEAR = 0.5


def _factory(REGISTRY):
    def make():
        wm, lear = REGISTRY["wm_power"](), REGISTRY["lear"]()

        def batch(prices, covs, h, qs, dates=None):
            a = np.asarray(wm.batch(prices, covs, h, qs, dates=dates), float)
            b = np.asarray(lear.batch(prices, covs, h, qs, dates=dates), float)
            return np.sort((1 - W_LEAR) * a + W_LEAR * b, 1)

        def predict(context_prices, context_covariates=None, horizon=24, quantiles=None, **kw):
            return batch([context_prices], [context_covariates], horizon, quantiles,
                         **{k: [v] for k, v in kw.items()})[0]
        predict.batch = batch
        predict.wants_dates = True
        predict.uses_covariates = True
        predict.meta = {"members": {"wm_power": wm.meta, "lear": lear.meta}, "w_lear": W_LEAR,
                        "weight_chosen_on": "VAL 2025-07-01..2025-10-31"}
        return predict
    return make


def register(REGISTRY):
    REGISTRY["wm_power_blend"] = _factory(REGISTRY)
