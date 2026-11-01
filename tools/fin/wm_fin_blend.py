"""wm_fin_blend: quantile blend of the frozen wm_fin world model with the har_intra_dow baseline (audit 2026-11-01).

q = sort(0.6 * q_wm_fin + 0.4 * q_har_intra_dow) over the 13 log-RV quantiles (weight 0.4 chosen on VAL
2025-07..10 only, before the TEST look). p_up = P(log RV(D) > ref) read off the blended quantile function by linear
interpolation, ref = mean log RV(D-7..D-1) from the context, so p_up > 0.5 exactly when the blended median > ref.
Uses only the harness context (both members are harness forecasters). Deterministic. Crypto only: har_intra_dow
needs hourly 5-min RV.
"""
import numpy as np

W_HAR = 0.4


def _p_above(q, levels, ref):
    """q [m, L] sorted quantiles, ref [m] -> P(Y > ref) by linear interpolation of the CDF (clipped to the grid ends)."""
    out = np.full(q.shape[0], np.nan)
    for i in range(q.shape[0]):
        if np.isfinite(q[i]).all() and np.isfinite(ref[i]):
            out[i] = 1.0 - np.interp(ref[i], q[i], levels, left=levels[0] / 2, right=1 - (1 - levels[-1]) / 2)
    return out


class FinBlend:
    kind = "forecaster"

    def __init__(self, registry):
        self.wm = registry["wm_fin"]()
        self.har = registry["har_intra_dow"]()
        self.meta = {"members": {"wm_fin": getattr(self.wm, "meta", {}), "har_intra_dow": "tools/fin/models.py"},
                     "w_har": W_HAR, "weight_chosen_on": "VAL 2025-07-01..2025-10-31"}

    def predict(self, ctx):
        a, b = self.wm.predict(ctx), self.har.predict(ctx)
        q = np.sort((1 - W_HAR) * np.asarray(a["q"], float) + W_HAR * np.asarray(b["q"], float), -1)
        lr = np.asarray(ctx["daily"]["logrv"], float)
        lb = int(ctx.get("lookback", 7))
        ref = np.where(np.isfinite(lr[-lb:]).all(0), lr[-lb:].mean(0), np.nan)
        out = {"q": q, "p_up": _p_above(q, np.asarray(ctx["quantiles"], float), ref)}
        if "p_ret_up" in a:
            out["p_ret_up"] = a["p_ret_up"]
        return out


def register(REGISTRY):
    REGISTRY["wm_fin_blend"] = lambda: FinBlend(REGISTRY)
