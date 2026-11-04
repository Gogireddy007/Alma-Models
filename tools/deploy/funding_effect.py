"""Effect of the missing October-2026 funding data on wm_fin: same coin-days (2026-10-01 .. 10-30), pinned data
(funding = 0 for October, as the TEST year saw it) vs corrected data (real October funding). Inference only; reads
the saved forecasts. -> runs/deploy_2026_10_11/fin/funding_effect.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "runs", "deploy_2026_10_11", "fin")
LV = np.round(np.arange(0.1, 0.91, 0.1), 2)
a = np.load(os.path.join(RUN, "preds", "fin_preds_pinned.npz"))
b = np.load(os.path.join(RUN, "preds", "fin_preds_corrected.npz"))
la = np.load(os.path.join(RUN, "preds", "fin_LABELS_local_pinned.npz"))      # local only (market data)
lb = np.load(os.path.join(RUN, "preds", "fin_LABELS_local_corrected.npz"))
days = [d for d in a["days"] if "2026-10-01" <= d <= "2026-10-30" and d in set(b["days"])]
ia = [list(a["days"]).index(d) for d in days]
ib = [list(b["days"]).index(d) for d in days]
L = la["log_rv"][ia]
assert np.allclose(L, lb["log_rv"][ib], equal_nan=True), "labels differ between variants"
out = {"days": [days[0], days[-1], len(days)], "models": {}}
for mn in ("wm_fin", "wm_direct", "wm_fin_blend", "har_intra_dow", "har_dow", "lgbm_dow", "chronos2"):
    qa, qb = a[f"q_{mn}"][ia], b[f"q_{mn}"][ib]
    ok = np.isfinite(L) & np.isfinite(qa[..., 2:11]).all(-1) & np.isfinite(qb[..., 2:11]).all(-1)

    def qs(q):
        d = L[..., None] - q[..., 2:11]
        return float(np.mean(np.where(d >= 0, LV * d, (LV - 1) * d).mean(-1)[ok]))
    out["models"][mn] = {"qs_funding_zero": qs(qa), "qs_real_funding": qs(qb), "n_coin_days": int(ok.sum()),
                         "max_abs_forecast_change": float(np.nanmax(np.abs(qa[..., 2:11] - qb[..., 2:11])))}
json.dump(out, open(os.path.join(RUN, "funding_effect.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
