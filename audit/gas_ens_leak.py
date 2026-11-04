"""Leak test for wm_gas_ens: scramble the GLOBAL storage/weather tables (what the wrapper's interval step reads) for
everything at/after the target week / origin day, rebuild, and require identical quantiles for all 6 series."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo", "tools", "gas"))
import storage as S, models as M
w0, o0, f0 = [x.copy() for x in S.load()]
ks = S.target_weeks("test", w0)[::5]; rng = np.random.default_rng(1); bad = 0; n = 0
for k in ks:
    for kind in S.ORIGINS:
        S._C.clear(); S._C["w"], S._C["o"], S._C["f"] = w0.copy(), o0.copy(), f0.copy()
        m = M.get("wm_gas_ens"); c = S.build_context(int(k), kind)
        base = m.quantiles([c]).copy()
        w, o, f = w0.copy(), o0.copy(), f0.copy()
        for s in S.SERIES:
            for col in (f"ds_{s}", f"dev_{s}", f"lvl_{s}"):
                w.loc[w.index >= k, col] = rng.normal(0, 100, (w.index >= k).sum())
        od = pd.Timestamp(S.origin_dt(w0, int(k), kind).date())
        o.loc[o.index >= od, ["hdd", "cdd"]] = rng.uniform(0, 60, ((o.index >= od).sum(), 2))
        f.loc[f.issue > od, ["hdd", "cdd"]] = rng.uniform(0, 60, ((f.issue > od).sum(), 2))
        S._C.clear(); S._C["w"], S._C["o"], S._C["f"] = w, o, f
        m2 = M.get("wm_gas_ens"); c2 = S.build_context(int(k), kind)
        pert = m2.quantiles([c2])
        n += 1; bad += not np.allclose(base, pert, atol=1e-6)
print(f"wm_gas_ens leak test: {bad} of {n} forecasts changed (all 6 series, intervals included)")
