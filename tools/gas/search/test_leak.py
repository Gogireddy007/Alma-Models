"""Information-set perturbation test for the architect's gas feature builder + models (docs/gas_search.md §0).

For sampled DEV target weeks (never VAL/TEST) and both origin kinds, raw sources are perturbed and the context is
rebuilt with the lead's storage.build_context; the architect's forecasts (feats.feats -> GasNet, every arch x
assim x regime, and the frozen wrappers if present) must be
  UNCHANGED when a FORBIDDEN input changes: storage levels of week W and later, observed weather from the origin
           day on, CPC issues not usable at the origin, HH prices dated after the origin day;
  CHANGED   when an ALLOWED input changes: level of week W-1 (-> dS_{W-1}, dev_{W-1}), observed weather of
           the latest allowed day (origin-1, or the week's Friday when a holiday-shifted origin falls after it), the CPC issue actually used, the HH price of the origin day.
Run: nice -n 10 .venv/bin/python tools/gas/search/test_leak.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GAS = os.path.dirname(HERE)
sys.path.insert(0, GAS)
sys.path.insert(0, HERE)
import data as D       # noqa: E402
import storage as S    # noqa: E402
import feats as FT     # noqa: E402
import nets as N       # noqa: E402

SC = {"dev_sd": [40.0, 12.0, 12.0, 3.0, 4.0, 20.0], "lvl_sd": [400.0, 100.0, 100.0, 30.0, 40.0, 150.0]}


def build_models():
    ms = []
    c0 = None
    for arch in ("wm", "direct"):
        for assim in (0, 1):
            for regime in (0, 1):
                torch.manual_seed(0)
                ms.append((f"{arch}/assim{assim}/reg{regime}", arch, assim, regime))
    return ms


def forecast(models, ctxs):
    fs = FT.stack([FT.feats(c, SC) for c in ctxs])
    b = {k: torch.tensor(v) for k, v in fs.items()}
    sd = torch.tensor(SC["dev_sd"])
    out = {}
    for name, m in models:
        with torch.no_grad():
            out[name] = m(b, sd)["q"].numpy()
    return out


def frozen_forecast(ctxs):
    out = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wm_gas", os.path.join(GAS, "wm_gas.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reg = {}
        mod.register(reg)
        for name, fac in reg.items():
            p = fac()
            out[name] = np.stack([p.storage(ctxs, [0.1, 0.5, 0.9], s) for s in S.SERIES], 1)
    except FileNotFoundError:
        pass
    return out


def main():
    rng = np.random.default_rng(7)
    w0, o, f = S.load()
    lvl, st = S.levels()
    lvl = lvl.reindex(w0.week)
    st = st.set_index("week").reindex(w0.week).reset_index()
    # sanity: rebuilding the weekly table from the levels reproduces the cached table for the sampled rows
    ks = sorted(rng.choice(S.target_weeks("dev", w0), 8, replace=False).tolist())
    ks += [int(S.target_weeks("dev", w0)[-1])]
    specs = build_models()
    models = []
    probe = S.build_context(ks[0], "thu", w0, o, f)
    fz = FT.feats(probe, SC)
    for name, arch, assim, regime in specs:
        torch.manual_seed(0)
        m = N.GasNet(fz["g"].shape[0], fz["u"].shape[1], 32, rollout=(arch == "wm"), assim=bool(assim),
                     regime=bool(regime)).eval()
        with torch.no_grad():                    # untrained nets have near-zero demand coefficients; make every
            m.coef.bias.fill_(0.5)               # input path visibly active for the sensitivity checks
        models.append((name, m))
    hh_orig = D.panel()
    ok = True
    counts = {}

    def chk(tag, a, b, must_change):
        nonlocal ok
        for name in a:
            ch = not np.allclose(a[name], b[name], atol=1e-6, rtol=0)
            good = ch if must_change else (not ch and np.array_equal(a[name], b[name]))
            counts.setdefault((tag, name), [0, 0, must_change])
            counts[(tag, name)][0] += ch
            counts[(tag, name)][1] += 1
            ok &= good

    for k in ks:
        for kind in S.ORIGINS:
            odt = S.origin_dt(w0, k, kind)
            od = pd.Timestamp(odt.date())
            c0 = S.build_context(k, kind, w0, o, f)
            base = {**forecast(models, [c0]), **frozen_forecast([c0])}

            def run(w=w0, oo=o, ff=f):
                c = S.build_context(k, kind, w, oo, ff)
                return {**forecast(models, [c]), **frozen_forecast([c])}
            # ---------------- forbidden
            l2 = lvl.copy()
            l2.iloc[k:] = rng.uniform(100, 5000, l2.iloc[k:].shape)
            chk("storage W..end", base, run(w=S.weekly_from(l2, st)), False)
            o2 = o.copy()
            mo = o2.index >= od
            o2.loc[mo] = rng.uniform(0, 60, (mo.sum(), 2))
            chk("obs wx >= origin", base, run(oo=o2), False)
            f2 = f.copy()
            bad = (f2.issue > od) | ((f2.issue == od) & (f2.last_modified > odt.tz_convert("UTC")))
            f2.loc[bad, ["hdd", "cdd"]] = rng.uniform(0, 60, (bad.sum(), 2))
            chk("unusable CPC issues", base, run(ff=f2), False)
            dates, hh, C, ex = hh_orig
            hh2 = hh.copy()
            hh2[dates > np.datetime64(od.date(), "D")] *= 3.0
            D._CACHE["p"] = (dates, hh2, C, ex)
            chk("HH after origin", base, run(), False)
            # ---------------- allowed
            hh3 = hh.copy()
            j = np.where(dates <= np.datetime64(od.date(), "D"))[0][-1]
            hh3[j] *= 1.5
            D._CACHE["p"] = (dates, hh3, C, ex)
            chk("HH last day <= origin", base, run(), True)
            D._CACHE["p"] = hh_orig
            l3 = lvl.copy()
            l3.iloc[k - 1] += 20.0
            chk("level W-1 (+20 all series)", base, run(w=S.weekly_from(l3, st)), True)
            # latest observed day the model may use: origin-1 if it lies in the target week, else the last day of
            # the week (pre-2025 holiday-shifted releases put the origin after the week end)
            last_obs = min(od - pd.Timedelta(days=1), pd.Timestamp(c0["wx_days"][-1]))
            o3 = o.copy()
            o3.loc[last_obs, "hdd"] += 10.0
            o3.loc[last_obs, "cdd"] += 5.0
            chk("obs wx latest allowed day", base, run(oo=o3), True)
            if (c0["wx_src"] >= 1).any():
                f4 = f.copy()
                sel = f4.issue == pd.Timestamp(c0["wx_issue"])
                f4.loc[sel, "hdd"] += 10.0
                f4.loc[sel, "cdd"] += 5.0
                chk("used CPC issue", base, run(ff=f4), True)
    for (tag, name), (nch, n, must) in counts.items():
        good = (nch == n) if must else (nch == 0)
        print(f"{tag:28s} {name:22s} changed {nch:2d}/{n:2d}  {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
