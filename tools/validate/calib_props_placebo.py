"""V5 — calibration, physics/economics property tests, and placebo tests.

    .venv/bin/python tools/validate/calib_props_placebo.py   # -> runs/validation/v5/RESULTS.json + .md

Calibration (uses the V4 post-training forecasts in runs/validation/posttrain, Jul 2025 - Nov 2026): for every
quantile level tau, the share of outcomes below the tau-quantile (ideal = tau), and interval coverage.

Property tests (frozen models, real October 2026 inputs, one input moved at a time):
  power  less wind (higher residual load, via the model's own intervention lever) -> no price quantile may fall;
  gas    colder weather forecast for the target week (+HDD) -> predicted storage injection must not rise;
  oil    recent returns made twice as volatile -> the 80% interval must widen;
  crypto recent realised volatility raised -> the predicted median log-RV must rise.

Placebo: the realised values are shuffled across dates inside each window (500 times). A forecast that carries real
information must score far better on the true pairing than on shuffled ones (one-sided permutation p-value).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PT = os.path.join(ROOT, "runs", "validation", "posttrain")
OUT = os.path.join(ROOT, "runs", "validation", "v5")
QS13 = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])
LV9 = np.round(np.arange(0.1, 0.91, 0.1), 2)
R = {"calibration": {}, "properties": [], "placebo": {}}


def gas_nocache(p):
    """The frozen gas wrappers memoise storage() by (target_week, origin) only, not by content. A test that re-queries
    the same week with altered inputs must clear every cache first, or it silently gets the old answer
    (found by the V3 positive control; tools/gas/wm_gas.py itself is frozen and is not changed)."""
    for o in [p] + list(getattr(p, "ens", [])):
        for attr in ("_cache", "_raw"):           # wm_gas: _cache; wm_gas_ens: _cache and the persistent _raw
            if hasattr(o, attr):
                setattr(o, attr, {})
    return p


def pin(y, q, lv):
    e = y[..., None] - q
    return np.where(e >= 0, lv * e, (lv - 1) * e).mean(-1)


def calib(name, y, q, lv):
    """y [n], q [n, L] -> reliability per level and 80% coverage. 13-level forecasts are reduced to the 9 levels
    0.1..0.9 that every model provides (TiRex and Chronos-Bolt have no 2.5/5/95/97.5% levels), so all models are
    compared on the same levels."""
    if q.shape[-1] == 13:
        q, lv = q[..., 2:11], LV9
    ok = np.isfinite(y) & np.isfinite(q).all(-1)
    y, q = y[ok], q[ok]
    below = (y[:, None] <= q).mean(0)
    lo, hi = list(np.round(lv, 3)).index(0.1), list(np.round(lv, 3)).index(0.9)
    R["calibration"][name] = {"n": int(ok.sum()), "levels": [float(x) for x in lv], "share_below": below.tolist(),
                              "max_abs_dev": float(np.max(np.abs(below - lv))),
                              "cov80": float(((y >= q[:, lo]) & (y <= q[:, hi])).mean())}


def placebo(name, y, q, lv, groups, n_perm=500, seed=0):
    """Shuffle y within groups (months); compare the true mean pinball with the shuffled distribution."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y) & np.isfinite(q).all(-1)
    y, q, groups = y[ok], q[ok], np.asarray(groups)[ok]
    true = float(pin(y, q, lv).mean())
    perm = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = rng.permutation(yp[m])
        perm.append(pin(yp, q, lv).mean())
    perm = np.array(perm)
    R["placebo"][name] = {"n": int(ok.sum()), "qs_true": true, "qs_shuffled_mean": float(perm.mean()),
                          "improvement_vs_shuffled": float(1 - true / perm.mean()),
                          "p_value": float((1 + (perm <= true).sum()) / (1 + n_perm))}


def prop(project, test, n, n_viol, detail):
    R["properties"].append({"project": project, "test": test, "n_cases": n, "violations": n_viol, "ok": n_viol == 0,
                            "detail": detail})
    print(f"[{'OK ' if n_viol == 0 else 'BAD'}] {project:6s} {test}: {n_viol}/{n} violations; {detail}", flush=True)


# =========================================================================== calibration + placebo (from V4)
def from_v4():
    # oil, 1-day
    r = json.load(open(os.path.join(PT, "oil", "results.json")))
    z = np.load(os.path.join(ROOT, r["preds_file"]))
    df = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    d, v = pd.to_datetime(df.date).to_numpy(), df.v.to_numpy(float)
    oi = z["origins"]
    ok = (oi < len(v)) & (d[np.minimum(oi, len(v) - 1)] >= np.datetime64("2025-07-01"))
    y = np.log(v[oi[ok]])
    mon = pd.to_datetime(d[oi[ok]]).strftime("%Y-%m")
    for mn in ("wm_sv", "wm_sv_direct", "rw_gauss", "chronos2", "tirex"):
        k = f"q_{mn}"
        if k in z.files:
            q = np.log(z[k][ok][:, :, 0])
            calib(f"oil/{mn}", y, q, QS13)
            placebo(f"oil/{mn}", y, q[:, 2:11], LV9, mon)
    # power, 24 hourly prices per day
    r = json.load(open(os.path.join(PT, "power", "results.json")))
    zq = np.load(os.path.join(ROOT, r["preds_file"]))
    zl = np.load(os.path.join(PT, "power", "preds", "power_LABELS_local.npz"))
    days = zq["delivery_days"]
    y = zl["actual"]
    mon = np.repeat(pd.to_datetime(days).strftime("%Y-%m").to_numpy()[:, None], 24, 1).ravel()
    for mn in ("wm_power", "wm_power_blend", "wm_power_direct", "lear", "chronos2", "tirex"):
        k = f"q_{mn}"
        if k in zq.files:
            q = np.moveaxis(zq[k], 1, 2).reshape(-1, 9)
            calib(f"power/{mn}", y.ravel(), q, LV9)
            placebo(f"power/{mn}", y.ravel(), q, LV9, mon)
    # gas, Thursday origin
    r = json.load(open(os.path.join(PT, "gas", "results.json")))
    z = np.load(os.path.join(ROOT, r["preds_file"]))
    w = r["windows"]["ALL_POST_TRAINING"]
    y = np.array(w["actual_bcf"], float)
    mon = pd.to_datetime(w["releases"]).strftime("%Y-%m").to_numpy()
    for mn in ("wm_gas_ens", "wm_gas", "wm_gas_direct", "st_ols_wx", "chronos2"):
        k = f"ALL_POST_TRAINING__thu__{mn}"
        if k in z.files:
            calib(f"gas/{mn}", y, z[k], QS13)
            placebo(f"gas/{mn}", y, z[k][:, 2:11], LV9, ["all"] * len(y))   # few weeks per month: shuffle the period
    # crypto, 24 coins
    r = json.load(open(os.path.join(PT, "fin", "results_corrected.json")))
    zq = np.load(os.path.join(ROOT, r["preds_file"]))
    zl = np.load(os.path.join(PT, "fin", "preds", "fin_LABELS_local_corrected.npz"))
    L = zl["log_rv"]
    mon = np.repeat(pd.to_datetime(zq["days"]).strftime("%Y-%m").to_numpy()[:, None], L.shape[1], 1).ravel()
    for mn in ("wm_fin", "wm_fin_blend", "wm_direct", "har_intra_dow", "chronos2", "tirex"):
        k = f"q_{mn}"
        if k in zq.files:
            q = zq[k].reshape(-1, 13)
            calib(f"crypto/{mn}", L.ravel(), q, QS13)
            placebo(f"crypto/{mn}", L.ravel(), q[:, 2:11], LV9, mon)


# =========================================================================== property tests
def prop_power():
    sys.path.insert(0, os.path.join(ROOT, "tools/power"))
    import data as D
    import wm_power as WP
    dates, P, C, flags = D.panel()
    idx = [i for i, x in enumerate(dates) if np.datetime64("2026-10-01") <= x <= np.datetime64("2026-10-31")]
    ens = WP.Ensemble(WP.FROZEN["wm_power"])
    cs = [D.build_context(i, 736, dates, P, C, flags) for i in idx]
    args = ([c["prices"] for c in cs], [c["covs"] for c in cs],
            [{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in cs])
    levels = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3]                 # wind change (fraction); lower wind = higher residual load
    Q = {x: ens.quantiles_eur(*args, rl_shift_frac_wind=-x if x != 0 else None) for x in levels}
    viol, n = 0, 0
    for a, b in zip(levels[:-1], levels[1:]):                  # wind rises from a to b -> no quantile may rise
        dq = Q[b] - Q[a]
        viol += int((dq > 1e-6).sum())
        n += dq.size
    med = {x: float(np.mean(Q[x][:, 6])) for x in levels}
    prop("power", "less wind never lowers any price quantile (merit order)", n, viol,
         f"mean median price by wind change {json.dumps({str(k): round(v, 2) for k, v in med.items()})}")


def prop_gas():
    sys.path.insert(0, os.path.join(ROOT, "tools/gas"))
    import storage as S
    import models as M
    w, o, f = S.load()
    ks = np.where((w.avail >= "2026-10-01") & (w.avail <= "2026-10-31") & w.ds_l48.notna())[0]
    p = M.get("wm_gas_ens")
    viol, n, eff = 0, 0, []
    for k in ks:
        for kind in S.ORIGINS:
            c = S.build_context(int(k), kind, w, o, f)
            fut = np.asarray(c["wx_src"]) != 0                 # days of the target week known only as forecasts
            meds = []
            for dh in (0, 5, 10, 20):
                cc = dict(c, wx=np.array(c["wx"], float))
                cc["wx"][fut, 0] += dh
                meds.append(float(np.asarray(gas_nocache(p).storage([cc], [0.5], "l48"), float)[0, 0]))
            viol += int(np.sum(np.diff(meds) > 1e-6))
            n += len(meds) - 1
            eff.append(meds[-1] - meds[0])
    prop("gas", "colder forecast (+HDD) never raises predicted injection", n, viol,
         f"mean change in the weekly median for +20 HDD/day on forecast days: {np.mean(eff):.1f} Bcf")


def prop_oil():
    sys.path.insert(0, os.path.join(ROOT, "tools/commodity"))
    import wm_commodity as W
    df = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(float)
    p = W.make_predict_ens([(os.path.join(ROOT, a), s) for a, s in W.FROZEN["wm_sv"]])
    viol, n, ratio = 0, 0, []
    for i in range(len(v) - 25, len(v)):
        c = v[i - 1024:i].copy()
        lr = np.diff(np.log(c))
        lr2 = lr.copy()
        lr2[-20:] = lr[-20:].mean() + 2 * (lr[-20:] - lr[-20:].mean())
        c2 = np.exp(np.log(c[0]) + np.r_[0, np.cumsum(lr2)])
        c2 *= c[-1] / c2[-1]
        q1, q2 = p.batch([c], 5, [0.1, 0.9])[0], p.batch([c2], 5, [0.1, 0.9])[0]
        w1, w2 = np.log(q1[1] / q1[0]), np.log(q2[1] / q2[0])          # log width per horizon
        viol += int(np.sum(w2 <= w1))
        n += len(w1)
        ratio.append(float(np.mean(w2 / w1)))
    prop("oil", "twice-as-volatile recent returns widen the 80% interval", n, viol,
         f"mean width ratio {np.mean(ratio):.2f}x")


def prop_fin():
    os.environ["ALMA_FIN_TEST_GO"] = "1"
    sys.path.insert(0, os.path.join(ROOT, "tools/fin"))
    import data as DT
    import harness as H
    import models as M
    d0, h0 = DT.load("crypto_daily", True), DT.load("crypto_hourly", True)
    P0 = H.Panel("crypto", frames={"daily": d0, "hourly": h0})
    p = M.get("wm_fin")
    days = [i for i, x in enumerate(P0.days) if np.datetime64("2026-10-05", "ns") <= x <= np.datetime64("2026-10-26", "ns")][::3]
    viol, n, eff = 0, 0, []
    td, th = pd.to_datetime(d0.t), pd.to_datetime(h0.t)
    for i in days:
        D = pd.Timestamp(P0.days[i])
        a = H.run(P0, p, [i])["q"][0][:, 6]
        d1, h1 = d0.copy(), h0.copy()
        md = ((td >= D - pd.Timedelta(days=3)) & (td < D)).to_numpy()
        mh = ((th >= D - pd.Timedelta(days=3)) & (th < D)).to_numpy()
        d1.loc[md, "rv"] = d0.loc[md, "rv"].to_numpy() * np.e           # +1 in log RV for the last 3 days
        if "rv5" in h1:
            h1.loc[mh, "rv5"] = h0.loc[mh, "rv5"].to_numpy() * np.e
        b = H.run(H.Panel("crypto", frames={"daily": d1, "hourly": h1}), p, [i])["q"][0][:, 6]
        okm = np.isfinite(a) & np.isfinite(b)
        viol += int(np.sum(b[okm] <= a[okm]))
        n += int(okm.sum())
        eff.append(float(np.mean(b[okm] - a[okm])))
    prop("crypto", "higher recent realised vol raises the predicted median vol", n, viol,
         f"mean rise in median log-RV for +1 log-RV over 3 days: {np.mean(eff):.3f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = sys.argv[1:] or ["power", "gas", "oil", "fin", "v4"]       # one project per process (module names clash)
    PROPS = {"power": prop_power, "gas": prop_gas, "oil": prop_oil, "fin": prop_fin}
    names = {"power": "power", "gas": "gas", "oil": "oil", "fin": "crypto"}
    prev = os.path.join(OUT, "RESULTS.json")
    if os.path.exists(prev):                                  # keep what is not re-run this time
        old = json.load(open(prev))
        R["properties"] = [p for p in old.get("properties", [])
                           if p["project"] not in {names[x] for x in parts if x in names} | {f"prop_{x}" for x in parts}]
        if "v4" not in parts:
            R["calibration"], R["placebo"] = old.get("calibration", {}), old.get("placebo", {})
    fns = [PROPS[x] for x in parts if x in PROPS] + ([from_v4] if "v4" in parts else [])
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001  (reported as a failed test, never hidden)
            R["properties"].append({"project": fn.__name__, "test": "ran to completion", "n_cases": 0, "violations": 1,
                                    "ok": False, "detail": f"{type(e).__name__}: {e}"})
            print("ERROR", fn.__name__, e, flush=True)
    json.dump(R, open(os.path.join(OUT, "RESULTS.json"), "w"), indent=1)
    L = ["# V5 — Calibration, property tests, placebo\n", "## Property tests (frozen models, October 2026 inputs)\n",
         "| project | property | cases | violations | detail |", "|---|---|---|---|---|"]
    for p in R["properties"]:
        L.append(f"| {p['project']} | {p['test']} | {p['n_cases']} | {p['violations']} | {p['detail']} |")
    L += ["\n## Calibration, Jul 2025 - Nov 2026 (share of outcomes below each quantile; ideal = the level)\n",
          "| model | n | max |share - level| | 80% coverage |", "|---|---|---|---|"]
    for k, c in R["calibration"].items():
        L.append(f"| {k} | {c['n']} | {c['max_abs_dev']:.3f} | {c['cov80']:.3f} |")
    L += ["\n## Placebo: true vs shuffled outcomes (500 shuffles within month)\n",
          "| model | n | QS true | QS shuffled | improvement | p |", "|---|---|---|---|---|---|"]
    for k, c in R["placebo"].items():
        L.append(f"| {k} | {c['n']} | {c['qs_true']:.4g} | {c['qs_shuffled_mean']:.4g} | "
                 f"{100 * c['improvement_vs_shuffled']:.1f}% | {c['p_value']:.4f} |")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("-> runs/validation/v5/RESULTS.md")


if __name__ == "__main__":
    main()
