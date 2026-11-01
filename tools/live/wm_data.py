"""
Alma-Live Gas: interim point-in-time adapter (architect-owned) until the data lead's asof(t) store lands.

Contract (docs/live_search.md 0.2): every input at decision time t (16:00 America/New_York on trade date t) is built
ONLY from raw rows whose availability time is <= t. All filtering happens inside `features_at(j, R)` through
availability arrays, so the identical code path is used for training panels, inference and the randomized-future
perturbation test (`perturb_future`).

    nice -n 10 .venv/bin/python tools/live/wm_data.py build     # -> runs/live/search/cache/panel.npz
    nice -n 10 .venv/bin/python tools/live/wm_data.py leaktest  # feature-level perturbation test
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GAS = os.path.join(ROOT, "data", "gas")
CACHE = os.path.join(ROOT, "runs", "live", "search", "cache")
PANEL = os.path.join(CACHE, "panel.npz")

FIRST_DECISION = pd.Timestamp("2015-01-02")
DEV_END = pd.Timestamp("2024-06-30")
VAL_START, VAL_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-10-31")
EXPOSED_START = pd.Timestamp("2025-11-01")
FOLD_YEARS = [2020, 2021, 2022, 2023]
H = 7                   # rollout calendar days
NU = 7                  # exogenous inputs per rollout day
FEATS = ["lp_m250", "r1", "r5", "r20", "lrv5", "lrv20", "lrv60", "lp_m60", "lp",
         "dev", "ds", "dev_chg4", "last_surp", "days_since_rep", "days_to_rep", "avg5_next", "lvl",
         "ohdd_a", "ocdd_a", "ohdd7_a", "ocdd7_a", "fc_age", "rev_hdd", "rev_cdd",
         "sin_doy", "cos_doy", "dow0", "dow1", "dow2", "dow3", "dow4", "p_age"]
NF = len(FEATS)
UFEATS = ["fhdd_a", "fcdd_a", "fmask", "fhdd", "fcdd", "weekend", "thursday"]
TARGETS = ["r1", "r2", "r5", "lrv5", "dsurp"]
# masked latent slots (store not landed): supply/rigs, production, LNG feedgas, COT positioning
MASKED_SLOTS = ["rigs", "production", "lng_feedgas", "cot_net"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _d(x):
    return np.asarray(pd.to_datetime(x).values.astype("datetime64[D]"))


def close_utc(day):
    """16:00 America/New_York on calendar day `day` as UTC datetime64[s]."""
    ts = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return np.datetime64(ts.tz_convert("UTC").tz_localize(None), "s")


def load_raw():
    """Raw tables as numpy arrays with availability keys. Pinned files: sha256 recorded in the panel."""
    d = pd.read_csv(os.path.join(GAS, "daily.csv"), usecols=["date", "hh"]).dropna()
    w = pd.read_csv(os.path.join(GAS, "weekly.csv"), usecols=["week", "avail", "lvl_l48", "ds_l48", "avg5_l48",
                                                              "dev_l48"])
    o = pd.read_csv(os.path.join(GAS, "cpc_obs.csv"))
    f = pd.read_csv(os.path.join(GAS, "cpc_fcst.csv"))
    lm = pd.to_datetime(f["last_modified"], utc=True).dt.tz_localize(None)
    R = {
        "p_date": _d(d["date"]), "p": d["hh"].to_numpy(float),
        "s_week": _d(w["week"]), "s_avail": _d(w["avail"]), "s_lvl": w["lvl_l48"].to_numpy(float),
        "s_ds": w["ds_l48"].to_numpy(float), "s_avg5": w["avg5_l48"].to_numpy(float),
        "s_dev": w["dev_l48"].to_numpy(float),
        "o_date": _d(o["date"]), "o_hdd": o["hdd"].to_numpy(float), "o_cdd": o["cdd"].to_numpy(float),
        "f_issue": _d(f["issue"]), "f_day": _d(f["day"]), "f_hdd": f["hdd"].to_numpy(float),
        "f_cdd": f["cdd"].to_numpy(float), "f_lm": np.asarray(lm.values.astype("datetime64[s]")),
    }
    # HH spot publication lag (red team LRT: EIA/FRED daily spot is published weekly). Conservative interim rule
    # until the store provides real publication timestamps: a trade date d in the week ending Friday F becomes
    # usable on the Wednesday F+5 (decision at 16:00 ET).
    wd = pd.DatetimeIndex(R["p_date"]).weekday.to_numpy()
    fri = R["p_date"] + (np.maximum(4 - wd, 0)).astype("timedelta64[D]")
    R["p_avail"] = fri + np.timedelta64(5, "D")
    R["pins"] = {k: sha256(os.path.join(GAS, k)) for k in ["daily.csv", "weekly.csv", "cpc_obs.csv", "cpc_fcst.csv"]}
    return R


def normals(R, year):
    """causal degree-day normal by day-of-year (1..366): mean of observed values on doy +/- 7, years < `year`."""
    m = R["o_date"] < np.datetime64(f"{year}-01-01")
    dates = pd.DatetimeIndex(R["o_date"][m])
    doy = dates.day_of_year.to_numpy()
    out = np.zeros((367, 2))
    h, c = R["o_hdd"][m], R["o_cdd"][m]
    for k in range(1, 367):
        dd = np.abs(doy - k)
        dd = np.minimum(dd, 366 - dd)
        s = dd <= 7
        out[k] = [np.nanmean(h[s]), np.nanmean(c[s])]
    return out


_NCACHE = {}


def _normal(R, year, key):
    k = (key, year)
    if k not in _NCACHE:
        _NCACHE[k] = normals(R, year)
    return _NCACHE[k]


def _usable_issue(R, t, before=None):
    """latest CPC issue usable at decision time t (and the previous usable one if before is given)."""
    tc = close_utc(t)
    ok = (R["f_issue"] < t) | ((R["f_issue"] == t) & (R["f_lm"] <= tc))
    iss = np.unique(R["f_issue"][ok])
    if len(iss) == 0:
        return None, None
    return iss[-1], (iss[-2] if len(iss) > 1 else None)


def _fc(R, issue):
    m = R["f_issue"] == issue
    # keys are numpy datetime64[D] (LRT-6: .tolist() gave datetime.date keys that never matched)
    return {np.datetime64(dd, "D"): v for dd, v in zip(R["f_day"][m], zip(R["f_hdd"][m], R["f_cdd"][m]))}


def features_at(j, R, nkey="base"):
    """Inputs at decision time t = R['p_date'][j] from rows available at t. Returns (x [NF], u [H, NU]) or None."""
    t = R["p_date"][j]
    tt = pd.Timestamp(t)
    # ---- price: trade dates <= t
    np_ = np.searchsorted(R["p_avail"], t, side="right")   # published spot only
    lp = np.log(np.maximum(R["p"][:np_], 0.01))
    if len(lp) < 261:
        return None
    r = np.diff(lp)
    x = np.zeros(NF)
    x[0] = lp[-1] - lp[-250:].mean()
    x[1], x[2], x[3] = r[-1], lp[-1] - lp[-6], lp[-1] - lp[-21]
    x[4] = np.log(r[-5:].std() + 1e-3)
    x[5] = np.log(r[-20:].std() + 1e-3)
    x[6] = np.log(r[-60:].std() + 1e-3)
    x[7] = lp[-1] - lp[-61]
    x[8] = lp[-1]
    # ---- storage: reports released (avail) <= t
    sm = R["s_avail"] <= t
    if sm.sum() < 300:
        return None
    sw, sds, sdev, savg, slvl = R["s_week"][sm], R["s_ds"][sm], R["s_dev"][sm], R["s_avg5"][sm], R["s_lvl"][sm]
    if not np.isfinite(sdev[-1]):
        return None
    x[9] = sdev[-1] / 200
    x[10] = sds[-1] / 100
    x[11] = (sdev[-1] - sdev[-5]) / 100
    x[12] = (sds[-1] - savg[-1]) / 50
    last_av = R["s_avail"][sm][-1]
    x[13] = (t - last_av).astype(int) / 7
    x[14] = (last_av + np.timedelta64(7, "D") - t).astype(int) / 7
    nxt = sw[-1] + np.timedelta64(7, "D")
    vals = []
    for k in range(1, 6):
        c = nxt - np.timedelta64(364 * k, "D")
        mm = np.abs((sw - c).astype(int)) <= 3
        if mm.any():
            vals.append(np.nanmean(sds[mm]))
    avg5_next = float(np.mean(vals)) if vals else 0.0
    x[15] = avg5_next / 100
    x[16] = slvl[-1] / 1000
    # ---- observed weather: calendar days <= t-1
    nm = _normal(R, tt.year, nkey)
    om = R["o_date"] <= t - np.timedelta64(1, "D")
    od, oh, oc = R["o_date"][om][-7:], R["o_hdd"][om][-7:], R["o_cdd"][om][-7:]
    odoy = pd.DatetimeIndex(od).day_of_year.to_numpy()
    ha, ca = oh - nm[odoy, 0], oc - nm[odoy, 1]
    x[17], x[18] = ha[-1] / 10, ca[-1] / 10
    x[19], x[20] = ha.sum() / 30, ca.sum() / 30
    # ---- CPC forecasts: latest usable issue (never realized weather)
    iss, prev = _usable_issue(R, t)
    u = np.zeros((H, NU))
    days = [np.datetime64(t + np.timedelta64(k, "D"), "D") for k in range(1, H + 1)]
    if iss is not None:
        x[21] = min((t - iss).astype(int), 10) / 5
        cur = _fc(R, iss)
        pv = _fc(R, prev) if prev is not None else {}
        rh = rc = 0.0
        for k, dd in enumerate(days):
            if dd in cur:
                h_, c_ = cur[dd]
                dy = pd.Timestamp(dd).day_of_year
                u[k, 0], u[k, 1], u[k, 2] = (h_ - nm[dy, 0]) / 10, (c_ - nm[dy, 1]) / 10, 1.0
                u[k, 3], u[k, 4] = h_ / 30, c_ / 30
                if dd in pv:
                    rh += h_ - pv[dd][0]
                    rc += c_ - pv[dd][1]
        x[22], x[23] = rh / 10, rc / 10
    else:
        x[21] = 2.0
    for k, dd in enumerate(days):
        wd = pd.Timestamp(dd).weekday()
        u[k, 5], u[k, 6] = float(wd >= 5), float(wd == 3)
    dy = tt.day_of_year
    x[24], x[25] = np.sin(2 * np.pi * dy / 365.25), np.cos(2 * np.pi * dy / 365.25)
    if tt.weekday() < 5:
        x[26 + tt.weekday()] = 1.0
    x[31] = (t - R["p_date"][np_ - 1]).astype(int) / 7
    return x.astype(np.float32), u.astype(np.float32)


def targets_at(j, R, t_feats_ok=True):
    """labels for decision row j (future data: used only as training/scoring labels)."""
    t = R["p_date"][j]
    lp = np.log(np.maximum(R["p"], 0.01))
    n = len(lp)
    y = np.full(len(TARGETS), np.nan, np.float32)
    steps = np.zeros(len(TARGETS), np.int64)
    if j + 5 < n:
        r = np.diff(lp[j:j + 6])
        y[0], y[1], y[2] = r[0], r[1], lp[j + 5] - lp[j]
        y[3] = np.log(np.sqrt(np.mean(r ** 2)) + 1e-3)
        off = [(R["p_date"][j + k] - t).astype(int) for k in (1, 2, 5)]
        steps[0], steps[1], steps[2] = [min(max(o, 1), H) for o in off]
        steps[3] = steps[2]
    nx = np.where(R["s_avail"] > t)[0]
    if len(nx):
        i = nx[0]
        if np.isfinite(R["s_ds"][i]) and np.isfinite(R["s_avg5"][i]):
            y[4] = R["s_ds"][i] - R["s_avg5"][i]
            steps[4] = min(max((R["s_avail"][i] - t).astype(int), 1), H)
    # last calendar day whose data the label depends on (for fold embargo)
    last = R["p_date"][min(j + 5, n - 1)]
    if len(nx):
        last = max(last, R["s_avail"][nx[0]])
    return y, steps, last


def build(R=None, verbose=True):
    R = R or load_raw()
    js = np.where(R["p_date"] >= np.datetime64(FIRST_DECISION.date()))[0]
    X, U, Y, ST, D, LAST = [], [], [], [], [], []
    for j in js:
        fu = features_at(j, R)
        if fu is None:
            continue
        y, st, last = targets_at(j, R)
        X.append(fu[0]); U.append(fu[1]); Y.append(y); ST.append(st); D.append(R["p_date"][j]); LAST.append(last)
    out = dict(X=np.stack(X), U=np.stack(U), Y=np.stack(Y), ST=np.stack(ST), D=np.array(D), LAST=np.array(LAST),
               pins=json.dumps(R["pins"]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(PANEL, **out)
    if verbose:
        print(f"panel rows {len(D)} {D[0]}..{D[-1]}  X {out['X'].shape} U {out['U'].shape}")
    return out


def load_panel():
    z = np.load(PANEL, allow_pickle=False)
    return {k: z[k] for k in z.files}


def perturb_future(R, t, rng):
    """copy of R with every row whose availability is AFTER decision time t replaced by random values."""
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    m = Q["p_avail"] > t
    Q["p"][m] = rng.uniform(0.5, 20, m.sum())
    m = Q["s_avail"] > t
    for k in ["s_lvl", "s_ds", "s_avg5", "s_dev"]:
        Q[k][m] = rng.normal(0, 300, m.sum())
    m = Q["o_date"] > t - np.timedelta64(1, "D")
    Q["o_hdd"][m] = rng.uniform(0, 60, m.sum()); Q["o_cdd"][m] = rng.uniform(0, 30, m.sum())
    tc = close_utc(t)
    m = (Q["f_issue"] > t) | ((Q["f_issue"] == t) & (Q["f_lm"] > tc))
    Q["f_hdd"][m] = rng.uniform(0, 60, m.sum()); Q["f_cdd"][m] = rng.uniform(0, 30, m.sum())
    return Q


def window_at(j, R, L, nkey):
    """input window [L, NF] and u [H, NU] for decision row j, rebuilt from R (rows j-L+1 .. j)."""
    xs = []
    for i in range(j - L + 1, j + 1):
        fu = features_at(i, R, nkey)
        if fu is None:
            return None
        xs.append(fu[0])
        u = fu[1]
    return np.stack(xs), u


def row_index(R, t):
    return int(np.searchsorted(R["p_date"], np.datetime64(t, "D")))


def leaktest(n=40, seed=0):
    R = load_raw()
    rng = np.random.default_rng(seed)
    js = np.where((R["p_date"] >= np.datetime64("2016-01-01")) & (R["p_date"] <= np.datetime64("2025-10-31")))[0]
    bad = 0
    for j in rng.choice(js, n, replace=False):
        t = R["p_date"][j]
        a = features_at(j, R)
        Q = perturb_future(R, t, rng)
        _NCACHE.clear()
        b = features_at(j, Q, nkey="pert")
        _NCACHE.clear()
        same = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        bad += not same
        if not same:
            print("LEAK", t, np.where(a[0] != b[0])[0])
    # allowed perturbation: the latest PUBLISHED price at t must change features
    j = js[len(js) // 2]
    jl = np.searchsorted(R["p_avail"], R["p_date"][j], side="right") - 1
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    Q["p"][jl] *= 1.1
    ch = not np.array_equal(features_at(j, R)[0], features_at(j, Q)[0])
    print(f"feature leaktest: {n - bad}/{n} identical under future perturbation; allowed perturbation changes: {ch}")
    return bad == 0 and ch


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "leaktest":
        sys.exit(0 if leaktest() else 1)
