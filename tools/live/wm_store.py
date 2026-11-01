"""
Alma-Live Gas: WM feature/label layer ON TOP OF the data lead's Store (LRT-13 parity by construction).

Features are a pure function of ctx = Store().asof(common.decision_time(d)) -- `features_from_ctx(ctx)` -- so the
training panel, the paper/live predict() (ctx from Store.asof(t, live=True)) and the randomized-future test all run
the identical code on the identical records. Labels come only from Store.labels() / store.storage_labels() (never in
ctx): r1 = UNG close(d) -> close(d+1) (`ung_ret`), r5 = sum of ung_ret d..d+4, lrv5 = log of the sum of their squares (= log ung_rv5),
dsurp = storage_labels.dev of ctx['next_storage']['week'] (first-print change minus the store's 5-yr expectation;
the harness's stor_dev target).

    nice -n 10 .venv/bin/python tools/live/wm_store.py build      # -> runs/live/search/cache/panel_store.npz
    nice -n 10 .venv/bin/python tools/live/wm_store.py leaktest
"""
import os
import sys
import copy
import json
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402
import store as SM  # noqa: E402

CACHE = os.path.join(ROOT, "runs", "live", "search", "cache")
PANEL = os.path.join(CACHE, "panel_store.npz")

FIRST_DECISION = pd.Timestamp("2015-10-01")
LAST_DECISION = pd.Timestamp("2026-10-23")
DEV_END = pd.Timestamp("2024-06-30")
VAL_START, VAL_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-10-31")
EXPOSED_START = pd.Timestamp("2025-11-01")
FOLD_YEARS = [2020, 2021, 2022, 2023]
H = 7
NU = 7
FEATS = ["u_lp_m250", "u_r1", "u_r5", "u_r20", "u_lrv5", "u_lrv20", "u_lrv60", "u_lp_m60",
         "dev", "ds", "dev_chg4", "last_surp", "days_since_rep", "days_to_rep", "avg5_next", "lvl",
         "ohdd_a", "ocdd_a", "ohdd7_a", "ocdd7_a", "fc_age", "rev_hdd", "rev_cdd",
         "sin_doy", "cos_doy", "dow0", "dow1", "dow2", "dow3", "dow4", "eia_tomorrow", "p_age"]
NF = len(FEATS)
UFEATS = ["fhdd_a", "fcdd_a", "fmask", "fhdd", "fcdd", "weekend", "thursday"]
TARGETS = ["r1", "r5", "lrv5", "dsurp"]
PRIMARY = ["r1", "lrv5", "dsurp"]
NORMAL_YEARS = 10
_DAYS = None


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def trading_days():
    global _DAYS
    if _DAYS is None:
        _DAYS = pd.DatetimeIndex(CM.trading_days("2010-01-01", "2026-12-31"))
    return _DAYS


def _avg5(s, week):
    """mean of s over the same week (+-3 days) in each of the 5 previous years (NaN unless all 5 exist)."""
    vals = []
    for k in range(1, 6):
        c = week - pd.Timedelta(days=364 * k)
        w = s[(s.index >= c - pd.Timedelta(days=3)) & (s.index <= c + pd.Timedelta(days=3))]
        if len(w):
            vals.append(float(w.mean()))
    return float(np.mean(vals)) if len(vals) == 5 else np.nan


def _normals(obs, year):
    """causal degree-day normals by day-of-year: years [year-10, year-1] of the ctx's observed weather."""
    o = obs[(obs.index.year >= year - NORMAL_YEARS) & (obs.index.year < year)]
    doy = o.index.day_of_year.to_numpy()
    h, c = o["hdd"].to_numpy(float), o["cdd"].to_numpy(float)
    out = np.zeros((367, 2))
    for q in range(1, 367):
        dd = np.abs(doy - q)
        dd = np.minimum(dd, 366 - dd)
        s = dd <= 7
        out[q] = [np.nanmean(h[s]), np.nanmean(c[s])]
    return out


def features_from_ctx(ctx):
    """ctx = Store.asof(decision_time(d)) -> (x [NF], u [H, NU]) float32, or None if history is insufficient."""
    d = pd.Timestamp(ctx["day"])
    t = pd.Timestamp(ctx["t"])
    x = np.zeros(NF)
    # ---- UNG (split-adjusted as of t, bars through d-1)
    u_ = ctx["px"].get("UNG")
    if u_ is None or len(u_) < 300:
        return None
    lp = np.log(u_["close"].to_numpy(float))
    lr = np.diff(lp)
    x[0] = lp[-1] - lp[-250:].mean()
    x[1], x[2], x[3] = lr[-1], lp[-1] - lp[-6], lp[-1] - lp[-21]
    x[4], x[5], x[6] = [np.log(np.sqrt(np.mean(lr[-n:] ** 2)) + 1e-3) for n in (5, 20, 60)]
    x[7] = lp[-1] - lp[-61]
    x[31] = (d - u_.index[-1]).days / 7
    # ---- storage (store's visible vintage per week; printed change is first print)
    st = ctx["storage"]
    if len(st) < 300:
        return None
    lvl, chg = st["l48"].dropna(), st["l48.chg_printed"].dropna()
    W = lvl.index[-1]
    a5, a5p = _avg5(lvl.iloc[:-1], W), _avg5(lvl.iloc[:-5], lvl.index[-5])
    x[8] = (lvl.iloc[-1] - a5) / 200
    x[9] = chg.iloc[-1] / 100
    x[10] = ((lvl.iloc[-1] - a5) - (lvl.iloc[-5] - a5p)) / 100
    x[11] = (chg.iloc[-1] - _avg5(chg.iloc[:-1], chg.index[-1])) / 50
    ra = st["released_at"].dropna()
    x[12] = (t - ra.iloc[-1]).total_seconds() / (7 * 86400)
    x[13] = 1.0 - x[12]
    x[14] = ctx["next_storage"]["expectation_l48"] / 100
    x[15] = lvl.iloc[-1] / 1000
    if not np.all(np.isfinite(x[8:16])):
        return None
    # ---- observed weather (ctx: published <= t)
    ob = ctx["cpc_obs"]
    nm = _normals(ob, d.year)
    o7 = ob.iloc[-7:]
    odoy = o7.index.day_of_year.to_numpy()
    ha, ca = o7["hdd"].to_numpy(float) - nm[odoy, 0], o7["cdd"].to_numpy(float) - nm[odoy, 1]
    x[16], x[17], x[18], x[19] = ha[-1] / 10, ca[-1] / 10, ha.sum() / 30, ca.sum() / 30
    # ---- CPC forecasts as issued (latest and previous visible issue)
    fc = ctx["cpc_fcst"]
    u = np.zeros((H, NU))
    days = [d + pd.Timedelta(days=j) for j in range(1, H + 1)]
    if len(fc):
        iss_all = sorted(fc["issue"].unique())
        iss = iss_all[-1]
        x[20] = min((d - pd.Timestamp(iss)).days, 10) / 5
        cur = fc[fc["issue"] == iss].set_index("day")
        pv = fc[fc["issue"] == iss_all[-2]].set_index("day") if len(iss_all) > 1 else None
        rh = rc = 0.0
        for j, dd in enumerate(days):
            if dd in cur.index:
                h_, c_ = float(cur.at[dd, "hdd"]), float(cur.at[dd, "cdd"])
                dy = dd.day_of_year
                u[j, 0], u[j, 1], u[j, 2] = (h_ - nm[dy, 0]) / 10, (c_ - nm[dy, 1]) / 10, 1.0
                u[j, 3], u[j, 4] = h_ / 30, c_ / 30
                if pv is not None and dd in pv.index:
                    rh += h_ - float(pv.at[dd, "hdd"]); rc += c_ - float(pv.at[dd, "cdd"])
        x[21], x[22] = rh / 10, rc / 10
    else:
        x[20] = 2.0
    for j, dd in enumerate(days):
        u[j, 5], u[j, 6] = float(dd.weekday() >= 5), float(dd.weekday() == 3)
    # ---- calendar
    dy = d.day_of_year
    x[23], x[24] = np.sin(2 * np.pi * dy / 365.25), np.cos(2 * np.pi * dy / 365.25)
    x[25 + d.weekday()] = 1.0
    td = trading_days()
    nd = td[td.searchsorted(d, side="right")]
    x[30] = float(nd.weekday() == 3)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
        return None
    return x.astype(np.float32), u.astype(np.float32)


def steps_for(d, ctx):
    """calendar readout steps (calendar knowledge only): next trading day, 5th trading day, nominal Thursday release
    of ctx['next_storage']['week'] (week end + 6 days)."""
    td = trading_days()
    i = td.searchsorted(d, side="right")
    s1 = min(max((td[i] - d).days, 1), H)
    s5 = min(max((td[i + 4] - d).days, 1), H)
    nw = ctx["next_storage"]["week"]
    s4 = min(max((nw + pd.Timedelta(days=6) - d).days, 1), H) if nw is not None else H
    return np.array([s1, s5, s5, s4], np.int64)


def targets_for(d, ctx, L, sl):
    y = np.full(len(TARGETS), np.nan, np.float32)
    last = d
    if d in L.index:
        i = L.index.get_loc(d)
        r = L["ung_ret"].iloc[i:i + 5].to_numpy(float)
        if len(r) == 5 and np.all(np.isfinite(r)):
            y[0], y[1] = r[0], r.sum()
            y[2] = np.log(np.sum(r ** 2) + 1e-10)          # = log(Store ung_rv5), the harness rv5 target
            last = pd.Timestamp(L["available_at"].iloc[i + 4]).tz_convert(CM.ET).tz_localize(None).normalize()
    nw = ctx["next_storage"]["week"]
    if nw is not None and nw in sl.index and np.isfinite(sl.at[nw, "dev"]):
        y[3] = sl.at[nw, "dev"]
        last = max(last, pd.Timestamp(sl.at[nw, "released_at"]).tz_convert(CM.ET).tz_localize(None).normalize())
    return y, last


def build(verbose=True):
    S = SM.Store()
    L, sl = S.labels(), SM.storage_labels(S)
    L.index = pd.DatetimeIndex(pd.to_datetime(L.index))
    sl.index = pd.DatetimeIndex(pd.to_datetime(sl.index))
    td = trading_days()
    days = td[(td >= FIRST_DECISION) & (td <= LAST_DECISION)]
    X, U, Y, ST, D, LAST = [], [], [], [], [], []
    for d in days:
        ctx = S.asof(CM.decision_time(d), with_features=False)
        fu = features_from_ctx(ctx)
        if fu is None:
            continue
        y, last = targets_for(d, ctx, L, sl)
        X.append(fu[0]); U.append(fu[1]); Y.append(y); ST.append(steps_for(d, ctx))
        D.append(np.datetime64(d.date())); LAST.append(np.datetime64(last.date()))
    pins = {s: sha256(os.path.join(CM.PIT, f"{s}.csv")) for s in SM.SOURCES}
    out = dict(X=np.stack(X), U=np.stack(U), Y=np.stack(Y), ST=np.stack(ST), D=np.array(D, dtype="datetime64[D]"),
               LAST=np.array(LAST, dtype="datetime64[D]"), pins=json.dumps(pins))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(PANEL, **out)
    if verbose:
        print(f"panel rows {len(D)} {D[0]}..{D[-1]} X {out['X'].shape} nanY {np.isnan(out['Y']).mean(0).round(3)}")
    return out


def load_panel():
    z = np.load(PANEL, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ randomized-future test hooks (run.py)
def load_raw():
    return SM.Store()


def row_index(R, t):
    return pd.Timestamp(t)


def perturb_future(S, d, rng):
    """copy of the Store where every record published after decision_time(d) gets a random value."""
    t = pd.Timestamp(CM.decision_time(pd.Timestamp(d))).value
    Q = copy.copy(S)
    Q.raw = {}
    for s, df in S.raw.items():
        df = df.copy()
        df["value"] = df["value"].astype(float)
        m = df["pa"].to_numpy() > t
        df.loc[m, "value"] = rng.uniform(0.1, 500, m.sum())
        Q.raw[s] = df
    return Q


def window_at(d, S, L, nkey=None):
    d = pd.Timestamp(d)
    td = trading_days()
    k = td.get_loc(d)
    xs = []
    for i in range(k - L + 1, k + 1):
        fu = features_from_ctx(S.asof(CM.decision_time(td[i]), with_features=False))
        if fu is None:
            return None
        xs.append(fu[0]); u = fu[1]
    return np.stack(xs), u


def leaktest(n=30, seed=0):
    S = SM.Store()
    rng = np.random.default_rng(seed)
    P = load_panel()
    bad = 0
    for r in rng.choice(len(P["D"]), n, replace=False):
        d = pd.Timestamp(P["D"][r])
        a = features_from_ctx(S.asof(CM.decision_time(d), with_features=False))
        Q = perturb_future(S, d, rng)
        b = features_from_ctx(Q.asof(CM.decision_time(d), with_features=False))
        ok = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[0], P["X"][r])
        bad += not ok
        if not ok:
            print("LEAK/MISMATCH", d.date(), np.where(a[0] != b[0])[0])
    print(f"store feature leaktest: {n - bad}/{n} identical under randomized-future perturbation (and == panel)")
    return bad == 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "leaktest":
        sys.exit(0 if leaktest() else 1)
