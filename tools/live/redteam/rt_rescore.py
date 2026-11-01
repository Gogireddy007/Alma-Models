"""Red-team independent re-derivation of the loop scores (S_F, rel_*) from raw PIT records.

1. Labels rebuilt from data/live/pit (no wm_store code): UNG split-adjusted closes -> r1 = log C(d+1)/C(d),
   r5, lrv5 = log(sqrt(mean r_{d+1..d+5}^2) + 1e-3); dsurp = first print of the next L48 printed change published
   after decision time t(d) (15:30 ET) minus the mean printed change of the same week (+-3 d) in the 5 previous years
   (first prints with obs < week). Compared with the panel Y.
2. Folds rebuilt independently (train: every label date < Jul 1 Y; test: Jul 1 Y .. Jun 30 Y+1 and label end
   <= 2024-06-30), standardisation from train labels, pinball over 0.05..0.95; S_F = mean over folds and the primary
   targets {r1, lrv5, dsurp} of QS(model)/QS(ridge ref), re-computed for every run with saved preds.

    nice -n 10 .venv/bin/python tools/live/redteam/rt_rescore.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rt_ref as RR  # noqa: E402

ROOT = RR.ROOT
PIT = os.path.join(ROOT, "data/live/pit")
OUT = os.path.join(ROOT, "runs/live/search")
QL = np.round(np.arange(0.05, 0.951, 0.05), 2)
TARGETS = ["r1", "r5", "lrv5", "dsurp"]
PRIMARY = ["r1", "lrv5", "dsurp"]
YEARS = [2020, 2021, 2022, 2023]


def ung():
    px = pd.read_csv(os.path.join(PIT, "px.csv"), dtype=str)
    c = px[px.series == "UNG.close"].drop_duplicates("obs", keep="last")
    s = pd.Series(c.value.astype(float).values, index=pd.to_datetime(c.obs)).sort_index()
    sp = px[px.series == "UNG.split"]
    for d, f in zip(pd.to_datetime(sp.obs), sp.value.astype(float)):
        s[s.index < d] = s[s.index < d] / f
    return s


def storage():
    st = pd.read_csv(os.path.join(PIT, "storage.csv"), dtype=str)
    ch = st[(st.series == "l48.chg_printed") & st.vintage.isin(["first_release", "current_backfill"])].copy()
    ch["obs"] = pd.to_datetime(ch.obs)
    ch["pa"] = pd.to_datetime(ch.published_at, utc=True)
    ch["v"] = ch.value.astype(float)
    return ch.sort_values("pa").drop_duplicates("obs", keep="first").reset_index(drop=True)


def labels(D):
    c = ung()
    lc = np.log(c)
    idx = {d: i for i, d in enumerate(c.index)}
    ch = storage()
    out = np.full((len(D), 4), np.nan)
    last = []
    for n, d in enumerate(pd.DatetimeIndex(D)):
        i = idx.get(d)
        L = d
        if i is not None and i + 5 < len(c):
            r = np.diff(lc.values[i:i + 6])
            out[n, 0], out[n, 1] = r[0], lc.values[i + 5] - lc.values[i]
            out[n, 2] = np.log(np.sum(r ** 2) + 1e-10)   # A3 redefinition (log rv5)
            L = c.index[i + 5]
        t = RR._et(d, 15, 30)
        fut = ch[ch.pa > t]
        if len(fut):
            j = fut.index[0]
            wk = ch.at[j, "obs"]
            past = ch[ch.obs < wk]
            vals = []
            for k in range(1, 6):
                m = (past.obs - (wk - pd.Timedelta(days=364 * k))).abs() <= pd.Timedelta(days=3)
                if m.any():
                    vals.append(past.v[m].mean())
            if len(vals) == 5:
                out[n, 3] = ch.at[j, "v"] - np.mean(vals)
                L = max(L, ch.at[j, "pa"].tz_convert(RR.ET).normalize().tz_localize(None))
        last.append(L)
    return out, pd.DatetimeIndex(last)


def pin(q, y):
    e = y[..., None] - q
    return np.where(np.isfinite(y), np.maximum(QL * e, (QL - 1) * e).mean(-1), np.nan)


def main():
    z = np.load(os.path.join(OUT, "cache", "panel_store.npz"))
    D = pd.DatetimeIndex(z["D"].astype("datetime64[D]"))
    Y, LAST = labels(D)
    PY = z["Y"]
    for j, t in enumerate(TARGETS):
        a, b = Y[:, j], PY[:, j]
        m = np.isfinite(a) & np.isfinite(b)
        print(f"label {t}: both finite {m.sum()}, max |diff| {np.abs(a[m] - b[m]).max():.2e}, "
              f"finite mismatch {int((np.isfinite(a) != np.isfinite(b)).sum())}")
    print("LAST mismatch days:", int((LAST.values.astype('datetime64[D]') != z["LAST"].astype('datetime64[D]')).sum()))
    L = 20
    folds = {}
    for yr in YEARS:
        s, e = pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr + 1}-06-30")
        ok = np.arange(len(D)) >= L - 1
        tr = np.where(ok & (LAST < s) & (D < s))[0]
        te = np.where(ok & (D >= s) & (D <= e) & (LAST <= pd.Timestamp("2024-06-30")))[0]
        folds[yr] = (tr, te)
    pd_ = os.path.join(OUT, "preds")
    if not os.path.exists(os.path.join(pd_, "refs.npz")):
        print("no refs.npz yet")
        return
    R = np.load(os.path.join(pd_, "refs.npz"))
    for yr, (tr, te) in folds.items():
        te_ref = R[f"{yr}_te"]
        if not np.array_equal(te_ref, te):
            print(f"fold {yr}: test rows differ from refs.npz ({len(te_ref)} vs {len(te)})")
    ym, ys = {}, {}
    for yr, (tr, te) in folds.items():
        ym[yr], ys[yr] = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0) + 1e-6
    prim = [TARGETS.index(t) for t in PRIMARY]

    def rel(Q, ref="linqr"):
        rs = []
        for yr, (tr, te) in folds.items():
            y = (Y[te] - ym[yr]) / ys[yr]
            a = np.nanmean(pin(Q[yr], y), 0)
            b = np.nanmean(pin(R[f"{yr}_{ref}"], y), 0)
            rs.append(a / b)
        rs = np.mean(rs, 0)
        return float(rs[prim].mean()), rs
    rows = list(csv.DictReader(open(os.path.join(OUT, "results.tsv")), delimiter="\t")) \
        if os.path.exists(os.path.join(OUT, "results.tsv")) else []
    nbad = 0
    for r in rows:
        f = os.path.join(pd_, f"{r['run_id']}.npz")
        if r.get("kind") == "ref":
            s, _ = rel({yr: R[f"{yr}_{r['family']}"] for yr in folds})
        elif os.path.exists(f):
            P = np.load(f)
            s, _ = rel({yr: P[str(yr)] for yr in folds})
        else:
            continue
        rep = float(r["S_F"]) if r.get("S_F") else np.nan
        flag = abs(s - rep) > 6e-4
        nbad += flag
        print(f"{r['run_id']:32s} {r.get('kind',''):8s} reported S_F {rep:.4f}  re-derived {s:.4f} {'MISMATCH' if flag else ''}")
    kinds = pd.Series([r.get("kind", "") for r in rows]).value_counts().to_dict()
    print(f"rows {len(rows)} by kind {kinds}; K (non-ref rows) = {sum(v for k, v in kinds.items() if k != 'ref')}; "
          f"mismatches {nbad}")


if __name__ == "__main__":
    main()
