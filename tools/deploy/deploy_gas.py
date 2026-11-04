"""Deploy the frozen US gas-storage world model (wm_gas, wm_gas_ens) on Oct 2026 and Nov 2026. Inference only.

    .venv/bin/python tools/deploy/deploy_gas.py [--models ...]

Target: weekly EIA Lower-48 storage change (Bcf), from Thursday and Friday origins (tools/gas/storage.py).
A target week belongs to a window when its EIA RELEASE date is in it.
  1. reproduction: the 52 TEST weeks are re-forecast and compared with the audit's cached forecasts
     (audit/gas_preds_test.npy). Identical forecasts prove the rebuilt inputs equal the audited ones.
  2. October 2026: the releases of 2026-10-01 .. 2026-10-29 (5 weeks), scored against every competitor.
  3. November 2026: no release has happened yet (first one 2026-11-05) -> an EX-ANTE forecast of the week ending
     2026-10-30 is written and sha256-sealed, to be scored by tools/deploy/score_exante.py after the release.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, os.path.join(ROOT, "tools", "gas"))
CANDS = ["wm_gas", "wm_gas_ens"]
DEFAULT = ("wm_gas_ens,wm_gas,wm_gas_direct,st_ols_wx,st_ols_wx_nolag,st_arima_dev,st_climatology,st_persistence,"
           "st_seasonal_naive,chronos2,chronos2_cov,chronos_bolt_base,tirex")
REPRO = ["wm_gas", "wm_gas_direct", "st_ols_wx", "st_persistence", "st_climatology"]


def extended_table(S, D):
    """Weekly table plus one row for the next, not yet released week (all its storage values NaN)."""
    lvl, st = S.levels()
    nxt = lvl.index.max() + pd.Timedelta(days=7)
    lvl2 = pd.concat([lvl, pd.DataFrame(np.nan, index=[nxt], columns=lvl.columns)])
    lvl2.index.name = lvl.index.name
    row = {c: np.nan for c in st.columns}
    row.update(week=nxt, vintage="NOT_RELEASED")
    st2 = pd.concat([st, pd.DataFrame([row])], ignore_index=True)
    st2["avail"], st2["avail_rule"] = D.release_dates(st2["week"], D.parse_schedule())
    w2 = S.weekly_from(lvl2, st2)
    w2 = pd.read_csv(pd.io.common.StringIO(w2.to_csv(index=False, date_format="%Y-%m-%d")), parse_dates=["week", "avail"])
    return w2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    a = ap.parse_args()
    C.no_training()
    watch = C.ParamWatch()
    C.track_torch_modules(watch)
    C.ensure_ckpts()
    pins = C.verify_checkpoints("gas")
    import data as D
    import storage as S
    import models as M

    w, o, f = S.load()
    res = {"project": "gas", "target": "EIA weekly storage change, Lower 48 (Bcf)", "scored_utc": C.stamp(),
           "git": C.git_state(), "checkpoints": pins, "candidates": CANDS, "last_released_week":
           str(w.week.iloc[-1].date()), "last_release": str(w.avail.iloc[-1].date()), "models": {}, "windows": {}}
    names = a.models.split(",")
    models = {}
    for mn in names:
        try:
            models[mn] = M.get(mn)
            res["models"][mn] = {k: v for k, v in dict(getattr(models[mn], "meta", {}) or {}).items()
                                 if isinstance(v, (str, int, float, list, dict))}
        except Exception as e:  # noqa: BLE001
            print(f"{mn}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            res["models"][mn] = {"skipped": f"{type(e).__name__}: {e}"}

    # ---- 1. reproduction of the audited TEST forecasts
    aud = np.load(os.path.join(ROOT, "audit", "gas_preds_test.npy"), allow_pickle=True).item()
    ks_test = S.target_weeks("test", w)
    rep = {}
    for kind in S.ORIGINS:
        ctxs = [S.build_context(int(k), kind, w, o, f) for k in ks_test]
        for mn in REPRO:
            if mn in models and f"{kind}|{mn}" in aud:
                q = np.asarray(models[mn].storage(ctxs, C.QS13, "l48"), float)
                ref = np.asarray(aud[f"{kind}|{mn}"], float)
                rep[f"{kind}|{mn}"] = {"n": int(len(ks_test)), "max_abs_diff_bcf": float(np.nanmax(np.abs(q - ref)))}
    res["reproduction_vs_audit"] = rep
    print("reproduction:", rep, flush=True)

    # ---- 2. scored windows (release date in window)
    QS = C.QS13
    pred_store = {}
    for wn, (lo, hi) in C.WINDOWS.items():
        ks = np.where((w.avail >= lo) & (w.avail <= hi) & w.avg5_l48.notna() & w.ds_l48.notna())[0]
        if not len(ks):
            res["windows"][wn] = {"n_weeks": 0, "note": "no EIA release in this window yet; see ex_ante"}
            continue
        ds, a5 = w.ds_l48.to_numpy(float)[ks], w.avg5_l48.to_numpy(float)[ks]
        res["windows"][wn] = {"n_weeks": int(len(ks)), "weeks": [str(w.week.iloc[k].date()) for k in ks],
                              "releases": [str(w.avail.iloc[k].date()) for k in ks], "actual_bcf": ds.tolist(),
                              "kinds": {}}
        for kind in S.ORIGINS:
            ctxs = [S.build_context(int(k), kind, w, o, f) for k in ks]
            per = {}
            for mn, p in models.items():
                q = np.asarray(p.storage(ctxs, QS, "l48"), float)
                pred_store[f"{wn}|{kind}|{mn}"] = q
                d = ds[:, None] - q[:, 2:11]
                per[mn] = {"abs_err": np.abs(q[:, 6] - ds), "qs": np.maximum(C.LV9 * d, (C.LV9 - 1) * d).mean(1),
                           "hit": ((q[:, 6] > a5) == (ds > a5)).astype(float),
                           "cov80": ((ds >= q[:, 2]) & (ds <= q[:, 10])).astype(float)}
            res["windows"][wn]["kinds"][kind] = {
                "median_forecast_bcf": {mn: pred_store[f"{wn}|{kind}|{mn}"][:, 6].tolist() for mn in models},
                "by_candidate": {c: {"abs_err": C.compare(per, c, "abs_err", block=1),
                                     "qs": C.compare(per, c, "qs", block=1)} for c in CANDS if c in per}}

    # ---- 3. ex-ante forecast for the next release
    w2 = extended_table(S, D)
    same = all(np.allclose(w2[c].to_numpy(float)[:len(w)], w[c].to_numpy(float), equal_nan=True)
               for c in w.columns if c.startswith(("ds_", "avg5_", "lvl_")))
    assert same, "extended table changed released weeks"
    k = len(w2) - 1
    assert np.isnan(w2.ds_l48.iloc[k]), "ex-ante target must not be known"
    ex = {"target_week_ending": str(w2.week.iloc[k].date()), "scheduled_release": str(w2.avail.iloc[k].date()),
          "avg5_l48_bcf": float(w2.avg5_l48.iloc[k]), "forecasts": {}}
    for kind in S.ORIGINS:
        odt = S.origin_dt(w2, k, kind)
        if odt.tz_convert("UTC") > pd.Timestamp.now(tz="UTC"):
            ex["forecasts"][kind] = {"origin": str(odt), "skipped": "origin is in the future"}
            continue
        ctx = S.build_context(k, kind, w2, o, f)
        ex["forecasts"][kind] = {"origin": str(odt), "wx_issue": ctx["wx_issue"], "wx_src": ctx["wx_src"].tolist(),
                                 "models": {}}
        for mn, p in models.items():
            q = np.asarray(p.storage([ctx], QS, "l48"), float)[0]
            pred_store[f"exante|{kind}|{mn}"] = q[None]
            ex["forecasts"][kind]["models"][mn] = {"median": float(q[6]), "q10": float(q[2]), "q90": float(q[10])}
    res["ex_ante"] = ex
    path, sh = C.save_preds("gas", "gas_preds", quantiles=np.array(QS), **{k.replace("|", "__"): v for k, v in pred_store.items()})
    res["preds_file"], res["preds_sha256"] = path, sh
    res["param_watch"] = watch.check()
    assert all(r["unchanged"] for r in res["param_watch"]), "model weights changed during deployment"
    res["checkpoints_after"] = C.verify_checkpoints("gas")
    print("->", C.write_json("gas", "results.json", res))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f}s")
