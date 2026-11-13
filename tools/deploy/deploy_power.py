"""Deploy the frozen German day-ahead power world model (wm_power) on Oct 2026 and Nov 2026. Inference only.

    .venv/bin/python tools/deploy/deploy_power.py [--models ...]

Delivery day D (a window holds D when D is in it). Information set = the project harness (tools/power/data.py
build_context): prices of days <= D-1, fundamentals of days <= D-2, context 736 days (as in TEST). Scores per day:
QS = pinball over the 9 levels 0.1..0.9 averaged over the 24 hours (EUR/MWh); MAE of the median; P2 = hourly direction
of the median vs the same hour of D-1 (ties excluded); P1 = daily base-price direction; cov80.
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
sys.path.insert(0, os.path.join(ROOT, "tools", "power"))
CTX_DAYS = 736
CANDS = ["wm_power", "wm_power_blend"]
DEFAULT = ("wm_power,wm_power_direct,wm_power_blend,lear,lear_nocov,arx,naive_epf,naive_w,naive_d,"
           "chronos2,chronos2_cov,chronos_bolt_base,tirex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    a = ap.parse_args()
    C.no_training()
    watch = C.ParamWatch()
    C.track_torch_modules(watch)
    C.ensure_ckpts()
    pins = C.verify_checkpoints("power")
    import data as D
    import models as M

    dates, P, Cv, flags = D.panel()
    lo = np.datetime64(C.FIRST.date(), "D")
    idx = np.array([i for i, d in enumerate(dates) if d >= lo and np.isfinite(P[i]).sum() >= 23])
    ctxs = [D.build_context(int(i), CTX_DAYS, dates, P, Cv, flags) for i in idx]
    cur, prev = P[idx], P[idx - 1]
    preds, meta = {}, {}
    for mn in a.models.split(","):
        t0 = time.time()
        try:
            p = M.get(mn)
            kw = {}
            if getattr(p, "wants_dates", False):                                # as tools/power/run_eval.py
                kw["dates"] = [{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in ctxs]
            q = np.asarray(p.batch([c["prices"] for c in ctxs], [c["covs"] for c in ctxs], 24, list(C.LV9), **kw),
                           float)
            meta[mn] = {k: v for k, v in dict(getattr(p, "meta", {})).items() if isinstance(v, (str, int, float, list, dict))}
        except Exception as e:  # noqa: BLE001
            print(f"{mn}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            meta[mn] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        meta[mn]["seconds"] = round(time.time() - t0, 1)
        preds[mn] = q
        print(f"{mn}: {q.shape} {meta[mn]['seconds']}s nonfinite={int((~np.isfinite(q)).sum())}", flush=True)

    res = {"project": "power", "target": "DE-LU day-ahead price, 24 hourly slots (EUR/MWh)",
           "data_cutoff_delivery_day": str(dates[idx[-1]]), "scored_utc": C.stamp(), "git": C.git_state(),
           "checkpoints": pins, "candidates": CANDS, "models": meta, "windows": {}}
    for wn, (wlo, whi) in C.WINDOWS.items():
        sel = (dates[idx] >= np.datetime64(wlo.date(), "D")) & (dates[idx] <= np.datetime64(whi.date(), "D"))
        if not sel.any():
            res["windows"][wn] = {"n": 0}
            continue
        y, yp = cur[sel], prev[sel]
        per = {}
        for mn, q in preds.items():
            qq = q[sel]                                           # [n, 9, 24]
            med = qq[:, 4]
            d = y[:, None] - qq
            m2 = np.isfinite(y) & np.isfinite(yp) & (y != yp)
            base_y, base_p, base_m = np.nanmean(y, 1), np.nanmean(yp, 1), np.nanmean(med, 1)
            per[mn] = {"qs": np.nanmean(np.maximum(C.LV9[None, :, None] * d, (C.LV9[None, :, None] - 1) * d), (1, 2)),
                       "abs_err": np.nanmean(np.abs(med - y), 1),
                       "p2_hit": np.nanmean(np.where(m2, (med > yp) == (y > yp), np.nan), 1),
                       "p1_hit": np.where(base_y != base_p, ((base_m > base_p) == (base_y > base_p)).astype(float), np.nan),
                       "cov80": np.nanmean((y >= qq[:, 0]) & (y <= qq[:, -1]), 1)}
        res["windows"][wn] = {"n": int(sel.sum()), "first": str(dates[idx][sel][0]), "last": str(dates[idx][sel][-1]),
                              "by_candidate": {c: {"qs": C.compare(per, c, "qs", block=7),
                                                   "p2_hit": C.compare(per, c, "p2_hit", block=7, higher_better=True)}
                                               for c in CANDS if c in per}}
    path, sh = C.save_preds("power", "power_preds", delivery_days=dates[idx].astype(str), levels=C.LV9,
                            **{f"q_{k}": q for k, q in preds.items()})
    # realised prices are market data (not redistributed): kept in a local, git-ignored file for the audit
    C.save_preds("power", "power_LABELS_local", delivery_days=dates[idx].astype(str), actual=cur, prev_day=prev)
    res["preds_file"], res["preds_sha256"] = path, sh
    res["param_watch"] = watch.check()
    assert all(r["unchanged"] for r in res["param_watch"]), "model weights changed during deployment"
    res["checkpoints_after"] = C.verify_checkpoints("power")
    print("->", C.write_json("power", "results.json", res))


if __name__ == "__main__":
    main()
