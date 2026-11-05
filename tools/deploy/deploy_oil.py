"""Deploy the frozen oil world model (wm_sv) on Oct 2026 and Nov 2026 WTI, against its competitors. Inference only.

    .venv/bin/python tools/deploy/deploy_oil.py [--models wm_sv,wm_sv_direct,rw_gauss,ewma94,auto_arima,...]

Origin i: context = WTI levels strictly before observation i (CTX=1024, as in TEST); target at horizon h = obs i+h-1.
A forecast belongs to a window when its TARGET date is in it. Scores: pinball over 9 quantiles on log price (as in
the audit), |median log error|, direction of the median vs the last observed price, 80% coverage.
Also writes an ex-ante forecast from the last observed day (for targets not yet published), sha256-stamped.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, os.path.join(ROOT, "tools", "commodity"))
CTX, HS, H = 1024, (1, 5, 20), 20
CAND = "wm_sv"
DEFAULT = "wm_sv,wm_sv_direct,rw_gauss,ewma94,auto_arima,auto_ets,chronos2,chronos_bolt_base,tirex"


def load_wti():
    df = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    return pd.to_datetime(df["date"]).to_numpy(), df["v"].to_numpy(float)


def ewma94(ctxs, h, qs):
    out = []
    for c in ctxs:
        lp = np.log(np.asarray(c)[np.asarray(c) > 0])
        r = np.diff(lp)
        w = 0.94 ** np.arange(len(r))[::-1]
        s = np.sqrt(np.sum(w * r ** 2) / w.sum())
        out.append(np.exp(lp[-1] + norm.ppf(qs)[:, None] * s * np.sqrt(np.arange(1, h + 1))[None]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    a = ap.parse_args()
    C.no_training()
    watch = C.ParamWatch()
    C.track_torch_modules(watch)
    C.ensure_ckpts()
    pins = C.verify_checkpoints("oil")
    import models as M

    d, v = load_wti()
    cutoff = pd.Timestamp(d[-1])
    first = np.datetime64(C.FIRST)
    # every origin whose h=1..20 targets can touch the windows, plus one ex-ante origin (index len(v))
    oi = np.arange(np.searchsorted(d, first) - H, len(v) + 1)
    ctxs = [v[max(0, i - CTX):i] for i in oi]
    preds, meta = {}, {}
    for mn in a.models.split(","):
        t0 = time.time()
        try:
            if mn == "ewma94":
                q = ewma94(ctxs, H, np.array(C.QS13))
                meta[mn] = {"kind": "baseline", "def": "Gaussian RW in log price, EWMA(0.94) vol (audit oil_check.py)"}
            else:
                p = M.get(mn)
                q = np.asarray(p.batch(ctxs, H, C.QS13), float)
                meta[mn] = dict(getattr(p, "meta", {}))
        except Exception as e:  # noqa: BLE001  (a competitor that is not installed is reported, not hidden)
            print(f"{mn}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            meta[mn] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        meta[mn]["seconds"] = round(time.time() - t0, 1)
        preds[mn] = q
        print(f"{mn}: {q.shape} {meta[mn]['seconds']}s", flush=True)

    # ---- scoring
    res = {"project": "oil", "target": "WTI spot (FRED DCOILWTICO), USD/bbl", "data_cutoff": str(cutoff.date()),
           "scored_utc": C.stamp(), "git": C.git_state(), "checkpoints": pins, "candidate": CAND,
           "models": meta, "windows": {}}
    for wn, (lo, hi) in C.WINDOWS.items():
        res["windows"][wn] = {}
        for h in HS:
            tgt = oi + h - 1
            ok = (tgt < len(v))
            tdate = np.where(ok, d[np.minimum(tgt, len(v) - 1)], np.datetime64("NaT"))
            sel = ok & (tdate >= np.datetime64(lo)) & (tdate <= np.datetime64(hi))
            if not sel.any():
                res["windows"][wn][f"h{h}"] = {"n": 0}
                continue
            y = np.log(v[tgt[sel]])
            last = np.log(v[oi[sel] - 1])
            per = {}
            for mn, q in preds.items():
                lq = np.log(np.maximum(q[sel][:, :, h - 1], 1e-9))
                q9 = lq[:, C.I9]
                med = lq[:, 6]
                per[mn] = {"qs": C.pinball9(y, q9), "abs_err": np.abs(med - y),
                           "hit": np.where(y != last, ((med > last) == (y > last)).astype(float), np.nan),
                           "cov80": ((y >= lq[:, 2]) & (y <= lq[:, 10])).astype(float)}
            s = C.compare(per, CAND, "qs", block=1, overlap=h)
            for mn in s:
                s[mn]["skill_vs_rw_gauss"] = (1 - s[mn]["qs"] / s["rw_gauss"]["qs"]) if "rw_gauss" in s else None
            res["windows"][wn][f"h{h}"] = {"n": int(sel.sum()), "first_target": str(pd.Timestamp(tdate[sel][0]).date()),
                                           "last_target": str(pd.Timestamp(tdate[sel][-1]).date()), "models": s}
    # ---- ex-ante forecast (targets after the data cutoff; cannot be scored yet)
    j = len(oi) - 1
    assert oi[j] == len(v)
    res["ex_ante"] = {"origin_after": str(cutoff.date()), "last_price": float(v[-1]),
                      "note": "forecast of the next 1..20 published WTI prints after the cutoff; sealed by sha256",
                      "median_h1_h5_h20": {mn: [float(q[j, 6, h - 1]) for h in HS] for mn, q in preds.items()},
                      "q10_q90_h5": {mn: [float(q[j, 2, 4]), float(q[j, 10, 4])] for mn, q in preds.items()}}
    path, sh = C.save_preds("oil", "wti_preds", origins=oi, origin_dates=np.array([str(pd.Timestamp(d[i - 1]).date())
                            for i in oi]), quantiles=np.array(C.QS13),
                            **{f"q_{k}": q for k, q in preds.items()})
    res["preds_file"], res["preds_sha256"] = path, sh
    res["param_watch"] = watch.check()
    assert all(r["unchanged"] for r in res["param_watch"]), "model weights changed during deployment"
    res["checkpoints_after"] = C.verify_checkpoints("oil")
    out = C.write_json("oil", "results.json", res)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
