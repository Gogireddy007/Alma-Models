"""Deploy the frozen crypto-volatility world model (wm_fin) on Oct 2026 and Nov 2026. Inference only.

    ALMA_FIN_TEST_GO=1 .venv/bin/python tools/deploy/deploy_fin.py --variant corrected [--models ...]

Decision day D (00:00 UTC), 24 Binance USDT-M perps (16 training + 8 held-out coins). Target: log RV(D) of 5-minute
returns (>= 274 bars). P1 = direction of the median vs the reference mean(log RV(D-7..D-1)) (the median rule of the
forward test); QS = pinball over the 9 levels 0.1..0.9 of log RV. Units: day (24 coins pooled), 7-day block bootstrap.

--variant labels which processed data in data/fin/*.parquet is being scored (their sha256 is recorded):
  pinned     = rebuilt from exactly the raw files of the original data/fin/manifest.json. October funding rates were
               not yet published when that data was frozen, so funding = 0 for every October day, and the 24 daily
               kline files of 2026-10-28 were 404 (that day has no row). This is what TEST saw.
  corrected  = adds the October monthly klines and funding files published on 2026-11-01/02 and the daily klines of
               2026-10-31 .. 2026-11-03. Funding for November is not published yet (monthly archive), so it is 0 there.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, os.path.join(ROOT, "tools", "fin"))
CANDS = ["wm_fin", "wm_fin_blend"]
DEFAULT = ("wm_fin,wm_direct,wm_fin_blend,har_intra_dow,har_dow,lgbm_dow,har,har_roll,lgbm,lgbm_asset,rv_persist,"
           "garch,chronos2,chronos_bolt_base,tirex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["pinned", "corrected"])
    ap.add_argument("--models", default=DEFAULT)
    a = ap.parse_args()
    os.environ["ALMA_FIN_TEST_GO"] = "1"           # rows dated >= 2025-11-01 live in *_SEALED.parquet
    C.no_training()
    watch = C.ParamWatch()
    C.track_torch_modules(watch)
    pins = C.verify_checkpoints("fin")
    data_sha = {os.path.relpath(f, ROOT): C.sha_file(f) for f in sorted(glob.glob(os.path.join(ROOT, "data/fin/*.parquet")))}
    import harness as H
    import models as M

    P = H.Panel("crypto", allow_test=True)
    lo = np.datetime64(C.FIRST.date(), "ns")
    days = np.nonzero(P.days >= lo)[0]
    # labels exactly as the official scorer (tools/fin/run_eval.py -> harness.Panel.labels): calendar grid, so a day
    # missing from Binance's archive (2026-10-28) leaves the 7-day reference of D = 10-29 .. 11-04 undefined (NaN).
    # (tools/forward/forward_eval.py uses a row-based rolling mean that silently skips the missing day instead.)
    LB = [P.labels(int(i)) for i in days]
    L = np.stack([x["logrv"] for x in LB])
    RF = np.stack([x["ref"] for x in LB])
    dd = pd.to_datetime(P.days[days])
    d = pd.concat([pd.read_parquet(os.path.join(ROOT, "data/fin/crypto_daily.parquet")),
                   pd.read_parquet(os.path.join(ROOT, "data/fin/crypto_daily_SEALED.parquet"))])
    d["day"] = pd.to_datetime(d.t).dt.tz_localize(None).dt.normalize()
    fsum = d.pivot_table(index="day", columns="symbol", values="funding_sum").reindex(index=dd, columns=P.symbols)
    preds, meta = {}, {}
    for mn in a.models.split(","):
        t0 = time.time()
        try:
            r = H.run(P, M.get(mn), days)
            preds[mn] = {"q": r["q"], "p_up": r["p_up"]}
            meta[mn] = {"seconds": round(time.time() - t0, 1), "updates": int(np.sum(r["updates"]))}
            assert meta[mn]["updates"] == 0, f"{mn} updated parameters during deployment"
        except Exception as e:  # noqa: BLE001
            print(f"{mn}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            meta[mn] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        print(f"{mn}: {meta[mn]['seconds']}s", flush=True)

    held = np.array([P.group[s] == "heldout" for s in P.symbols])
    res = {"project": "fin", "variant": a.variant, "target": "next-day log realized vol, 24 USDT-M perps",
           "data_cutoff": str(dd[-1].date()), "scored_utc": C.stamp(), "git": C.git_state(), "data_sha256": data_sha,
           "checkpoints": pins, "candidates": CANDS, "models": meta,
           "funding_nonzero_share_by_window": {}, "windows": {}}
    for wn, (wlo, whi) in C.WINDOWS.items():
        sel = (dd >= wlo) & (dd <= whi)
        res["funding_nonzero_share_by_window"][wn] = float((fsum[sel].to_numpy() != 0).mean()) if sel.any() else None
        if not sel.any():
            res["windows"][wn] = {"n_days": 0}
            continue
        per = {}
        for mn, pr in preds.items():
            q = pr["q"][sel]
            Ls, Rs = L[sel], RF[sel]
            okq = np.isfinite(Ls) & np.isfinite(q[..., 2:11]).all(-1)                 # P3: needs log RV only
            lab = np.isfinite(Ls) & np.isfinite(Rs) & (Ls != Rs)                       # P1: official label
            okh = lab & np.isfinite(q[..., 6])
            e = Ls[..., None] - q[..., 2:11]
            hit = np.where(okh, (q[..., 6] > Rs) == (Ls > Rs), np.nan)
            per[mn] = {"hit": np.nanmean(hit, 1),
                       "qs": np.nanmean(np.where(okq, np.maximum(C.LV9 * e, (C.LV9 - 1) * e).mean(-1), np.nan), 1),
                       "cov80": np.nanmean(np.where(okq, (Ls >= q[..., 2]) & (Ls <= q[..., 10]), np.nan), 1),
                       "p1_coverage": np.array([okh.sum() / max(1, lab.sum())]),
                       "p3_coverage": np.array([okq.sum() / max(1, np.isfinite(Ls).sum())])}
            if held is not None:
                per[mn]["hit_heldout"] = np.nanmean(hit[:, held], 1)
        res["windows"][wn] = {"n_days": int(sel.sum()), "first": str(dd[sel][0].date()), "last": str(dd[sel][-1].date()),
                              "n_coin_days_labelled_p1": int((np.isfinite(L[sel]) & np.isfinite(RF[sel])).sum()),
                              "n_coin_days_logrv_p3": int(np.isfinite(L[sel]).sum()),
                              "by_candidate": {c: {"qs": C.compare(per, c, "qs", block=7),
                                                   "hit": C.compare(per, c, "hit", block=7, higher_better=True)}
                                               for c in CANDS if c in per}}
    day_str = np.array(dd.strftime("%Y-%m-%d").tolist(), dtype="U10")
    path, sh = C.save_preds("fin", f"fin_preds_{a.variant}", days=day_str, symbols=np.array(P.symbols, dtype="U20"),
                            **{f"q_{k}": v["q"] for k, v in preds.items()})
    # realised log RV is derived market data (not redistributed): local, git-ignored file for the audit
    C.save_preds("fin", f"fin_LABELS_local_{a.variant}", days=day_str, log_rv=L, ref=RF)
    res["preds_file"], res["preds_sha256"] = path, sh
    res["param_watch"] = watch.check()
    assert all(r["unchanged"] for r in res["param_watch"]), "model weights changed during deployment"
    res["checkpoints_after"] = C.verify_checkpoints("fin")
    print("->", C.write_json("fin", f"results_{a.variant}.json", res))


if __name__ == "__main__":
    main()
