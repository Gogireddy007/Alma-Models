"""Forward test of the frozen models on data that did not exist when they were frozen (after 2026-11-01).

The TEST year (2025-11 .. 2026-10) has been scored more than once, so it is no longer a clean hold-out. Every model
listed in FREEZE was frozen on 2026-11-01; this script refuses to score if any frozen code or checkpoint file has
changed since (sha256 check against tools/forward/FREEZE.json), then scores only target periods strictly after
FORWARD_START. It never fits or tunes anything.

    .venv/bin/python tools/forward/forward_eval.py freeze              # once: write FREEZE.json (done 2026-11-01)
    .venv/bin/python tools/forward/forward_eval.py refresh [--project gas,power,fin]   # download new public data
    .venv/bin/python tools/forward/forward_eval.py score   [--project gas,power,fin]   # -> runs/forward/*.json + .md

Data availability: gas = weekly EIA report (Thursdays); power = Energy-Charts, daily; fin = Binance monthly archive
(published after month end, so fin forward days arrive a month at a time). Scores with fewer than ~20 weeks (gas) or
~60 days (power, fin) are too noisy to conclude anything; the scoreboard prints n and bootstrap intervals.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
OUT = os.path.join(ROOT, "runs", "forward")
FREEZE_PATH = os.path.join(HERE, "FREEZE.json")
FORWARD_START = pd.Timestamp("2026-11-02")      # first target day/release strictly after the freeze day
DRYRUN = os.environ.get("ALMA_FORWARD_DRYRUN_START")   # pipeline test only: pretend the window starts earlier
if DRYRUN:
    FORWARD_START = pd.Timestamp(DRYRUN)
    OUT = os.path.join(OUT, "DRYRUN_NOT_A_RESULT")

FREEZE_FILES = {
    "gas": ["tools/gas/wm_gas.py", "tools/gas/wm_gas_ens.py", "tools/gas/search/feats.py", "tools/gas/search/nets.py",
            "tools/gas/storage.py", "tools/gas/models.py",
            "winning_models/gas_wm_gas/checkpoints/wm_gas.pt", "winning_models/gas_wm_gas/checkpoints/wm_gas_direct.pt",
            "winning_models/gas_wm_gas_ens/checkpoints/x_wm_s3456.pt",
            "winning_models/gas_wm_gas_ens/checkpoints/x_dir_s3456.pt"],
    "power": ["tools/power/wm_power.py", "tools/power/wm_power_blend.py", "tools/power/wm_data.py",
              "tools/power/wm_models.py", "tools/power/models.py", "tools/power/data.py"]
             + [f"winning_models/power_wm_power/checkpoints/frz_phys_{a}_s{s}.pt" for a in ("none", "direct") for s in range(3)],
    "fin": ["tools/fin/wm_fin.py", "tools/fin/wm_core.py", "tools/fin/wm_fin_blend.py", "tools/fin/models.py",
            "tools/fin/harness.py", "winning_models/fin_wm_fin/checkpoints/frozen_wm_fin.pt",
            "winning_models/fin_wm_fin/checkpoints/frozen_wm_direct.pt"],
}
MODELS = {"gas": ["wm_gas_ens", "wm_gas", "wm_gas_direct", "st_ols_wx"],
          "power": ["wm_power_blend", "wm_power", "lear"],
          "fin": ["wm_fin_blend", "wm_fin", "har_intra_dow"]}


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def cmd_freeze(_):
    if os.path.exists(FREEZE_PATH):
        raise SystemExit("FREEZE.json exists; a forward freeze is written once")
    rec = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "forward_start": str(FORWARD_START.date()),
           "models": MODELS, "sha256": {p: sha(p) for fs in FREEZE_FILES.values() for p in fs}}
    json.dump(rec, open(FREEZE_PATH, "w"), indent=1)
    print(f"wrote {FREEZE_PATH}: {len(rec['sha256'])} files")


def check_freeze(project):
    if DRYRUN and not os.path.exists(FREEZE_PATH):
        return {"frozen_utc": "DRYRUN (no freeze yet)"}
    rec = json.load(open(FREEZE_PATH))
    bad = [p for p in FREEZE_FILES[project] if sha(p) != rec["sha256"].get(p)]
    if bad:
        raise SystemExit(f"refused: frozen files changed since {rec['frozen_utc']}: {bad}")
    return rec


def sh(*args, env=None):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True, env=dict(os.environ, **(env or {})))


def ensure_ckpts():
    """The original wrappers load runs/<project>/search/ckpt/; a fresh clone keeps them in winning_models/."""
    import shutil
    for proj, src in (("gas", "gas_wm_gas"), ("power", "power_wm_power"), ("commodity", "oil_wm_sv")):
        d = os.path.join(ROOT, "runs", proj, "search", "ckpt")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(os.path.join(ROOT, "winning_models", src, "checkpoints")):
            if f.endswith(".pt") and not os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(ROOT, "winning_models", src, "checkpoints", f), d)


# =========================================================================== refresh
def cmd_refresh(a):
    today = time.strftime("%Y-%m-%d")
    for p in a.project.split(","):
        if p == "gas":
            sh("tools/gas/fetch.py", "--only", "storage,hh_fred,hh_eia,ng_c1,ng_c2,weather", "--until", today)
            sh("tools/gas/data.py", "build")
            sh("tools/gas/fetch.py", "--only", "cpc_obs,cpc_fcst", "--until", today)
            sh("tools/gas/storage.py", "build")
        elif p == "power":
            sh("tools/power/fetch.py", "--only", "price_de_lu,public_power", "--until", today)
            sh("tools/power/data.py", "build")
        elif p == "fin":
            uni = json.load(open(os.path.join(ROOT, "data", "fin", "universe.json")))
            syms = ",".join(uni["train"] + uni["heldout"])
            last = (pd.Timestamp(today) - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)).strftime("%Y-%m")
            for kind, extra in (("klines", ["--interval", "5m"]), ("fundingRate", [])):
                sh("tools/fin/fetch.py", "get", "--market", "um", "--kind", kind, *extra, "--symbols", syms,
                   "--start", "2026-11", "--end", last)
            sh("tools/fin/data.py", "build")


# =========================================================================== scoring helpers
LV9 = np.round(np.arange(0.1, 0.91, 0.1), 2)


def boot_ci(d, B=2000, block=1, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        bs.append(d[idx].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def score_gas():
    sys.path.insert(0, os.path.join(ROOT, "tools", "gas"))
    import storage as S
    import models as M
    w, o, f = S.load()
    ks = np.where(w.avail >= FORWARD_START)[0]
    ks = ks[w.avg5_l48.notna().to_numpy()[ks]]
    res = {"n_weeks": int(len(ks)), "weeks": [str(w.week.iloc[k].date()) for k in ks], "kinds": {}}
    if not len(ks):
        return res
    ds, a5 = w.ds_l48.to_numpy(float)[ks], w.avg5_l48.to_numpy(float)[ks]
    QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
    for kind in S.ORIGINS:
        ctxs = [S.build_context(int(k), kind, w, o, f) for k in ks]
        per = {}
        for mn in MODELS["gas"]:
            q = np.asarray(M.get(mn).storage(ctxs, QS, "l48"), float)
            d = ds[:, None] - q[:, 2:11]
            per[mn] = {"abs_err": np.abs(q[:, 6] - ds), "qs": np.maximum(LV9 * d, (LV9 - 1) * d).mean(1),
                       "hit": ((q[:, 6] > a5) == (ds > a5)).astype(float), "cov80": (ds >= q[:, 2]) & (ds <= q[:, 10])}
        res["kinds"][kind] = summarize(per, MODELS["gas"][0], "abs_err", block=1)
    return res


def score_power():
    sys.path.insert(0, os.path.join(ROOT, "tools", "power"))
    import data as D
    import models as M
    dates, P, C, flags = D.panel()
    idx = np.array([i for i, d in enumerate(dates) if d >= np.datetime64(FORWARD_START.date()) and np.isfinite(P[i]).sum() >= 23])
    res = {"n_days": int(len(idx)), "first": str(dates[idx[0]]) if len(idx) else None,
           "last": str(dates[idx[-1]]) if len(idx) else None}
    if not len(idx):
        return res
    QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ctxs = [D.build_context(int(i), 736) for i in idx]
    cur, prev = P[idx], P[idx - 1]
    per = {}
    for mn in MODELS["power"]:
        p = M.get(mn)
        q = np.asarray(p.batch([c["prices"] for c in ctxs], [c["covs"] for c in ctxs], 24, QS,
                               dates=[{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in ctxs]), float)
        med = q[:, 4]
        d = cur[:, None] - q
        m2 = np.isfinite(cur) & np.isfinite(prev) & (cur != prev)
        per[mn] = {"qs": np.nanmean(np.maximum(LV9[None, :, None] * d, (LV9[None, :, None] - 1) * d), (1, 2)),
                   "abs_err": np.nanmean(np.abs(med - cur), 1),
                   "hit": np.nanmean(np.where(m2, (med > prev) == (cur > prev), np.nan), 1),
                   "cov80": np.nanmean((cur >= q[:, 0]) & (cur <= q[:, -1]), 1)}
    res["summary"] = summarize(per, MODELS["power"][0], "qs", block=7)
    return res


def score_fin():
    os.environ["ALMA_FIN_TEST_GO"] = "1"        # forward rows live in the *_SEALED parquet (dated >= 2025-11-01)
    sys.path.insert(0, os.path.join(ROOT, "tools", "fin"))
    import harness as H
    import models as M
    P = H.Panel("crypto", allow_test=True)
    days = np.nonzero(P.days >= np.datetime64(FORWARD_START.date(), "ns"))[0]
    res = {"n_days": int(len(days))}
    if not len(days):
        return res
    d = pd.concat([pd.read_parquet(os.path.join(ROOT, "data/fin/crypto_daily.parquet")),
                   pd.read_parquet(os.path.join(ROOT, "data/fin/crypto_daily_SEALED.parquet"))])
    d["day"] = pd.to_datetime(d.t).dt.tz_localize(None).dt.normalize()
    lrv = np.log(d.pivot_table(index="day", columns="symbol", values="rv")
                 .where(d.pivot_table(index="day", columns="symbol", values="n_lab") >= 274))
    ref = lrv.shift(1).rolling(7).mean().where(lrv.shift(1).rolling(7).count() == 7)
    dd = pd.to_datetime(P.days[days])
    L, RF = lrv.reindex(index=dd, columns=P.symbols).to_numpy(), ref.reindex(index=dd, columns=P.symbols).to_numpy()
    per = {}
    for mn in MODELS["fin"]:
        r = H.run(P, M.get(mn), days)
        q = r["q"]
        ok = np.isfinite(L) & np.isfinite(RF) & np.isfinite(q[..., 2:11]).all(-1)
        e = L[..., None] - q[..., 2:11]
        per[mn] = {"hit": np.nanmean(np.where(ok, (q[..., 6] > RF) == (L > RF), np.nan), 1),
                   "qs": np.nanmean(np.where(ok, np.maximum(LV9 * e, (LV9 - 1) * e).mean(-1), np.nan), 1)}
    res["summary"] = summarize(per, MODELS["fin"][0], "qs", block=7)
    return res


def summarize(per, cand, key, block):
    out = {}
    for mn, v in per.items():
        out[mn] = {k: float(np.nanmean(x)) for k, x in v.items()}
        if mn != cand:
            out[mn][f"{cand}_minus_this_{key}"] = float(np.nanmean(per[cand][key] - v[key]))
            out[mn][f"ci95_{key}"] = boot_ci(per[cand][key] - v[key], block=block)
            if "hit" in v:
                out[mn]["diff_hit"] = float(np.nanmean(per[cand]["hit"] - v["hit"]))
                out[mn]["ci95_hit"] = boot_ci(per[cand]["hit"] - v["hit"], block=block)
    return out


def cmd_score(a):
    projects = a.project.split(",")
    if len(projects) > 1:            # each project imports its own `data` module: one process per project
        for p in projects:
            subprocess.run([PY, os.path.abspath(__file__), "score", "--project", p], cwd=ROOT, check=True)
        return
    p = projects[0]
    os.makedirs(OUT, exist_ok=True)
    ensure_ckpts()
    res = {"DRYRUN": bool(DRYRUN), "project": p, "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "forward_start": str(FORWARD_START.date()), "freeze": check_freeze(p)["frozen_utc"]}
    res["result"] = {"gas": score_gas, "power": score_power, "fin": score_fin}[p]()
    path = os.path.join(OUT, f"forward_{p}_{time.strftime('%Y%m%d')}.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:4000])
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "refresh", "score"])
    ap.add_argument("--project", default="gas,power,fin")
    a = ap.parse_args()
    {"freeze": cmd_freeze, "refresh": cmd_refresh, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
