"""
Alma-Gas walk-forward evaluation (docs/gas_prereg.md is binding).

    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task storage --split val [--models a,b]
    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task price   --split val [--models a,b]
    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task all --split val --metrics-only
    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task all --split val --rehearse        # TEST procedure on VAL
    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task all --split test --i-have-auditor-go   # ONCE

Plug-in: any name in models.REGISTRY (architect modules tools/gas/wm_*.py add theirs via register()) can be passed
with --models; an object with .storage is run on the storage task, one with .batch on the price task.
Forecasts: runs/gas/<split>/storage/<kind>/<model>.npz (q [n_weeks, n_series, 13], weeks = target week indices)
and runs/gas/<split>/price/<model>.npz (q [n_origins, 13, 20], origins = panel indices). Cached files are reused
only if their index list matches (never in strict mode). Metrics: runs/gas/<split>/metrics.json + report.md.

TEST is sealed: --split test refuses to run without a committed runs/gas/FROZEN.json, the explicit flag
--i-have-auditor-go, a clean tools/gas + docs/gas_prereg.md, pinned data hashes, no existing RUN.json / forecasts.
"""
import os
import gc
import sys
import json
import time
import hashlib
import argparse
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data as D          # noqa: E402
import storage as S       # noqa: E402

_cm = os.path.join(ROOT, "tools", "commodity", "metrics.py")
import importlib.util     # noqa: E402
_spec = importlib.util.spec_from_file_location("commodity_metrics", _cm)
MT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MT)

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
IMED, I10, I90 = QS.index(0.5), QS.index(0.1), QS.index(0.9)
CTX = 1024
H_PRICE = (1, 5, 20)
VOL_W = 5                   # vol direction: 5-day realised vol vs the previous 5 days; non-overlap step 10
OUT = os.path.join(ROOT, "runs", "gas")
FROZEN = os.path.join(OUT, "FROZEN.json")
CHUNK = 64

STORAGE_MODELS = ["st_climatology", "st_persistence", "st_seasonal_naive", "st_ols_wx", "st_ols_wx_nolag",
                  "st_arima_dev", "chronos2", "chronos2_cov", "chronos_bolt_base", "tirex", "timestral_5m",
                  "alma_s0", "alma_direct_s0"]
PRICE_MODELS = ["rw_gauss", "auto_ets", "auto_arima", "px_fund_ols", "chronos2", "chronos2_cov", "chronos_bolt_base",
                "tirex", "timestral_5m", "kronos_small", "alma_s0", "alma_direct_s0"]
STORAGE_BASELINES = {"majority", "st_climatology", "st_persistence", "st_seasonal_naive", "st_ols_wx",
                     "st_ols_wx_nolag", "st_arima_dev"}
PRICE_BASELINES = {"majority", "momentum", "seasonal", "rw_gauss", "auto_ets", "auto_arima", "px_fund_ols",
                   "vol_meanrev", "vol_persist", "garch"}
FM = {"chronos2", "chronos2_cov", "chronos_bolt_base", "tirex", "timestral_5m", "kronos_small"}
ALMA = {"alma_s0", "alma_direct_s0"}
CONTAMINATED = {"timestral_5m": "released 2026-10-15; training data may include the TEST period"}
ALMA_PINS = {
    "data/ckpt/wm_ssm_none_s0.pt": "3c959a64099550fa729e19ae2645dff3e8eff695d6ca999e9eeaabf4217ec422",
    "data/ckpt/wm_ssm_direct_s0.pt": "be2faf59694f334db56614ee767b9fbebdb26dde94800d0d7764a8d1316fc9be",
}
DATA_PINS = {                                   # amendment A1 (data snapshot 2026-10-31)
    "data/gas/daily.csv": "569dd1e10464f3487816650f9c1197d43bd06692319ec618aeab769396d5265c",
    "data/gas/weekly.csv": "929d210b01d15bb034e7b0fdcb06a7319ed755484d15391f97ad158176892309",
    "data/gas/cpc_obs.csv": "083991d243534519dbdecc70ec3261e81abfab7003f831a0e191869b46c52fc9",
    "data/gas/cpc_fcst.csv": "fd7352d1b5a833b302d4d8644a7e4a4f396e46ea7c4f2e45b2fb8e0ff3265a3e",
}
RUN_DIR = {}


def split_dir(split):
    return RUN_DIR.get(split, os.path.join(OUT, split))


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def frozen():
    return json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}


def kind_of(mn, task):
    if mn in (STORAGE_BASELINES if task == "storage" else PRICE_BASELINES):
        return "baseline"
    if mn in FM:
        return "fm"
    if mn in ALMA:
        return "alma"
    fz = frozen()
    return "candidate" if mn in fz.get("models", []) else "model"


# =========================================================================== forecasting
def _save(dst, **kw):
    np.savez(dst + ".tmp.npz", **kw)
    os.replace(dst + ".tmp.npz", dst)


def forecast_storage(split, names, force=False, strict=False):
    import models as M
    w, o, f = S.load()
    ks = S.target_weeks(split, w)
    ctx_cache = {}
    for mn in names:
        try:
            p = M.get(mn)
        except (KeyError, RuntimeError) as e:
            if strict:
                raise
            print(f"[{split}/storage] {mn}: SKIP ({e})", flush=True)
            continue
        if not hasattr(p, "storage"):
            continue
        for kind in S.ORIGINS:
            sdir = os.path.join(split_dir(split), "storage", kind)
            os.makedirs(sdir, exist_ok=True)
            dst = os.path.join(sdir, f"{mn}.npz")
            if strict and os.path.exists(dst):
                raise SystemExit(f"strict mode: {dst} exists")
            if os.path.exists(dst) and not force and np.array_equal(np.load(dst)["weeks"], ks):
                print(f"[{split}/storage/{kind}] {mn}: cached", flush=True)
                continue
            if kind not in ctx_cache:
                ctx_cache[kind] = [S.build_context(int(k), kind, w, o, f) for k in ks]
            ctxs = ctx_cache[kind]
            t0 = time.time()
            q = np.stack([np.asarray(p.storage(ctxs, QS, s), float) for s in S.SERIES], 1)
            if q.shape != (len(ks), len(S.SERIES), len(QS)):
                raise SystemExit(f"{mn}: storage shape {q.shape}")
            meta = dict(getattr(p, "meta", {}), model=mn, task="storage", kind=kind, split=split, series=S.SERIES,
                        quantiles=QS, seconds=round(time.time() - t0, 1), n=len(ks),
                        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            _save(dst, q=q, weeks=ks, meta=np.array(json.dumps(meta, default=str)))
            print(f"[{split}/storage/{kind}] {mn}: n={len(ks)} {meta['seconds']}s "
                  f"nonfinite={int((~np.isfinite(q[:, :, I9])).sum())}", flush=True)
        del p
        gc.collect()


def forecast_price(split, names, force=False, strict=False):
    import models as M
    dates, y, C, extra = D.panel()
    idx = D.split_origins(split, dates, CTX)
    sdir = os.path.join(split_dir(split), "price")
    os.makedirs(sdir, exist_ok=True)
    H = max(H_PRICE)
    for mn in names:
        try:
            p = M.get(mn)
        except (KeyError, RuntimeError) as e:
            if strict:
                raise
            print(f"[{split}/price] {mn}: SKIP ({e})", flush=True)
            continue
        if not hasattr(p, "batch"):
            continue
        stride = getattr(p, "stride", 1) if split != "dev" else 1
        oi = idx[::stride]
        dst = os.path.join(sdir, f"{mn}.npz")
        if strict and os.path.exists(dst):
            raise SystemExit(f"strict mode: {dst} exists")
        if os.path.exists(dst) and not force and np.array_equal(np.load(dst)["origins"], oi):
            print(f"[{split}/price] {mn}: cached", flush=True)
            continue
        t0 = time.time()
        qs = []
        for s in range(0, len(oi), CHUNK):
            ctxs = [D.build_context(int(i), CTX, dates, y, C, extra) for i in oi[s:s + CHUNK]]
            q = np.asarray(p.batch(ctxs, H, QS), float)
            if q.shape != (len(ctxs), len(QS), H):
                raise SystemExit(f"{mn}: price shape {q.shape}")
            qs.append(q)
            print(f"[{split}/price] {mn}: {min(s + CHUNK, len(oi))}/{len(oi)} {time.time() - t0:.0f}s", flush=True)
        q = np.concatenate(qs)
        meta = dict(getattr(p, "meta", {}), model=mn, task="price", split=split, H=H, quantiles=QS, ctx=CTX,
                    stride=stride, seconds=round(time.time() - t0, 1), n=len(oi),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        _save(dst, q=q, origins=oi, meta=np.array(json.dumps(meta, default=str)))
        print(f"[{split}/price] {mn}: n={len(oi)} {meta['seconds']}s nonfinite={int((~np.isfinite(q[:, I9])).sum())}",
              flush=True)
        del p, q, qs
        gc.collect()


def load_npz(d, key):
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz") and not fn.endswith(".tmp.npz"):
            z = np.load(os.path.join(d, fn))
            out[fn[:-4]] = (z["q"], z[key], json.loads(str(z["meta"])))
    return out


# =========================================================================== statistics helpers
def dir_stats(pred, act, sub=None, block=1):
    n = len(act)
    hit = (pred == act).astype(float)
    k = int(hit.sum())
    sub = np.ones(n, bool) if sub is None else sub
    ns, ks = int(sub.sum()), int(hit[sub].sum())
    lo, hi = MT.wilson(ks, ns)
    _, ppt = MT.pesaran_timmermann(pred[sub], act[sub])
    blo, bhi = MT.block_boot_ci(hit, block) if n > 1 else (np.nan, np.nan)
    return {"n": n, "acc": k / n if n else np.nan, "boot_lo": blo, "boot_hi": bhi, "block": block,
            "n_nonoverlap": ns, "acc_nonoverlap": ks / ns if ns else np.nan, "ci_lo": lo, "ci_hi": hi,
            "p_binom": MT.binom_p(ks, ns), "p_pt": ppt, "base_rate_up": float(act.mean()) if n else np.nan,
            "frac_up_pred": float(pred.mean()) if n else np.nan}


def pair_dir(pa, pb, act, sub, h):
    """Direction, model a vs b (0/1 losses). Primary p: h == 1 (and weekly storage): exact McNemar on all labels
    (storage) / DM-HLN (daily h = 1); h > 1: exact McNemar on the non-overlapping subsample. DM-HLN lag-2h kept as
    supplementary."""
    la, lb = (pa != act).astype(float), (pb != act).astype(float)
    d = la - lb
    n01, n10, pm = MT.mcnemar_exact(la[sub] > 0, lb[sub] > 0)
    _, st, pdm = MT.dm_hln(d, 1)
    _, _, pw = MT.dm_hln(d, h, lag=2 * h) if h > 1 else (None, None, pdm)
    return {"n": int(len(d)), "n_sub": int(sub.sum()), "acc_diff": float(-d.mean()) if len(d) else np.nan,
            "p_mcnemar": pm, "n_a_wrong_b_right": n01, "n_a_right_b_wrong": n10, "p_dm": pdm, "p_dm_wide": pw}


def pair_loss(la, lb, sub, h, block=4):
    d = la - lb
    _, st, p = MT.dm_hln(d[sub], 1)
    lo, hi = MT.block_boot_ci(d, block) if len(d) > 1 else (np.nan, np.nan)
    _, _, pw = MT.dm_hln(d, h, lag=2 * h) if h > 1 else (None, None, p)
    return {"n": int(len(d)), "n_sub": int(sub.sum()), "mean_diff": float(d.mean()) if len(d) else np.nan, "p": p,
            "stat": st, "boot_lo": lo, "boot_hi": hi, "p_dm_wide": pw}


def boot_weeks_ratio(num, den, B=2000, seed=0):
    """Week-clustered bootstrap CI of sum(num)/sum(den) (resample whole weeks)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n, size=(B, n))
    r = num[ix].sum(1) / np.maximum(den[ix].sum(1), 1e-12)
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


# =========================================================================== storage scoring
def storage_majority(w):
    """DEV up-fraction of y_W (weeks released <= 2025-06-30 with a defined label), per series."""
    m = (w.avail <= D.TRAIN_END)
    out = {}
    for s in S.SERIES:
        dv = w.loc[m, f"dev_{s}"].dropna()
        dv = dv[dv != 0]
        out[s] = float((dv > 0).mean())
    return out


def score_storage(split):
    w, _, _ = S.load()
    ks = S.target_weeks(split, w)
    maj = storage_majority(w)
    res = {"n_weeks": int(len(ks)), "first_week": str(w.week.iloc[ks[0]].date()) if len(ks) else None,
           "last_week": str(w.week.iloc[ks[-1]].date()) if len(ks) else None, "dev_majority_upfrac": maj,
           "kinds": {}}
    for kind in S.ORIGINS:
        preds = load_npz(os.path.join(split_dir(split), "storage", kind), "weeks")
        out = {"series": {}, "regions_pooled": {}}
        for si, s in enumerate(S.SERIES):
            dv = w[f"dev_{s}"].to_numpy(float)[ks]
            ds = w[f"ds_{s}"].to_numpy(float)[ks]
            a5 = w[f"avg5_{s}"].to_numpy(float)[ks]
            ok = np.isfinite(dv) & (dv != 0)
            act = (dv > 0).astype(int)
            majc = int(maj[s] > 0.5)
            cand = {"majority": np.where(ok, majc, -1)}
            med, qsl, ael = {}, {}, {}
            for mn, (q, wk, meta) in preds.items():
                pos = {int(k): j for j, k in enumerate(wk)}
                m = np.full(len(ks), np.nan)
                Q = np.full((len(ks), len(QS)), np.nan)
                for t, k in enumerate(ks):
                    if int(k) in pos:
                        Q[t] = q[pos[int(k)], si]
                m = Q[:, IMED]
                p = np.where(m > a5, 1, np.where(m < a5, 0, majc))
                cand[mn] = np.where(np.isfinite(m) & ok, p, -1)
                fin = np.isfinite(Q[:, I9]).all(1) & np.isfinite(ds)
                qsl[mn] = np.where(fin, MT.pinball(np.nan_to_num(ds), np.nan_to_num(Q[:, I9]), [QS[i] for i in I9]), np.nan)
                ael[mn] = np.where(np.isfinite(m) & np.isfinite(ds), np.abs(m - ds), np.nan)
                med[mn] = (Q, fin)
            blk = {"base_rate_up": float(act[ok].mean()) if ok.any() else np.nan, "n_ties_or_missing": int((~ok).sum()),
                   "models": {}, "pairwise": {}, "pairwise_mae": {}, "pairwise_qs": {}}
            for mn, p in cand.items():
                sel = p >= 0
                if not sel.any():
                    continue
                r = dir_stats(p[sel], act[sel], block=1)
                r["kind"] = kind_of(mn, "storage")
                r["contaminated"] = CONTAMINATED.get(mn)
                if mn in qsl:
                    f = np.isfinite(qsl[mn])
                    r["qs"] = float(np.nanmean(qsl[mn])) if f.any() else None
                    r["mae_bcf"] = float(np.nanmean(ael[mn])) if np.isfinite(ael[mn]).any() else None
                    Q, fin = med[mn]
                    if fin.any():
                        r["cov80"] = float(np.mean((ds[fin] >= Q[fin, I10]) & (ds[fin] <= Q[fin, I90])))
                blk["models"][mn] = r
            names = list(blk["models"])
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    sel = (cand[a] >= 0) & (cand[b] >= 0)
                    if sel.sum() >= 5:
                        blk["pairwise"][f"{a}|{b}"] = pair_dir(cand[a][sel], cand[b][sel], act[sel],
                                                               np.ones(sel.sum(), bool), 1)
                    if a in qsl and b in qsl:
                        for key, L in (("pairwise_mae", ael), ("pairwise_qs", qsl)):
                            f = np.isfinite(L[a]) & np.isfinite(L[b])
                            if f.sum() >= 5:
                                blk[key][f"{a}|{b}"] = pair_loss(L[a][f], L[b][f], np.ones(f.sum(), bool), 1, block=1)
            out["series"][s] = blk
        # regions pooled (secondary): 5 regions x weeks, week-clustered bootstrap
        regs = [s for s in S.SERIES if s != "l48"]
        for mn in set.intersection(*[set(out["series"][r]["models"]) for r in regs]) if regs else []:
            num, den = np.zeros(len(ks)), np.zeros(len(ks))
            for r in regs:
                dv = w[f"dev_{r}"].to_numpy(float)[ks]
                okr = np.isfinite(dv) & (dv != 0)
                if mn == "majority":
                    pr = np.where(okr, int(maj[r] > 0.5), -1)
                else:
                    q = preds[mn][0]
                    pos = {int(k): j for j, k in enumerate(preds[mn][1])}
                    a5 = w[f"avg5_{r}"].to_numpy(float)[ks]
                    mm = np.array([q[pos[int(k)], S.SERIES.index(r), IMED] if int(k) in pos else np.nan for k in ks])
                    pr = np.where(np.isfinite(mm) & okr, np.where(mm > a5, 1, np.where(mm < a5, 0, int(maj[r] > 0.5))), -1)
                sel = pr >= 0
                num += np.where(sel, pr == (dv > 0), 0)
                den += sel
            if den.sum():
                lo, hi = boot_weeks_ratio(num, den)
                out["regions_pooled"][mn] = {"acc": float(num.sum() / den.sum()), "n_labels": int(den.sum()),
                                             "ci_lo": lo, "ci_hi": hi, "kind": kind_of(mn, "storage")}
        res["kinds"][kind] = out
    add_holm_storage(res)
    return res


def add_holm_storage(res):
    """Holm per (origin kind, endpoint) family over claim models (candidates + alma) x every non-contaminated
    competitor, on the primary p (McNemar for direction; DM-HLN for MAE/QS); plus a descriptive FM-vs-baseline
    family. 'beats' = better AND Holm p < 0.05 (AND bootstrap CI excluding 0 for losses)."""
    for kind, blk in res["kinds"].items():
        b = blk["series"].get("l48")
        if not b:
            continue
        names = list(b["models"])
        claim = [m for m in names if b["models"][m]["kind"] in ("candidate", "alma")]
        for key, pk in (("pairwise", "p_mcnemar"), ("pairwise_mae", "p"), ("pairwise_qs", "p")):
            pw = b[key]
            for a in claim:
                ks = [f"{a}|{c}" for c in names if c != a and not CONTAMINATED.get(c) and f"{a}|{c}" in pw]
                for k, pa in zip(ks, MT.holm([pw[k][pk] for k in ks])):
                    pw[k]["p_holm_own"] = float(pa)
                    better = pw[k]["acc_diff"] > 0 if key == "pairwise" else pw[k]["mean_diff"] < 0
                    ci_ok = True if key == "pairwise" else (pw[k]["boot_hi"] < 0)
                    pw[k]["beats"] = bool(better and pa < 0.05 and ci_ok)
            ks = [f"{a}|{c}" for a in names if b["models"][a]["kind"] == "fm"
                  for c in names if b["models"][c]["kind"] == "baseline" and f"{a}|{c}" in pw]
            for k, pa in zip(ks, MT.holm([pw[k][pk] for k in ks])):
                pw[k]["p_holm_descriptive"] = float(pa)


# =========================================================================== price scoring
def logret(v):
    lv = np.log(np.maximum(v, 0.01))
    return np.diff(lv)


def price_majority(d, v):
    tr = d <= np.datetime64(D.TRAIN_END.date(), "D")
    vt = v[tr]
    out = {}
    for h in H_PRICE:
        ch = vt[h:] - vt[:-h]
        ch = ch[ch != 0]
        out[f"T1_{h}"] = float((ch > 0).mean())
    r = logret(vt)
    ys = []
    for t in range(VOL_W, len(r) - VOL_W):
        a, b = r[t - VOL_W:t], r[t:t + VOL_W]
        ys.append(np.std(b, ddof=1) > np.std(a, ddof=1))
    out["T2"] = float(np.mean(ys))
    return out


_GARCH = {}


def garch_series(d, v):
    if "g" not in _GARCH:
        r = logret(v)
        rd = d[1:]
        m = (rd >= np.datetime64("2000-01-01")) & (rd <= np.datetime64(D.TRAIN_END.date(), "D")) & np.isfinite(r)
        th = MT.garch_fit(r[m])
        s2 = MT.garch_filter(np.where(np.isfinite(r), r, 0.0), th, float(np.var(r[m])))
        _GARCH["g"] = (th, s2)
    return _GARCH["g"]


def score_price(split):
    d, v, _, _ = D.panel()
    idx = D.split_origins(split, d, CTX)
    end = D.split_end(split, d)
    preds = load_npz(os.path.join(split_dir(split), "price"), "origins")
    maj = price_majority(d, v)
    res = {"split": split, "n_origins": int(len(idx)), "first_origin_target": str(d[idx[0]]) if len(idx) else None,
           "train_majority_upfrac": maj, "T1": {}, "T2": {}, "T4": {}}
    r = logret(v)
    for h in H_PRICE:
        ok = np.array([(i - 1 + h < len(v)) and d[min(i - 1 + h, len(v) - 1)] <= end for i in idx])
        oi, posn = idx[ok], np.arange(len(idx))[ok]
        last, fut = v[oi - 1], v[oi - 1 + h]
        nz = fut != last
        oi, posn, last, fut = oi[nz], posn[nz], last[nz], fut[nz]
        sub = posn % h == 0
        act = (fut > last).astype(int)
        majc = int(maj[f"T1_{h}"] > 0.5)
        cand = {"majority": np.full(len(oi), majc), "momentum": (v[oi - 1] > v[oi - 1 - h]).astype(int)}
        seas = []
        for i in oi:
            ch = []
            for k in range(1, 6):
                j = int(np.searchsorted(d, d[i - 1] - np.timedelta64(365 * k, "D"), side="right")) - 1
                ch.append(np.log(max(v[j + h], .01)) - np.log(max(v[j], .01)))
            seas.append(int(np.mean(ch) > 0))
        cand["seasonal"] = np.array(seas)
        for mn, (q, qo, meta) in preds.items():
            pos = {int(o): j for j, o in enumerate(qo)}
            med = np.array([q[pos[int(o)], IMED, h - 1] if int(o) in pos else np.nan for o in oi])
            p = np.where(med > last, 1, np.where(med < last, 0, majc))
            cand[mn] = np.where(np.isfinite(med), p, -1)
        out = {"n_ties_excluded": int((~nz).sum()), "base_rate_up": float(act.mean()) if len(act) else np.nan,
               "models": {}, "pairwise": {}}
        for mn, p in cand.items():
            s = p >= 0
            if s.sum() < 3:
                continue
            out["models"][mn] = dict(dir_stats(p[s], act[s], sub[s], 2 * h), kind=kind_of(mn, "price"),
                                     contaminated=CONTAMINATED.get(mn))
        for a in out["models"]:
            for b in out["models"]:
                if a != b:
                    s = (cand[a] >= 0) & (cand[b] >= 0)
                    if s.sum() >= 5:
                        pw = pair_dir(cand[a][s], cand[b][s], act[s], sub[s], h)
                        pw["p"] = pw["p_dm"] if h == 1 else pw["p_mcnemar"]
                        out["pairwise"][f"{a}|{b}"] = pw
        res["T1"][str(h)] = out
    # ---- T2: 5-day realised-vol direction
    ok = np.array([(i + VOL_W - 1 < len(v)) and d[min(i + VOL_W - 1, len(v) - 1)] <= end for i in idx])
    oi, posn = idx[ok], np.arange(len(idx))[ok]
    past = np.array([np.std(r[i - 1 - VOL_W:i - 1], ddof=1) for i in oi])       # returns ending at obs i-1
    futv = np.array([np.std(r[i - 1:i - 1 + VOL_W], ddof=1) for i in oi])      # returns over obs i .. i+4
    sub = posn % (2 * VOL_W) == 0
    act = (futv > past).astype(int)
    th, s2 = garch_series(d, v)
    cand = {"majority": np.full(len(oi), int(maj["T2"] > 0.5)),
            "vol_meanrev": np.array([int(past[k] < np.median([np.std(r[j - VOL_W:j], ddof=1)
                                                               for j in range(i - 1 - 251, i)])) for k, i in enumerate(oi)]),
            "vol_persist": np.array([int(past[k] > np.std(r[i - 1 - 2 * VOL_W:i - 1 - VOL_W], ddof=1))
                                     for k, i in enumerate(oi)]),
            "garch": np.array([int(MT.garch_mean_var(s2[i - 1], th, k=VOL_W) > past[k] ** 2) for k, i in enumerate(oi)])}
    for mn, (q, qo, meta) in preds.items():
        pos = {int(o): j for j, o in enumerate(qo)}
        p = np.full(len(oi), -1)
        for k, o in enumerate(oi):
            if int(o) in pos:
                qq = np.maximum(q[pos[int(o)]], 0.01)
                kk = np.arange(1, VOL_W + 1)
                sm = np.mean((np.log(qq[I90, :VOL_W]) - np.log(qq[I10, :VOL_W])) / (2 * 1.2816 * np.sqrt(kk)))
                p[k] = int(sm > past[k]) if np.isfinite(sm) else -1
        cand[mn] = p
    out = {"base_rate_up": float(act.mean()) if len(act) else np.nan, "models": {}, "pairwise": {}}
    for mn, p in cand.items():
        s = p >= 0
        if s.sum() >= 3:
            out["models"][mn] = dict(dir_stats(p[s], act[s], sub[s], 2 * VOL_W), kind=kind_of(mn, "price"),
                                     contaminated=CONTAMINATED.get(mn))
    for a in out["models"]:
        for b in out["models"]:
            if a != b:
                s = (cand[a] >= 0) & (cand[b] >= 0)
                if s.sum() >= 5:
                    pw = pair_dir(cand[a][s], cand[b][s], act[s], sub[s], 2 * VOL_W)
                    pw["p"] = pw["p_mcnemar"]
                    out["pairwise"][f"{a}|{b}"] = pw
    res["T2"][str(VOL_W)] = out
    # ---- T4: log-price quantile score vs rw_gauss
    for h in H_PRICE:
        ok = np.array([(i - 1 + h < len(v)) and d[min(i - 1 + h, len(v) - 1)] <= end for i in idx])
        oi, posn = idx[ok], np.arange(len(idx))[ok]
        y = np.log(np.maximum(v[oi - 1 + h], 0.01))
        sub = posn % h == 0
        L, cov = {}, {}
        for mn, (q, qo, meta) in preds.items():
            pos = {int(o): j for j, o in enumerate(qo)}
            sel = np.array([int(o) in pos for o in oi])
            l = np.full(len(oi), np.nan)
            c = np.full(len(oi), np.nan)
            if sel.any():
                Q = np.log(np.maximum(np.stack([q[pos[int(o)], :, h - 1] for o in oi[sel]]), 0.01))
                fin = np.isfinite(Q[:, I9]).all(1)
                ll = np.full(sel.sum(), np.nan)
                ll[fin] = MT.pinball(y[sel][fin], Q[fin][:, I9], [QS[i] for i in I9])
                l[sel] = ll
                c[sel] = np.where(fin, (y[sel] >= Q[:, I10]) & (y[sel] <= Q[:, I90]), np.nan)
            L[mn], cov[mn] = l, c
        out = {"models": {}, "pairwise": {}}
        rw = L.get("rw_gauss")
        for mn in L:
            s = np.isfinite(L[mn]) & (np.isfinite(rw) if rw is not None else True)
            if s.sum() < 3:
                continue
            row = {"n": int(s.sum()), "qs": float(L[mn][s].mean()), "cov80": float(np.nanmean(cov[mn][s])),
                   "kind": kind_of(mn, "price"), "contaminated": CONTAMINATED.get(mn)}
            if rw is not None:
                row["skill_vs_rw"] = 1 - row["qs"] / float(rw[s].mean())
                if mn != "rw_gauss":
                    pw = pair_loss(L[mn][s], rw[s], sub[s], h, block=2 * h)
                    row["p_vs_rw"] = pw["p"]
            out["models"][mn] = row
        for a in out["models"]:
            for b in out["models"]:
                if a != b:
                    s = np.isfinite(L[a]) & np.isfinite(L[b])
                    if s.sum() >= 5:
                        out["pairwise"][f"{a}|{b}"] = pair_loss(L[a][s], L[b][s], sub[s], h, block=2 * h)
        res["T4"][str(h)] = out
    # Holm: per target family over (non-baseline model x baseline x horizon), descriptive for FMs
    for tgt in ("T1", "T2"):
        keys, ps = [], []
        for h, blk in res[tgt].items():
            bases = [m for m, rr in blk["models"].items() if rr["kind"] == "baseline"]
            for m, rr in blk["models"].items():
                if rr["kind"] != "baseline":
                    for b in bases:
                        if f"{m}|{b}" in blk["pairwise"]:
                            keys.append((h, m, b))
                            ps.append(blk["pairwise"][f"{m}|{b}"]["p"])
        for (h, m, b), pa in zip(keys, MT.holm(ps)):
            res[tgt][h]["pairwise"][f"{m}|{b}"]["p_holm_family"] = float(pa)
    return res


# =========================================================================== report
def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(res, path):
    L = [f"# Alma-Gas {res['split']} report", "", f"commit {res.get('commit')}, scored {res.get('scored')}, "
         f"rehearsal={res.get('rehearsal')}", ""]
    st = res.get("storage")
    if st:
        L += [f"## Storage (PRIMARY): sign(dS_W - avg5_W), {st['n_weeks']} target weeks {st['first_week']} .. "
              f"{st['last_week']}", ""]
        for kind, blk in st["kinds"].items():
            b = blk["series"].get("l48", {})
            L += [f"### origin {kind}, Lower 48 (base rate up {fmt(b.get('base_rate_up'))}; DEV majority up-frac "
                  f"{fmt(st['dev_majority_upfrac']['l48'])})", "",
                  "| model | kind | n | acc | Wilson 95% | p binom | p PT | MAE Bcf | QS | cov80 | McNemar p vs st_persistence | vs st_ols_wx |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for mn, r in sorted(b.get("models", {}).items(), key=lambda kv: -kv[1]["acc"]):
                p1 = b["pairwise"].get(f"{mn}|st_persistence", {}).get("p_mcnemar")
                p2 = b["pairwise"].get(f"{mn}|st_ols_wx", {}).get("p_mcnemar")
                L.append(f"| {mn}{' (CONTAMINATED)' if r.get('contaminated') else ''} | {r['kind']} | {r['n']} | "
                         f"{fmt(r['acc'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {fmt(r['p_binom'])} | "
                         f"{fmt(r['p_pt'])} | {fmt(r.get('mae_bcf'), 1)} | {fmt(r.get('qs'), 2)} | {fmt(r.get('cov80'), 2)} | "
                         f"{fmt(p1)} | {fmt(p2)} |")
            pool = blk.get("regions_pooled", {})
            if pool:
                L += ["", "Regions pooled (secondary; 5 regions x weeks, week-clustered bootstrap):", "",
                      "| model | n labels | acc | 95% CI |", "|---|---|---|---|"]
                for mn, r in sorted(pool.items(), key=lambda kv: -kv[1]["acc"]):
                    L.append(f"| {mn} | {r['n_labels']} | {fmt(r['acc'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] |")
            L.append("")
    pr = res.get("price")
    if pr:
        L += ["## Henry Hub spot (secondary / negative control)", ""]
        for h, blk in pr["T1"].items():
            L += [f"### T1 direction h={h} (base rate up {fmt(blk['base_rate_up'])})", "",
                  "| model | kind | n | acc | block-boot 95% | n non-overlap | acc non-ovl | Wilson (non-ovl) | p binom |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for mn, r in sorted(blk["models"].items(), key=lambda kv: -kv[1]["acc"]):
                L.append(f"| {mn}{' (CONTAMINATED)' if r.get('contaminated') else ''} | {r['kind']} | {r['n']} | "
                         f"{fmt(r['acc'])} | [{fmt(r['boot_lo'])}, {fmt(r['boot_hi'])}] | {r['n_nonoverlap']} | "
                         f"{fmt(r['acc_nonoverlap'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {fmt(r['p_binom'])} |")
            L.append("")
        for h, blk in pr["T2"].items():
            L += [f"### T2 {h}-day realised-vol direction (base rate up {fmt(blk['base_rate_up'])})", "",
                  "| model | kind | n | acc | n non-ovl | acc non-ovl | Wilson (non-ovl) |", "|---|---|---|---|---|---|---|"]
            for mn, r in sorted(blk["models"].items(), key=lambda kv: -kv[1]["acc"]):
                L.append(f"| {mn} | {r['kind']} | {r['n']} | {fmt(r['acc'])} | {r['n_nonoverlap']} | "
                         f"{fmt(r['acc_nonoverlap'])} | [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] |")
            L.append("")
        for h, blk in pr["T4"].items():
            L += [f"### T4 log-price QS h={h}", "", "| model | n | QS | skill vs rw | cov80 | p vs rw |", "|---|---|---|---|---|---|"]
            for mn, r in sorted(blk["models"].items(), key=lambda kv: kv[1]["qs"]):
                L.append(f"| {mn} | {r['n']} | {fmt(r['qs'], 4)} | {fmt(r.get('skill_vs_rw'))} | {fmt(r['cov80'], 2)} | "
                         f"{fmt(r.get('p_vs_rw'))} |")
            L.append("")
    open(path, "w").write("\n".join(L) + "\n")


# =========================================================================== strict mode + main
def preflight(task_models):
    import models as M
    if M.LOAD_ERRORS:
        raise SystemExit(f"preflight: architect module import errors: {M.LOAD_ERRORS}")
    fz = frozen()
    metas = {}
    for mn in sorted(set(sum(task_models.values(), []))):
        p = M.get(mn)
        meta = dict(getattr(p, "meta", {}))
        ck = meta.get("ckpt")
        if mn.startswith("alma") and (ck not in ALMA_PINS or ALMA_PINS[ck] != meta.get("ckpt_sha256")):
            raise SystemExit(f"preflight: {mn} checkpoint {ck} not pinned / sha mismatch")
        for path, sh in (fz.get("sha256") or {}).items():
            if ck == path and meta.get("ckpt_sha256") != sh:
                raise SystemExit(f"preflight: {mn} checkpoint {ck} sha256 != FROZEN.json")
        metas[mn] = meta
        del p
    return metas


def check_data_pins():
    if not DATA_PINS:
        raise SystemExit("strict mode refused: run_eval.DATA_PINS is empty (data snapshot not pinned)")
    for rel, sh in DATA_PINS.items():
        if sha_file(os.path.join(ROOT, rel)) != sh:
            raise SystemExit(f"strict mode refused: {rel} sha256 != pin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["storage", "price", "all"])
    ap.add_argument("--split", required=True, choices=["dev", "val", "test"])
    ap.add_argument("--models", default="")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rehearse", action="store_true", help="exact TEST procedure on VAL into runs/gas/rehearsal_val/")
    ap.add_argument("--i-have-auditor-go", action="store_true", help="required for --split test")
    a = ap.parse_args()
    tasks = ["storage", "price"] if a.task == "all" else [a.task]
    strict = (a.split == "test" or a.rehearse) and not a.metrics_only
    if a.rehearse:
        if a.split != "val":
            raise SystemExit("--rehearse only with --split val")
        RUN_DIR["val"] = os.path.join(OUT, "rehearsal_val")
    if a.split == "test" and not a.i_have_auditor_go:
        raise SystemExit("TEST is sealed: pass --i-have-auditor-go only after the manager relays the auditor's GO")
    run_rec, rp = None, None
    if strict:
        rdir = split_dir(a.split)
        if a.split == "test":
            if not os.path.exists(FROZEN):
                raise SystemExit("strict mode refused: runs/gas/FROZEN.json missing")
            tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", "runs/gas/FROZEN.json"],
                                     capture_output=True, text=True).returncode == 0
            if not tracked:
                raise SystemExit("strict mode refused: runs/gas/FROZEN.json is not committed (git add -f + commit)")
        if a.task != "all":
            raise SystemExit("strict mode refused: TEST runs both tasks in one run (--task all)")
        rp = os.path.join(rdir, "RUN.json")
        if os.path.exists(rp):
            raise SystemExit(f"strict mode refused: already run ({rp}); TEST runs exactly once")
        if os.path.isdir(rdir) and any(fn.endswith(".npz") for _, _, fs in os.walk(rdir) for fn in fs):
            raise SystemExit(f"strict mode refused: {rdir} already holds forecast files")
        if a.models or a.force:
            raise SystemExit("strict mode refused: uses the pre-registered model lists (no --models/--force)")
        check_data_pins()
        fz = frozen() or {"models": [], "note": "rehearsal before freeze"}
        extra = [m for m in fz.get("models", [])]
        task_models = {"storage": STORAGE_MODELS + [m for m in extra if m not in STORAGE_MODELS],
                       "price": PRICE_MODELS + [m for m in extra if m not in PRICE_MODELS]}
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "tools/gas", "docs/gas_prereg.md"],
                               capture_output=True, text=True).stdout.strip()
        if dirty and a.split == "test":
            raise SystemExit(f"strict mode refused: uncommitted changes:\n{dirty}")
        metas = preflight(task_models)
        run_rec = {"status": "started", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_head(),
                   "tools_gas_dirty": dirty, "frozen": fz, "frozen_sha256": sha_file(FROZEN) if os.path.exists(FROZEN) else None,
                   "data_pins": DATA_PINS, "alma_pins": ALMA_PINS, "models": task_models, "model_meta": metas,
                   "split": a.split, "rehearsal": a.rehearse, "argv": sys.argv}
        os.makedirs(rdir, exist_ok=True)
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    else:
        sel = [m for m in a.models.split(",") if m]
        task_models = {"storage": sel or STORAGE_MODELS, "price": sel or PRICE_MODELS}
        if a.split == "test" and not os.path.exists(os.path.join(split_dir("test"), "RUN.json")):
            raise SystemExit("--metrics-only on test needs a completed TEST run")
    path = None
    try:
        if not a.metrics_only:
            if "storage" in tasks:
                forecast_storage(a.split, task_models["storage"], force=a.force, strict=strict)
            if "price" in tasks:
                forecast_price(a.split, task_models["price"], force=a.force, strict=strict)
        path = os.path.join(split_dir(a.split), "metrics.json")
        res = json.load(open(path)) if os.path.exists(path) and a.task != "all" else {}
        res.update(split=a.split, commit=git_head(), scored=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rehearsal=a.rehearse,
                   data_sha256={rel: sha_file(os.path.join(ROOT, rel)) for rel in
                                ("data/gas/daily.csv", "data/gas/weekly.csv", "data/gas/cpc_obs.csv", "data/gas/cpc_fcst.csv")})
        if "storage" in tasks:
            res["storage"] = score_storage(a.split)
        if "price" in tasks:
            res["price"] = score_price(a.split)
        os.makedirs(split_dir(a.split), exist_ok=True)
        json.dump(res, open(path, "w"), indent=1, default=float)
        write_report(res, os.path.join(split_dir(a.split), "report.md"))
    except BaseException as e:
        if run_rec is not None:
            import traceback
            run_rec.update(status="failed", failed=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
            json.dump(run_rec, open(rp, "w"), indent=1, default=str)
        raise
    if run_rec is not None:
        rdir = split_dir(a.split)
        hashes = {os.path.relpath(os.path.join(dp, fn), ROOT): sha_file(os.path.join(dp, fn))
                  for dp, _, fs in os.walk(rdir) for fn in sorted(fs) if fn.endswith(".npz") or fn == "metrics.json"}
        json.dump(hashes, open(os.path.join(rdir, "hashes.json"), "w"), indent=1, sort_keys=True)
        run_rec.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       hashes_sha256=sha_file(os.path.join(rdir, "hashes.json")))
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    print(f"metrics -> {path}")


if __name__ == "__main__":
    main()
