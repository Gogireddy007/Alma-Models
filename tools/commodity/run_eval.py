"""
Commodity walk-forward evaluation (docs/commodity_prereg.md is binding).

    nice -n 10 .venv/bin/python tools/commodity/run_eval.py --split val [--models a,b] [--series wti,inv]
    nice -n 10 .venv/bin/python tools/commodity/run_eval.py --split val --metrics-only
    nice -n 10 .venv/bin/python tools/commodity/run_eval.py --split test     # ONCE; needs runs/commodity/FROZEN.json

Forecasts: runs/commodity/<split>/<series>/<model>.npz (q [n_origins, 13, H], origin indices, meta). A cached file is
reused only if its origin list matches. Metrics: runs/commodity/<split>/metrics.json (+ metrics.md).
"""
import os
import gc
import sys
import json
import time
import argparse
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data as D          # noqa: E402
import metrics as MT      # noqa: E402

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
IMED, I10, I90 = QS.index(0.5), QS.index(0.1), QS.index(0.9)
CTX = 1024
H_DAILY, H_WEEKLY = (1, 5, 20), (1, 4)
OUT = os.path.join(ROOT, "runs", "commodity")
FROZEN = os.path.join(OUT, "FROZEN.json")
KRONOS_STRIDE = 5          # amendment A1 (validation timing): kronos on every 5th origin of VAL/TEST

# amendment A2 (audit B1): the s3/s4 direct twins are dropped (direct_s3 is a half-trained checkpoint, direct_s4 does
# not exist); s1-s4 rollouts are descriptive only. Every Alma checkpoint is pinned by sha256 and verified in strict mode.
PREREG_MODELS = ["alma_s0", "alma_s0_tta", "alma_direct_s0", "alma_direct_s0_tta", "alma_s1", "alma_s2", "alma_s3",
                 "alma_s4", "chronos2", "chronos_bolt_base", "tirex", "timestral_5m", "kronos_small", "auto_ets",
                 "auto_arima", "rw_gauss"]
ALMA_PINS = {
    "data/ckpt/wm_ssm_none_s0.pt": "3c959a64099550fa729e19ae2645dff3e8eff695d6ca999e9eeaabf4217ec422",
    "data/ckpt/wm_ssm_direct_s0.pt": "be2faf59694f334db56614ee767b9fbebdb26dde94800d0d7764a8d1316fc9be",
    "data/ckpt/wm_ssm_none_s1.pt": "92a6301a2f2acff345526e330c5d33e57063187c4517faa07941709be02a9a33",
    "data/ckpt/wm_ssm_none_s2.pt": "86a8aa4590efc3688ae3e9ba6c3b91a03c9bdcfd67bbd8dde25338d2c187efc5",
    "data/ckpt/wm_ssm_none_s3.pt": "cd14fbf3d0b370fef62046971e9b40f8c30bb98ca1319d21bb55518b137d4308",
    "data/ckpt/wm_ssm_none_s4.pt": "08c2574ffc13592c3319371945e9a3bf1cfb6cac942ebfb39140f09b23029bb6",
}
RUN_DIR = {}                # split -> output directory (rehearsal redirects VAL to a fresh directory)


def split_dir(split):
    return RUN_DIR.get(split, os.path.join(OUT, split))
CONTAMINATED = {"timestral_5m": "released 2026-10-15; training data may include the TEST period"}
DAILY_ONLY = {"kronos_small"}
SUBSAMPLED = {"kronos_small": "every 5th origin (amendment A1); its own accuracy/CI rest on that subsample"}
PRIMARY_SERIES = ("wti", "brent", "hh", "inv")


# --------------------------------------------------------------------------- windows
def window(split, name):
    d, v = D.load(name)
    if split == "dev":
        lo, hi, score_end, stride = D.DEV_START, D.TRAIN_END, D.TRAIN_END, (1 if name == "inv" else 5)
    elif split == "val":
        lo, hi, score_end, stride = D.VAL_START, D.VAL_END, D.VAL_END, 1
    elif split == "test":
        lo, hi, score_end, stride = D.TEST_START, pd.Timestamp(d[-1]), pd.Timestamp(d[-1]), 1
    else:
        raise SystemExit(split)
    idx = np.where((d >= np.datetime64(lo)) & (d <= np.datetime64(hi)))[0]
    idx = idx[idx >= CTX][::stride]
    return d, v, idx, np.datetime64(score_end)


# --------------------------------------------------------------------------- forecasting
def forecast(split, name, model_names, force=False, strict=False):
    import models as M
    d, v, idx, _ = window(split, name)
    H = H_WEEKLY[-1] if name == "inv" else H_DAILY[-1]
    sdir = os.path.join(split_dir(split), name)
    os.makedirs(sdir, exist_ok=True)
    for mn in model_names:
        if name == "inv" and mn in DAILY_ONLY:
            continue
        if mn.startswith("kronos") and name != "wti":
            continue
        oi = idx[::KRONOS_STRIDE] if (mn.startswith("kronos") and split != "dev") else idx
        dst = os.path.join(sdir, f"{mn}.npz")
        if strict and os.path.exists(dst):
            raise SystemExit(f"strict mode: {dst} already exists (no cached forecasts allowed)")
        if os.path.exists(dst) and not force:
            z = np.load(dst)
            if np.array_equal(z["origins"], oi):
                print(f"[{split}/{name}] {mn}: cached", flush=True)
                continue
        try:
            p = M.get(mn)
        except (KeyError, RuntimeError) as e:
            if strict:
                raise
            print(f"[{split}/{name}] {mn}: SKIP ({e})", flush=True)
            continue
        if strict and "ckpt" in getattr(p, "meta", {}):
            ck = p.meta["ckpt"]
            if ALMA_PINS.get(ck) != p.meta.get("ckpt_sha256"):
                raise SystemExit(f"strict mode: {ck} sha256 {p.meta.get('ckpt_sha256')} != pinned {ALMA_PINS.get(ck)}")
        ctxs = [v[max(0, i - CTX):i] for i in oi]
        kw = {}
        if getattr(p, "wants_freq", False):
            kw["freq"] = "W" if name == "inv" else "B"
        if getattr(p, "wants_dates", False):
            kw["dates"] = [d[max(0, i - CTX):i] for i in oi]
        t0 = time.time()
        if hasattr(p, "batch"):
            q = np.asarray(p.batch(ctxs, H, QS, **kw), float)
        else:
            q = np.stack([np.asarray(p(c, H, QS, **{k: w[j] if k == "dates" else w for k, w in kw.items()}), float)
                          for j, c in enumerate(ctxs)])
        dt = time.time() - t0
        if q.shape != (len(oi), len(QS), H):
            raise SystemExit(f"{mn}: shape {q.shape} != {(len(oi), len(QS), H)}")
        meta = dict(getattr(p, "meta", {}), model=mn, series=name, split=split, H=H, quantiles=QS,
                    seconds=round(dt, 1), n=len(oi), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        np.savez(dst + ".tmp.npz", q=q, origins=oi, meta=np.array(json.dumps(meta)))
        os.replace(dst + ".tmp.npz", dst)
        nf = int((~np.isfinite(q[:, I9])).sum())
        print(f"[{split}/{name}] {mn}: n={len(oi)} {dt:.1f}s nonfinite(0.1..0.9)={nf}", flush=True)
        del p, q
        gc.collect()


# --------------------------------------------------------------------------- targets and baselines
def logret(v):
    with np.errstate(invalid="ignore", divide="ignore"):
        lv = np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)
    return np.diff(lv)                                      # r[j] = log v[j+1] - log v[j]


def train_majority(name, v, d):
    """TRAIN majority class per target (up-fraction over all TRAIN pairs)."""
    tr = d <= np.datetime64(D.TRAIN_END)
    vt = v[tr]
    out = {}
    for h in (H_WEEKLY if name == "inv" else H_DAILY):
        ch = vt[h:] - vt[:-h]
        ch = ch[ch != 0]
        out[f"T1_{h}"] = float((ch > 0).mean())
    if name != "inv":
        r = logret(vt)
        ys = []
        for t in range(20, len(r) - 20):
            a, b = r[t - 20:t], r[t:t + 20]
            if np.isfinite(a).all() and np.isfinite(b).all():
                ys.append(np.std(b, ddof=1) > np.std(a, ddof=1))
        out["T2"] = float(np.mean(ys))
    return out


_GARCH = {}


def garch_series(name, v, d):
    """GARCH(1,1) fitted on TRAIN log returns 2000-01-01..2025-06-30; filtered over all data. Returns s2_next[i] =
    predicted variance of r between obs i-1 and i given returns up to obs i-1 (i = origin index)."""
    if name in _GARCH:
        return _GARCH[name]
    r = logret(v)
    rd = d[1:]
    m = (rd >= np.datetime64("2000-01-01")) & (rd <= np.datetime64(D.TRAIN_END)) & np.isfinite(r)
    th = MT.garch_fit(r[m])
    rr = np.where(np.isfinite(r), r, 0.0)                   # returns touching a non-positive price: treated as 0
    s2 = MT.garch_filter(rr, th, float(np.var(r[m])))       # s2[j] = var(r[j] | r[:j]); r[j] ends at obs j+1
    # origin i (last obs i-1): next return is r[i-1] -> s2[i-1]
    _GARCH[name] = (th, s2)
    return _GARCH[name]


# --------------------------------------------------------------------------- scoring
def load_preds(split, name):
    sdir = os.path.join(split_dir(split), name)
    out = {}
    if not os.path.isdir(sdir):
        return out
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            z = np.load(os.path.join(sdir, f))
            out[f[:-4]] = (z["q"], z["origins"], json.loads(str(z["meta"])))
    return out


def dir_stats(pred, act, sub, step, block):
    """Amendment A3 (B4): overlap-robust statistics. `sub` marks the non-overlapping subsample (window position
    divisible by `step`: h for T1/T3, 40 for T2). Primary inference = exact binomial, Wilson CI and PT on `sub`;
    the full-sample accuracy gets a circular block-bootstrap CI (block = 2h; 40 for T2). iid full-sample Wilson/PT
    are kept only as 'naive' fields."""
    n = len(act)
    hit = (pred == act).astype(float)
    k = int(hit.sum())
    nlo, nhi = MT.wilson(k, n)
    blo, bhi = MT.block_boot_ci(hit, block) if n else (np.nan, np.nan)
    ns = int(sub.sum())
    ks = int(hit[sub].sum())
    slo, shi = MT.wilson(ks, ns)
    _, pps = MT.pesaran_timmermann(pred[sub], act[sub])
    _, ppn = MT.pesaran_timmermann(pred, act)
    return {"n": n, "acc": k / n if n else np.nan, "boot_lo": blo, "boot_hi": bhi, "block": block,
            "n_nonoverlap": ns, "acc_nonoverlap": ks / ns if ns else np.nan, "ci_lo": slo, "ci_hi": shi,
            "p_binom": MT.binom_p(ks, ns), "p_pt": pps, "step": step,
            "naive_wilson_lo": nlo, "naive_wilson_hi": nhi, "naive_p_pt": ppn,
            "frac_up_pred": float(pred.mean()) if n else np.nan}


def pair_test(la, lb, sub, h, wide_lag):
    """Model a vs b on a loss series (0/1 for direction, QS for T4); negative diff = a better.
    Primary p (amendment A3): h == 1 -> DM-HLN (lag 0) on all origins; h > 1 or T2 -> on the non-overlapping
    subsample: exact McNemar for 0/1 losses, DM-HLN (lag 0) for QS. Supplementary: DM-HLN with Bartlett lag wide_lag."""
    d = la - lb
    _, st_w, p_w = MT.dm_hln(d, h, lag=wide_lag)
    binary = np.isin(la, (0.0, 1.0)).all() and np.isin(lb, (0.0, 1.0)).all()
    if h == 1:
        md, st, p = MT.dm_hln(d, 1)
        n01 = n10 = None
    elif binary:
        n01, n10, p = MT.mcnemar_exact(la[sub] > 0, lb[sub] > 0)
        st = None
    else:
        _, st, p = MT.dm_hln(d[sub], 1)
        n01 = n10 = None
    return {"n": int(len(d)), "n_sub": int(sub.sum()), "mean_diff": float(np.mean(d)) if len(d) else np.nan,
            "p": p, "stat": st, "n01": n01, "n10": n10, "p_dm_wide": p_w, "dm_wide": st_w, "wide_lag": wide_lag}


def score_series(split, name):
    d, v, idx, score_end = window(split, name)
    preds = load_preds(split, name)
    hs = H_WEEKLY if name == "inv" else H_DAILY
    maj = train_majority(name, v, d)
    res = {"series": name, "split": split, "n_origins": int(len(idx)), "train_majority_upfrac": maj,
           "first_origin": str(d[idx[0]])[:10] if len(idx) else None,
           "last_origin": str(d[idx[-1]])[:10] if len(idx) else None, "T1": {}, "T2": {}, "T4": {}}
    r = logret(v)
    # ---------------- T1 / T3 (direction)
    for h in hs:
        ok = np.array([(i - 1 + h < len(v)) and d[min(i - 1 + h, len(v) - 1)] <= score_end for i in idx])
        oi, posn = idx[ok], np.arange(len(idx))[ok]
        last, fut = v[oi - 1], v[oi - 1 + h]
        nz = fut != last
        oi, posn, last, fut = oi[nz], posn[nz], last[nz], fut[nz]
        subm = posn % h == 0
        act = (fut > last).astype(int)
        majc = int(maj[f"T1_{h}"] > 0.5)
        cand = {"majority": np.full(len(oi), majc), "momentum": (v[oi - 1] > v[oi - 1 - h]).astype(int)}
        if name == "inv":
            seas = []
            for i in oi:
                ch = [v[i - 1 - 52 * k + h] - v[i - 1 - 52 * k] for k in range(1, 6)]
                seas.append(int(np.mean(ch) > 0))
            cand["seasonal"] = np.array(seas)
        base_names = list(cand)
        for mn, (q, qo, meta) in preds.items():
            pos = {o: j for j, o in enumerate(qo)}
            sel = np.array([o in pos for o in oi])
            med = np.full(len(oi), np.nan)
            med[sel] = [q[pos[o], IMED, h - 1] for o in oi[sel]]
            p = np.where(med > last, 1, np.where(med < last, 0, majc))
            p = np.where(np.isfinite(med), p, -1)
            cand[mn] = p
        out = {"n_ties_excluded": int((~nz).sum()), "base_rate_up": float(act.mean()) if len(act) else np.nan,
               "models": {}, "pairwise": {}}
        for mn, p in cand.items():
            s = p >= 0
            out["models"][mn] = dict(dir_stats(p[s], act[s], subm[s], h, 2 * h),
                                     kind="baseline" if mn in base_names else "model",
                                     contaminated=CONTAMINATED.get(mn), subsample=SUBSAMPLED.get(mn))
        names = list(cand)
        for a in names:
            for b in names:
                if a == b:
                    continue
                s = (cand[a] >= 0) & (cand[b] >= 0)
                la = (cand[a][s] != act[s]).astype(float)
                lb = (cand[b][s] != act[s]).astype(float)
                pw = pair_test(la, lb, subm[s], h, 2 * h)
                pw["acc_diff"] = -pw["mean_diff"]
                out["pairwise"][f"{a}|{b}"] = pw
        res["T1"][str(h)] = out
    # ---------------- T2 (vol direction), daily only
    if name != "inv":
        ok = np.array([(i + 19 < len(v)) and d[min(i + 19, len(v) - 1)] <= score_end for i in idx])
        oi, posn = idx[ok], np.arange(len(idx))[ok]
        past = np.array([np.std(r[i - 21:i - 1], ddof=1) for i in oi])        # 20 returns ending at obs i-1
        futv = np.array([np.std(r[i - 1:i + 19], ddof=1) for i in oi])        # 20 returns over obs i..i+19
        fin = np.isfinite(past) & np.isfinite(futv)
        oi, posn, past, futv = oi[fin], posn[fin], past[fin], futv[fin]
        subm = posn % 40 == 0
        act = (futv > past).astype(int)
        th, s2 = garch_series(name, v, d)
        roll = lambda i: np.array([np.std(r[j - 20:j], ddof=1) for j in range(i - 1 - 251, i)])  # noqa: E731
        cand = {"majority": np.full(len(oi), int(maj["T2"] > 0.5)),
                "vol_meanrev": np.array([int(past[k] < np.nanmedian(roll(i))) for k, i in enumerate(oi)]),
                "vol_persist": np.array([int(past[k] > np.std(r[i - 41:i - 21], ddof=1)) for k, i in enumerate(oi)]),
                "garch": np.array([int(MT.garch_mean_var(s2[i - 1], th) > past[k] ** 2) for k, i in enumerate(oi)])}
        base_names = list(cand)
        last = v[oi - 1]
        for mn, (q, qo, meta) in preds.items():
            pos = {o: j for j, o in enumerate(qo)}
            p = np.full(len(oi), -1)
            for k, o in enumerate(oi):
                if o in pos:
                    qq = np.maximum(q[pos[o]], 0.01)
                    kk = np.arange(1, 21)
                    sm = np.mean((np.log(qq[I90, :20]) - np.log(qq[I10, :20])) / (2 * 1.2816 * np.sqrt(kk)))
                    p[k] = int(sm > past[k]) if np.isfinite(sm) else -1
            cand[mn] = p
        out = {"base_rate_up": float(act.mean()) if len(act) else np.nan, "models": {}, "pairwise": {}}
        for mn, p in cand.items():
            s = p >= 0
            out["models"][mn] = dict(dir_stats(p[s], act[s], subm[s], 40, 40),
                                     kind="baseline" if mn in base_names else "model",
                                     contaminated=CONTAMINATED.get(mn), subsample=SUBSAMPLED.get(mn))
        for a in cand:
            for b in cand:
                if a != b:
                    s = (cand[a] >= 0) & (cand[b] >= 0)
                    pw = pair_test((cand[a][s] != act[s]).astype(float), (cand[b][s] != act[s]).astype(float),
                                   subm[s], 40, 40)
                    pw["acc_diff"] = -pw["mean_diff"]
                    out["pairwise"][f"{a}|{b}"] = pw
        res["T2"]["20"] = out
    # ---------------- T4 (log-price quantile score), daily series only
    for h in (hs if name != "inv" else ()):
        ok = np.array([(i - 1 + h < len(v)) and d[min(i - 1 + h, len(v) - 1)] <= score_end for i in idx])
        oi, posn = idx[ok], np.arange(len(idx))[ok]
        y = v[oi - 1 + h]
        pos_ok = y > 0
        oi, posn, y = oi[pos_ok], posn[pos_ok], np.log(y[pos_ok])
        subm = posn % h == 0
        losses, cov = {}, {}
        for mn, (q, qo, meta) in preds.items():
            pos = {o: j for j, o in enumerate(qo)}
            L = np.full(len(oi), np.nan)
            c = {80: np.full(len(oi), np.nan), 90: np.full(len(oi), np.nan), 95: np.full(len(oi), np.nan)}
            sel = np.array([o in pos for o in oi])
            if sel.any():
                Q = np.stack([q[pos[o], :, h - 1] for o in oi[sel]])
                lq = np.log(np.maximum(Q, 0.01))
                L[sel] = MT.pinball(y[sel], lq[:, I9], [QS[i] for i in I9])
                for lev, (a, b) in {80: (0.1, 0.9), 90: (0.05, 0.95), 95: (0.025, 0.975)}.items():
                    qa, qb = lq[:, QS.index(a)], lq[:, QS.index(b)]
                    if np.isfinite(qa).all() and np.isfinite(qb).all():
                        c[lev][sel] = ((y[sel] >= qa) & (y[sel] <= qb)).astype(float)
            losses[mn], cov[mn] = L, c
        out = {"models": {}, "pairwise": {}}
        rw = losses.get("rw_gauss")
        for mn, L in losses.items():
            s = np.isfinite(L) & (np.isfinite(rw) if rw is not None else True)
            row = {"n": int(s.sum()), "qs": float(np.mean(L[s])) if s.any() else np.nan,
                   "contaminated": CONTAMINATED.get(mn), "subsample": SUBSAMPLED.get(mn)}
            if rw is not None and s.any():
                row["qs_rw"] = float(np.mean(rw[s]))
                row["skill_vs_rw"] = 1 - row["qs"] / row["qs_rw"]
                pw = pair_test(L[s], rw[s], subm[s], h, 2 * h)
                row["p_vs_rw"], row["p_vs_rw_dm_wide"], row["n_sub"] = pw["p"], pw["p_dm_wide"], pw["n_sub"]
            for lev in (80, 90, 95):
                cc = cov[mn][lev][s]
                row[f"cov{lev}"] = float(np.mean(cc)) if np.isfinite(cc).all() and len(cc) else None
            out["models"][mn] = row
        for a in losses:
            for b in losses:
                if a != b:
                    s = np.isfinite(losses[a]) & np.isfinite(losses[b])
                    pw = pair_test(losses[a][s], losses[b][s], subm[s], h, 2 * h)
                    pw["qs_diff"] = pw["mean_diff"]
                    out["pairwise"][f"{a}|{b}"] = pw
        res["T4"][str(h)] = out
    return res


def add_holm(res):
    """Holm within each (series, target) family over all model x baseline x horizon comparisons."""
    for tgt in ("T1", "T2"):
        keys, ps = [], []
        for h, blk in res[tgt].items():
            bases = [m for m, r in blk["models"].items() if r["kind"] == "baseline"]
            for m, r in blk["models"].items():
                if r["kind"] != "model":
                    continue
                for b in bases:
                    keys.append((h, m, b))
                    ps.append(blk["pairwise"][f"{m}|{b}"]["p"])
        adj = MT.holm(ps)
        for (h, m, b), pa in zip(keys, adj):
            res[tgt][h]["pairwise"][f"{m}|{b}"]["p_holm_family"] = float(pa)
    keys, ps = [], []
    for h, blk in res["T4"].items():
        for m, r in blk["models"].items():
            if m != "rw_gauss" and "p_vs_rw" in r:
                keys.append((h, m))
                ps.append(r["p_vs_rw"])
    for (h, m), pa in zip(keys, MT.holm(ps)):
        res["T4"][h]["models"][m]["p_vs_rw_holm_family"] = float(pa)
    return res


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def preflight(models):
    """Strict mode, before RUN.json: every model resolves, Alma checkpoints complete and sha-pinned, no architect
    module failed to import. Raises on any problem."""
    import models as M
    if M.LOAD_ERRORS:
        raise SystemExit(f"preflight: architect module import errors: {M.LOAD_ERRORS}")
    metas = {}
    for mn in models:
        p = M.get(mn)                           # KeyError / RuntimeError (incomplete ckpt) propagate
        meta = dict(getattr(p, "meta", {}))
        if "ckpt" in meta and meta["ckpt"] in ALMA_PINS and ALMA_PINS[meta["ckpt"]] != meta.get("ckpt_sha256"):
            raise SystemExit(f"preflight: {meta['ckpt']} sha256 mismatch vs pin")
        if mn.startswith("alma") and meta.get("ckpt") not in ALMA_PINS:
            raise SystemExit(f"preflight: {mn} checkpoint {meta.get('ckpt')} not pinned")
        metas[mn] = meta
        del p
    return metas


def sha_file(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "val", "test"])
    ap.add_argument("--models", default="")
    ap.add_argument("--series", default=",".join(PRIMARY_SERIES))
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rehearse", action="store_true",
                    help="run the exact TEST procedure on VAL into a fresh runs/commodity/rehearsal_val/")
    a = ap.parse_args()
    series = [s for s in a.series.split(",") if s]
    strict = (a.split == "test" or a.rehearse) and not a.metrics_only
    if a.rehearse:
        if a.split != "val":
            raise SystemExit("--rehearse only with --split val")
        RUN_DIR["val"] = os.path.join(OUT, "rehearsal_val")
    run_rec, rp = None, None
    if strict:
        rdir = split_dir(a.split)
        if not os.path.exists(FROZEN) and not a.rehearse:
            raise SystemExit("strict mode refused: runs/commodity/FROZEN.json missing (candidate not frozen)")
        rp = os.path.join(rdir, "RUN.json")
        if os.path.exists(rp):
            raise SystemExit(f"strict mode refused: already run ({rp}); TEST runs exactly once")
        if os.path.isdir(rdir) and any(f.endswith(".npz") for _, _, fs in os.walk(rdir) for f in fs):
            raise SystemExit(f"strict mode refused: {rdir} already holds forecast files")
        if a.models or a.force:
            raise SystemExit("strict mode refused: uses the pre-registered model list (no --models/--force)")
        fz = json.load(open(FROZEN)) if os.path.exists(FROZEN) else {"models": [], "note": "rehearsal before freeze"}
        models = PREREG_MODELS + list(fz.get("models", []))
        for ck, sh in (fz.get("alma_sha256") or {}).items():
            if ALMA_PINS.get(ck) != sh:
                raise SystemExit(f"strict mode refused: FROZEN.json alma_sha256 {ck} != run_eval pin")
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "tools/commodity", "docs/commodity_prereg.md"],
                               capture_output=True, text=True).stdout.strip()
        if dirty and a.split == "test":
            raise SystemExit(f"strict mode refused: uncommitted changes:\n{dirty}")
        metas = preflight(models)
        run_rec = {"status": "started", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_head(),
                   "tools_commodity_dirty": dirty, "frozen": fz, "frozen_sha256": sha_file(FROZEN) if os.path.exists(FROZEN) else None,
                   "alma_pins": ALMA_PINS, "models": models, "model_meta": metas, "series": series,
                   "split": a.split, "rehearsal": a.rehearse, "argv": sys.argv}
        os.makedirs(rdir, exist_ok=True)
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)   # written first: a crashed run still counts
    else:
        models = [m for m in a.models.split(",") if m] or PREREG_MODELS
        if a.split == "test" and not os.path.exists(os.path.join(split_dir("test"), "RUN.json")):
            raise SystemExit("--metrics-only on test needs a completed TEST run")
    try:
        if not a.metrics_only:
            for s in series:
                forecast(a.split, s, models, force=a.force, strict=strict)
        allres = {"split": a.split, "commit": git_head(), "scored": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "rehearsal": a.rehearse, "series": {}}
        for s in series:
            allres["series"][s] = add_holm(score_series(a.split, s))
        path = os.path.join(split_dir(a.split), "metrics.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(allres, open(path, "w"), indent=1, default=float)
    except BaseException as e:
        if run_rec is not None:
            import traceback
            run_rec.update(status="failed", failed=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
            json.dump(run_rec, open(rp, "w"), indent=1, default=str)
        raise
    if run_rec is not None:
        rdir = split_dir(a.split)
        hashes = {os.path.relpath(os.path.join(dp, f), ROOT): sha_file(os.path.join(dp, f))
                  for dp, _, fs in os.walk(rdir) for f in sorted(fs) if f.endswith(".npz") or f == "metrics.json"}
        json.dump(hashes, open(os.path.join(rdir, "hashes.json"), "w"), indent=1, sort_keys=True)
        run_rec.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       hashes_sha256=sha_file(os.path.join(rdir, "hashes.json")))
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    print(f"metrics -> {path}")


if __name__ == "__main__":
    main()
