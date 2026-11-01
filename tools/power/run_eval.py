"""
Alma-Power walk-forward evaluation (docs/power_prereg.md is binding).

    nice -n 10 .venv/bin/python tools/power/run_eval.py --split val [--models a,b]
    nice -n 10 .venv/bin/python tools/power/run_eval.py --split val --metrics-only
    nice -n 10 .venv/bin/python tools/power/run_eval.py --split val --rehearse      # TEST procedure on VAL
    nice -n 10 .venv/bin/python tools/power/run_eval.py --split test                # ONCE; needs runs/power/FROZEN.json

Forecasts: runs/power/<split>/<model>.npz (q [n_days, 13, 24], days = target day indices into data.panel(),
dates, meta). A cached file is reused only if its day list matches (never in strict mode).
Metrics: runs/power/<split>/metrics.json; report: runs/power/<split>/report.md (tools/power/report.py).
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
import data as D            # noqa: E402
import metrics as MT        # noqa: E402

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
IMED = QS.index(0.5)
CTX_DAYS = 736
CHUNK = 32
OUT = os.path.join(ROOT, "runs", "power")
FROZEN = os.path.join(OUT, "FROZEN.json")
PEAK = np.arange(8, 20)
OFFPEAK = np.setdiff1d(np.arange(24), PEAK)
HOUR_BLOCKS = {"00-03": range(0, 4), "04-07": range(4, 8), "08-11": range(8, 12), "12-15": range(12, 16),
               "16-19": range(16, 20), "20-23": range(20, 24)}
KRONOS_STRIDE = 5

PREREG_MODELS = ["naive_d", "naive_w", "naive_epf", "lear", "lear_nocov", "arx", "chronos2", "chronos2_cov",
                 "chronos_bolt_base", "tirex", "timestral_5m", "kronos_small", "alma_s0", "alma_direct_s0"]
KIND = {"naive_d": "baseline", "naive_w": "baseline", "naive_epf": "baseline", "majority": "baseline",
        "dow_majority": "baseline", "lear": "competitor", "lear_nocov": "competitor", "arx": "competitor",
        "chronos2": "fm", "chronos2_cov": "fm", "chronos_bolt_base": "fm", "tirex": "fm", "timestral_5m": "fm",
        "kronos_small": "fm", "alma_s0": "alma", "alma_direct_s0": "alma"}
CONTAMINATED = {"timestral_5m": "released 2026-10-15; training data may include the TEST period"}
SUBSAMPLED = {"kronos_small": f"every {KRONOS_STRIDE}th target day (VAL/TEST)"}
ALMA_PINS = {
    "data/ckpt/wm_ssm_none_s0.pt": "3c959a64099550fa729e19ae2645dff3e8eff695d6ca999e9eeaabf4217ec422",
    "data/ckpt/wm_ssm_direct_s0.pt": "be2faf59694f334db56614ee767b9fbebdb26dde94800d0d7764a8d1316fc9be",
}
DATA_SHA256 = "79a2c96e90b51ea7f6db1879830e63615b1f82b3a97392953e1b08e651f76dea"   # data/power/hourly.csv (amendment A1)
REF_P4 = "naive_epf"
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


def frozen_models():
    if not os.path.exists(FROZEN):
        return []
    return list(json.load(open(FROZEN)).get("models", []))


def kind_of(mn):
    if mn in KIND:
        return KIND[mn]
    return "candidate" if mn in frozen_models() else "model"


# --------------------------------------------------------------------------- forecasting
def forecast(split, model_names, force=False, strict=False):
    import models as M
    dates, P, C, flags = D.panel()
    idx = D.split_days(split)
    sdir = split_dir(split)
    os.makedirs(sdir, exist_ok=True)
    for mn in model_names:
        days = idx[::KRONOS_STRIDE] if (mn.startswith("kronos") and split != "dev") else idx
        dst = os.path.join(sdir, f"{mn}.npz")
        if strict and os.path.exists(dst):
            raise SystemExit(f"strict mode: {dst} already exists (no cached forecasts allowed)")
        if os.path.exists(dst) and not force:
            z = np.load(dst)
            if np.array_equal(z["days"], days):
                print(f"[{split}] {mn}: cached", flush=True)
                continue
        try:
            p = M.get(mn)
        except (KeyError, RuntimeError) as e:
            if strict:
                raise
            print(f"[{split}] {mn}: SKIP ({e})", flush=True)
            continue
        if strict and "ckpt" in getattr(p, "meta", {}) and p.meta["ckpt"] in ALMA_PINS:
            if ALMA_PINS[p.meta["ckpt"]] != p.meta.get("ckpt_sha256"):
                raise SystemExit(f"strict mode: {p.meta['ckpt']} sha256 mismatch vs pin")
        t0 = time.time()
        qs = []
        for s in range(0, len(days), CHUNK):
            ws = [D.build_context(int(i), CTX_DAYS, dates, P, C, flags) for i in days[s:s + CHUNK]]
            prices = [w["prices"] for w in ws]
            covs = [w["covs"] for w in ws]
            kw = {}
            if getattr(p, "wants_dates", False):
                kw["dates"] = [{"ctx_dates": w["ctx_dates"], "target_day": w["target_day"]} for w in ws]
            if hasattr(p, "batch"):
                q = np.asarray(p.batch(prices, covs, 24, QS, **kw), float)
            else:
                q = np.stack([np.asarray(p(pr, cv, 24, QS, **({"dates": kw["dates"][j]} if kw else {})), float)
                              for j, (pr, cv) in enumerate(zip(prices, covs))])
            if q.shape != (len(ws), len(QS), 24):
                raise SystemExit(f"{mn}: shape {q.shape} != {(len(ws), len(QS), 24)}")
            qs.append(q)
            print(f"[{split}] {mn}: {min(s + CHUNK, len(days))}/{len(days)} {time.time() - t0:.0f}s", flush=True)
            del ws, prices, covs
        q = np.concatenate(qs)
        dt = time.time() - t0
        meta = dict(getattr(p, "meta", {}), model=mn, split=split, quantiles=QS, ctx_days=CTX_DAYS,
                    uses_covariates=getattr(p, "uses_covariates", None), seconds=round(dt, 1), n=len(days),
                    created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        np.savez(dst + ".tmp.npz", q=q, days=days, dates=dates[days].astype(str), meta=np.array(json.dumps(meta, default=str)))
        os.replace(dst + ".tmp.npz", dst)
        nf = int((~np.isfinite(q[:, I9])).sum())
        print(f"[{split}] {mn}: n={len(days)} {dt:.1f}s nonfinite(0.1..0.9)={nf}", flush=True)
        del p, q, qs
        gc.collect()


def load_preds(split):
    sdir = split_dir(split)
    out = {}
    if not os.path.isdir(sdir):
        return out
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            z = np.load(os.path.join(sdir, f))
            out[f[:-4]] = (z["q"], z["days"], json.loads(str(z["meta"])))
    return out


# --------------------------------------------------------------------------- labels
def weekday(d):
    return int((np.datetime64(d, "D").astype(int) + 3) % 7)


def day_ok(P, flags, i):
    """All slots present except the structural spring-forward slot."""
    need = np.ones(24, bool)
    if i in set(flags["dst_gap"].tolist()):
        need[2] = False
    return bool(np.isfinite(P[i][need]).all())


def spread(x):
    return np.nanmean(x[PEAK]) - np.nanmean(x[OFFPEAK])


def dev_majority(dates, P, flags):
    """Up-fractions on DEV (delivery days <= 2025-06-30): overall and by weekday (P1/P3) / weekday x slot (P2)."""
    end = np.datetime64(D.TRAIN_END.date(), "D")
    p1, p3, wd = [], [], []
    p2 = np.zeros((7, 24))
    n2 = np.zeros((7, 24))
    up2 = n2_all = 0
    for i in range(1, len(dates)):
        if dates[i] > end:
            break
        w = weekday(dates[i])
        if day_ok(P, flags, i) and day_ok(P, flags, i - 1):
            b1, b0 = np.nanmean(P[i]), np.nanmean(P[i - 1])
            s1, s0 = spread(P[i]), spread(P[i - 1])
            if b1 != b0:
                p1.append((w, b1 > b0))
            if s1 != s0:
                p3.append((w, s1 > s0))
        ok = np.isfinite(P[i]) & np.isfinite(P[i - 1]) & (P[i] != P[i - 1])
        up = (P[i] > P[i - 1]) & ok
        p2[w] += up
        n2[w] += ok
        up2 += up.sum()
        n2_all += ok.sum()
    out = {"P1": float(np.mean([u for _, u in p1])), "P3": float(np.mean([u for _, u in p3])),
           "P2": float(up2 / n2_all),
           "P1_wd": [float(np.mean([u for w, u in p1 if w == k])) for k in range(7)],
           "P3_wd": [float(np.mean([u for w, u in p3 if w == k])) for k in range(7)],
           "P2_wd_slot": (p2 / np.maximum(n2, 1)).tolist()}
    return out


# --------------------------------------------------------------------------- scoring
def score(split):
    dates, P, C, flags = D.panel()
    idx = D.split_days(split)
    preds = load_preds(split)
    maj = dev_majority(dates, P, flags)
    gap = set(flags["dst_gap"].tolist())
    res = {"split": split, "first_day": str(dates[idx[0]]), "last_day": str(dates[idx[-1]]), "n_days": int(len(idx)),
           "dev_majority": {k: maj[k] for k in ("P1", "P2", "P3", "P1_wd", "P3_wd")},
           "models": {}, "pairwise": {}, "excluded": {}}
    # ---------------- medians per model per day
    med, quant = {}, {}
    for mn, (q, days, meta) in preds.items():
        pos = {int(d): j for j, d in enumerate(days)}
        med[mn] = {i: q[pos[i], IMED] for i in idx if i in pos}
        quant[mn] = {i: q[pos[i]] for i in idx if i in pos}
    # ---------------- labels
    ok1 = [i for i in idx if day_ok(P, flags, i) and day_ok(P, flags, i - 1)]
    res["excluded"]["P1_P3_missing_days"] = int(len(idx) - len(ok1))
    lab = {"P1": {}, "P3": {}}
    for i in ok1:
        b1, b0 = np.nanmean(P[i]), np.nanmean(P[i - 1])
        if b1 != b0:
            lab["P1"][i] = int(b1 > b0)
        s1, s0 = spread(P[i]), spread(P[i - 1])
        if s1 != s0:
            lab["P3"][i] = int(s1 > s0)
    res["excluded"]["P1_ties"] = len(ok1) - len(lab["P1"])
    res["excluded"]["P3_ties"] = len(ok1) - len(lab["P3"])
    # P2 labels: per day arrays (NaN = not scored)
    lab2 = {}
    nties2 = 0
    for i in idx:
        ok = np.isfinite(P[i]) & np.isfinite(P[i - 1])
        tie = ok & (P[i] == P[i - 1])
        nties2 += int(tie.sum())
        y = np.where(ok & ~tie, (P[i] > P[i - 1]).astype(float), np.nan)
        lab2[i] = y
    res["excluded"]["P2_ties"] = nties2
    # ---------------- direction predictions
    def pred_dir(mn, tgt, i):
        w = weekday(dates[i])
        if mn == "majority":
            return int(maj[tgt] > 0.5) if tgt != "P2" else np.full(24, int(maj["P2"] > 0.5))
        if mn == "dow_majority":
            if tgt == "P2":
                return (np.array(maj["P2_wd_slot"][w]) > 0.5).astype(int)
            return int(maj[tgt + "_wd"][w] > 0.5)
        if i not in med[mn]:
            return None
        m = med[mn][i]
        avail = (np.arange(24) != 2) if i in gap else np.ones(24, bool)      # structural DST gap only
        if tgt == "P1":
            pv, rv = np.mean(m[avail]), np.nanmean(P[i - 1])
            return None if not np.isfinite(pv) else (int(pv > rv) if pv != rv else int(maj["P1"] > 0.5))
        if tgt == "P3":
            mm = np.where(avail, m, np.nan)
            pv, rv = spread(mm), spread(P[i - 1])
            return None if not np.isfinite(pv) else (int(pv > rv) if pv != rv else int(maj["P3"] > 0.5))
        r = P[i - 1]
        d = np.where(m > r, 1, np.where(m < r, 0, int(maj["P2"] > 0.5)))
        return np.where(np.isfinite(m), d, -1)

    cand = ["majority", "dow_majority"] + list(preds)
    for tgt in ("P1", "P3"):
        days_t = sorted(lab[tgt])
        act = np.array([lab[tgt][i] for i in days_t])
        hits = {}
        for mn in cand:
            pr = [pred_dir(mn, tgt, i) for i in days_t]
            s = np.array([p is not None for p in pr])
            if not s.any():
                continue
            pv = np.array([p for p in pr if p is not None])
            h = (pv == act[s]).astype(float)
            k, n = int(h.sum()), len(h)
            lo, hi = MT.wilson(k, n)
            _, ppt = MT.pesaran_timmermann(pv, act[s])
            res["models"].setdefault(mn, {})[tgt] = {
                "n": n, "acc": k / n, "ci_lo": lo, "ci_hi": hi, "p_binom": MT.binom_p(k, n), "p_pt": ppt,
                "base_rate_up": float(act[s].mean()), "frac_up_pred": float(pv.mean())}
            hits[mn] = dict(zip(np.array(days_t)[s].tolist(), h))
        res["pairwise"][tgt] = pairwise_daily(hits)
    # ---------------- P2
    hits2 = {}
    for mn in cand:
        num, den, dlist, blk = [], [], [], {b: [0, 0] for b in HOUR_BLOCKS}
        blk_daily = {b: ([], []) for b in HOUR_BLOCKS}
        for i in idx:
            pr = pred_dir(mn, "P2", i)
            if pr is None:
                continue
            y = lab2[i]
            s = np.isfinite(y) & (pr >= 0)
            if not s.any():
                continue
            h = (pr[s] == y[s]).astype(float)
            num.append(h.sum()), den.append(s.sum()), dlist.append(i)
            hh = np.full(24, np.nan)
            hh[s] = h
            for b, sl in HOUR_BLOCKS.items():
                v = hh[list(sl)]
                v = v[np.isfinite(v)]
                blk_daily[b][0].append(v.sum()), blk_daily[b][1].append(len(v))
        if not den:
            continue
        num, den = np.array(num), np.array(den)
        acc = num.sum() / den.sum()
        lo, hi = MT.boot_ratio_ci(num, den)
        rate = num / den
        _, st, p = MT.dm_hln(rate - 0.5, 1)
        nlo, nhi = MT.wilson(int(num.sum()), int(den.sum()))
        r = {"n_pairs": int(den.sum()), "n_days": int(len(den)), "acc": float(acc), "ci_lo": lo, "ci_hi": hi,
             "p_vs_half_dm": p, "naive_wilson_lo": nlo, "naive_wilson_hi": nhi, "blocks": {}}
        for b, (bn, bd) in blk_daily.items():
            bn, bd = np.array(bn, float), np.array(bd, float)
            if bd.sum():
                blo, bhi = MT.boot_ratio_ci(bn, bd)
                r["blocks"][b] = {"acc": float(bn.sum() / bd.sum()), "ci_lo": blo, "ci_hi": bhi, "n": int(bd.sum())}
        res["models"].setdefault(mn, {})["P2"] = r
        hits2[mn] = dict(zip(dlist, rate))
    res["pairwise"]["P2"] = pairwise_daily(hits2)
    # ---------------- P4
    qsd, aed, sqd, cov80, cov95 = {}, {}, {}, {}, {}
    for mn in preds:
        qsd[mn], aed[mn], sqd[mn], cov80[mn], cov95[mn] = {}, {}, {}, {}, {}
        for i, q in quant[mn].items():
            y = P[i]
            s = np.isfinite(y)
            if not s.any():
                continue
            Q = q[:, s].T                                               # [n_slots, 13]
            if np.isfinite(Q[:, I9]).all():
                qsd[mn][i] = float(MT.pinball(y[s], Q[:, I9], [QS[k] for k in I9]).mean())
                cov80[mn][i] = ((y[s] >= Q[:, QS.index(0.1)]) & (y[s] <= Q[:, QS.index(0.9)])).mean()
            if np.isfinite(Q[:, [0, -1]]).all():
                cov95[mn][i] = ((y[s] >= Q[:, 0]) & (y[s] <= Q[:, -1])).mean()
            if np.isfinite(Q[:, IMED]).all():
                aed[mn][i] = float(np.abs(y[s] - Q[:, IMED]).mean())
                sqd[mn][i] = float(((y[s] - Q[:, IMED]) ** 2).mean())
    ref_q, ref_a = qsd.get(REF_P4, {}), aed.get(REF_P4, {})
    for mn in preds:
        r = {}
        if qsd[mn]:
            ds = sorted(qsd[mn])
            r["n"] = len(ds)
            r["qs"] = float(np.mean([qsd[mn][i] for i in ds]))
            r["crps_approx"] = 2 * r["qs"]
            common = [i for i in ds if i in ref_q]
            if common:
                r["skill_vs_naive_epf"] = 1 - np.mean([qsd[mn][i] for i in common]) / np.mean([ref_q[i] for i in common])
            r["cov80"] = float(np.mean([cov80[mn][i] for i in ds]))
        if cov95[mn]:
            r["cov95"] = float(np.mean(list(cov95[mn].values())))
        if aed[mn]:
            ds = sorted(aed[mn])
            r["mae"] = float(np.mean([aed[mn][i] for i in ds]))
            r["rmse"] = float(np.sqrt(np.mean([sqd[mn][i] for i in ds])))
            common = [i for i in ds if i in ref_a]
            if common:
                r["rmae_vs_naive_epf"] = float(np.mean([aed[mn][i] for i in common]) / np.mean([ref_a[i] for i in common]))
        res["models"].setdefault(mn, {})["P4"] = r
    res["pairwise"]["P4_QS"] = pairwise_daily(qsd, loss=True)
    res["pairwise"]["P4_MAE"] = pairwise_daily(aed, loss=True)
    for mn in res["models"]:
        res["models"][mn]["kind"] = kind_of(mn)
        res["models"][mn]["contaminated"] = CONTAMINATED.get(mn)
        res["models"][mn]["subsample"] = SUBSAMPLED.get(mn)
    add_holm(res)
    return res


def pairwise_daily(series, loss=False):
    """series: model -> {day: value}; value = hit (P1/P3), daily hit rate (P2) or daily loss (P4).
    Returns 'a|b' -> stats on common days. diff = a - b in the metric's natural direction
    (accuracy: a - b > 0 means a better; loss: a - b < 0 means a better)."""
    out = {}
    names = list(series)
    for a in names:
        for b in names:
            if a == b:
                continue
            common = sorted(set(series[a]) & set(series[b]))
            if len(common) < 10:
                continue
            xa = np.array([series[a][i] for i in common], float)
            xb = np.array([series[b][i] for i in common], float)
            d = xa - xb
            md, st, p = MT.dm_hln(d, 1)
            lo, hi = MT.boot_mean_ci(d)
            r = {"n": len(common), "mean_diff": float(d.mean()), "p": p, "stat": st, "boot_lo": lo, "boot_hi": hi,
                 "a_better": bool(d.mean() < 0) if loss else bool(d.mean() > 0)}
            if not loss and np.isin(xa, (0.0, 1.0)).all() and np.isin(xb, (0.0, 1.0)).all():
                n01, n10, pm = MT.mcnemar_exact(xa == 0, xb == 0)
                r.update(mcnemar_p=pm, n_a_wrong_b_right=n01, n_a_right_b_wrong=n10)
            out[f"{a}|{b}"] = r
    return out


def add_holm(res):
    """Holm per target family over (claim model, competitor) pairs, claim models = candidates + alma; plus a
    descriptive family of FM/competitor vs baseline pairs."""
    names = list(res["models"])
    claim = [m for m in names if kind_of(m) in ("candidate", "alma")]
    for tgt, pw in res["pairwise"].items():
        for fam, pairs in (("claim", [(a, b) for a in claim for b in names if b != a and not CONTAMINATED.get(b)]),
                           ("descriptive", [(a, b) for a in names if kind_of(a) in ("fm", "competitor")
                                            for b in names if kind_of(b) == "baseline"])):
            keys = [f"{a}|{b}" for a, b in pairs if f"{a}|{b}" in pw]
            adj = MT.holm([pw[k]["p"] for k in keys])
            for k, pa in zip(keys, adj):
                pw[k][f"p_holm_{fam}"] = float(pa)
        # per claim model: Holm over its own competitors (the prereg section 7 claim rule)
        for a in claim:
            keys = [f"{a}|{b}" for b in names if b != a and not CONTAMINATED.get(b) and f"{a}|{b}" in pw]
            for k, pa in zip(keys, MT.holm([pw[k]["p"] for k in keys])):
                pw[k]["p_holm_own"] = float(pa)
                pw[k]["beats"] = bool(pw[k]["a_better"] and pa < 0.05 and
                                      (pw[k]["boot_lo"] > 0 or pw[k]["boot_hi"] < 0))


# --------------------------------------------------------------------------- strict mode
def preflight(models):
    import models as M
    if M.LOAD_ERRORS:
        raise SystemExit(f"preflight: architect module import errors: {M.LOAD_ERRORS}")
    fz = json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}
    metas = {}
    for mn in models:
        p = M.get(mn)
        meta = dict(getattr(p, "meta", {}))
        ck = meta.get("ckpt")
        if ck in ALMA_PINS and ALMA_PINS[ck] != meta.get("ckpt_sha256"):
            raise SystemExit(f"preflight: {ck} sha256 mismatch vs pin")
        if mn.startswith("alma") and ck not in ALMA_PINS:
            raise SystemExit(f"preflight: {mn} checkpoint {ck} not pinned")
        for path, sh in (fz.get("sha256") or {}).items():
            if ck == path and meta.get("ckpt_sha256") != sh:
                raise SystemExit(f"preflight: {mn} checkpoint {ck} sha256 != FROZEN.json")
        metas[mn] = meta
        del p
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "val", "test"])
    ap.add_argument("--models", default="")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rehearse", action="store_true", help="exact TEST procedure on VAL into runs/power/rehearsal_val/")
    a = ap.parse_args()
    strict = (a.split == "test" or a.rehearse) and not a.metrics_only
    if a.rehearse:
        if a.split != "val":
            raise SystemExit("--rehearse only with --split val")
        RUN_DIR["val"] = os.path.join(OUT, "rehearsal_val")
    run_rec, rp = None, None
    if strict:
        rdir = split_dir(a.split)
        if not os.path.exists(FROZEN) and not a.rehearse:
            raise SystemExit("strict mode refused: runs/power/FROZEN.json missing (candidate not frozen)")
        rp = os.path.join(rdir, "RUN.json")
        if os.path.exists(rp):
            raise SystemExit(f"strict mode refused: already run ({rp}); TEST runs exactly once")
        if os.path.isdir(rdir) and any(f.endswith(".npz") for _, _, fs in os.walk(rdir) for f in fs):
            raise SystemExit(f"strict mode refused: {rdir} already holds forecast files")
        if a.models or a.force:
            raise SystemExit("strict mode refused: uses the pre-registered model list (no --models/--force)")
        if DATA_SHA256 is None or sha_file(D.HOURLY) != DATA_SHA256:
            raise SystemExit("strict mode refused: data/power/hourly.csv sha256 != run_eval.DATA_SHA256 pin")
        fz = json.load(open(FROZEN)) if os.path.exists(FROZEN) else {"models": [], "note": "rehearsal before freeze"}
        models = PREREG_MODELS + [m for m in fz.get("models", []) if m not in PREREG_MODELS]
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "tools/power", "docs/power_prereg.md"],
                               capture_output=True, text=True).stdout.strip()
        if dirty and a.split == "test":
            raise SystemExit(f"strict mode refused: uncommitted changes:\n{dirty}")
        metas = preflight(models)
        run_rec = {"status": "started", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_head(),
                   "tools_power_dirty": dirty, "frozen": fz,
                   "frozen_sha256": sha_file(FROZEN) if os.path.exists(FROZEN) else None,
                   "data_sha256": DATA_SHA256, "alma_pins": ALMA_PINS, "models": models, "model_meta": metas,
                   "split": a.split, "rehearsal": a.rehearse, "argv": sys.argv}
        os.makedirs(rdir, exist_ok=True)
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    else:
        models = [m for m in a.models.split(",") if m] or PREREG_MODELS
        if a.split == "test" and not os.path.exists(os.path.join(split_dir("test"), "RUN.json")):
            raise SystemExit("--metrics-only on test needs a completed TEST run")
    try:
        if not a.metrics_only:
            forecast(a.split, models, force=a.force, strict=strict)
        res = score(a.split)
        res.update(commit=git_head(), scored=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rehearsal=a.rehearse,
                   data_sha256=sha_file(D.HOURLY))
        path = os.path.join(split_dir(a.split), "metrics.json")
        json.dump(res, open(path, "w"), indent=1, default=float)
        import report
        report.write(path)
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
