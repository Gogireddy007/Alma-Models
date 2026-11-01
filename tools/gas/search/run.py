"""Alma-Gas architect search runner (docs/gas_search.md section 0 = binding protocol).

    nice -n 10 .venv/bin/python tools/gas/search/run.py cache                       # contexts of train weeks
    nice -n 10 .venv/bin/python tools/gas/search/run.py bt --cfg '{"arch":"wm",...}' --seed 0 --tag r1_x
    nice -n 10 .venv/bin/python tools/gas/search/run.py refs                        # reference rows

DEV backtest = rolling origin, 6 yearly folds by RELEASE date (Jul 1 Y .. Jun 30 Y+1, Y = 2019..2024; the harness
DEV window). Each fold trains ONLY on storage.train_weeks() released strictly before the fold start (asserted).
Every run (crashes included) is appended to runs/gas/search/results.tsv; predictions to runs/gas/search/preds/.
Only weeks from storage.train_weeks() (released <= 2025-06-30) are ever loaded, built or scored here.
"""
import os
import sys
import json
import time
import pickle
import argparse
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GAS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(GAS))
sys.path.insert(0, GAS)
sys.path.insert(0, HERE)
import storage as S          # noqa: E402
import data as D             # noqa: E402
import feats as FT           # noqa: E402

OUT = os.path.join(ROOT, "runs", "gas", "search")
CACHE = os.path.join(OUT, "cache", "ctx_train.pkl")
TSV = os.path.join(OUT, "results.tsv")
QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
FOLD_YEARS = list(range(2019, 2025))
COLS = ["time", "run", "cfg", "seed", "status", "n_params", "mae_thu", "mae_fri", "qs_thu", "qs_fri", "acc_thu",
        "acc_fri", "accreg_thu", "accreg_fri", "cov80_thu", "cov80_fri", "S", "seconds", "note"]


# =========================================================================== data (train weeks only)
def build_cache():
    w, o, f = S.load()
    ks = S.train_weeks()
    assert (w.avail.iloc[ks] <= D.TRAIN_END).all()
    items = []
    t0 = time.time()
    for k in ks:
        for kind in S.ORIGINS:
            c = S.build_context(int(k), kind, w, o, f)
            items.append(c)
    lab = {int(k): {"avail": w.avail.iloc[k], "ds": np.array([w[f"ds_{s}"].iloc[k] for s in S.SERIES]),
                    "dev": np.array([w[f"dev_{s}"].iloc[k] for s in S.SERIES])} for k in ks}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump({"items": items, "lab": lab, "train_end": str(D.TRAIN_END.date())}, open(CACHE, "wb"))
    print(f"cached {len(items)} contexts ({len(ks)} weeks, released {lab[int(ks[0])]['avail'].date()}.."
          f"{lab[int(ks[-1])]['avail'].date()}) in {time.time() - t0:.0f}s")


_CC = {}


def load_cache():
    if "c" not in _CC:
        z = pickle.load(open(CACHE, "rb"))
        assert max(v["avail"] for v in z["lab"].values()) <= D.TRAIN_END
        _CC["c"] = z
    return _CC["c"]


def folds():
    """[(fold_start, train item idx, test item idx)] over the cached items."""
    z = load_cache()
    av = np.array([z["lab"][c["k"]]["avail"] for c in z["items"]])
    dev_ks = set(int(k) for k in S.target_weeks("dev"))
    out = []
    for y in FOLD_YEARS:
        lo, hi = pd.Timestamp(f"{y}-07-01"), min(pd.Timestamp(f"{y + 1}-07-01"), D.TRAIN_END + pd.Timedelta(days=1))
        tr = np.where(av < lo)[0]
        te = np.array([i for i in np.where((av >= lo) & (av < hi))[0] if z["items"][i]["k"] in dev_ks])
        assert av[tr].max() < lo <= av[te].min()
        out.append((lo, tr, te))
    return out


def scales_of(idx):
    z = load_cache()
    dev = np.array([z["lab"][z["items"][i]["k"]]["dev"] for i in idx])
    lv = np.array([[z["items"][i]["lvl_hist"][s][-1] for s in S.SERIES] for i in idx])
    return {"dev_sd": np.nanstd(dev, 0).tolist(), "lvl_sd": np.nanstd(lv, 0).tolist()}


def fc_err_sd(idx_before, cutoff):
    """Std of CPC forecast error (forecast - later observed) by lead 1..7, HDD and CDD, from the cached training
    contexts; only target days < cutoff (their observations were public before the fold start)."""
    z = load_cache()
    obs = {}
    for c in z["items"]:
        for d, v in zip(c["obs_daily_dates"][-30:], c["obs_daily"][-30:]):
            obs[str(d)] = v
    err = {L: [] for L in range(1, 8)}
    for i in idx_before:
        c = z["items"][i]
        for j, L in enumerate(c["wx_src"]):
            dd = c["wx_days"][j]
            if L >= 1 and pd.Timestamp(dd) < cutoff and dd in obs:
                err[int(L)].append(c["wx"][j] - obs[dd])
    sd = np.ones((8, 2))
    for L in range(1, 8):
        if len(err[L]) >= 20:
            sd[L] = np.std(np.array(err[L]), 0)
    return sd.tolist()


# =========================================================================== metrics
def metrics(q, lab):
    """q [n,6,13], lab dict list. -> dict of L48 MAE/QS/acc/cov80 + pooled regional acc."""
    ds = np.array([l["ds"] for l in lab])
    dv = np.array([l["dev"] for l in lab])
    a5 = ds - dv
    med = q[:, :, QS.index(0.5)]
    out = {}
    e = ds[:, 0] - med[:, 0]
    out["mae"] = float(np.mean(np.abs(e)))
    ql = np.array([QS[i] for i in I9])
    d = ds[:, 0][:, None] - q[:, 0, I9]
    out["qs"] = float(np.mean(np.maximum(ql * d, (ql - 1) * d)))
    m = dv[:, 0] != 0
    out["acc"] = float(np.mean(((med[m, 0] - a5[m, 0]) > 0) == (dv[m, 0] > 0)))
    out["cov80"] = float(np.mean((ds[:, 0] >= q[:, 0, QS.index(0.1)]) & (ds[:, 0] <= q[:, 0, QS.index(0.9)])))
    mr = dv[:, 1:] != 0
    out["accreg"] = float(np.mean((((med[:, 1:] - a5[:, 1:]) > 0) == (dv[:, 1:] > 0))[mr]))
    return out


def score_row(pred, ref=None):
    """pred: {kind: (q, lab)} -> row dict; S vs reference metrics {kind: metrics}."""
    r = {}
    for kind in S.ORIGINS:
        m = metrics(*pred[kind])
        for k2 in ("mae", "qs", "acc", "accreg", "cov80"):
            r[f"{k2}_{kind}"] = round(m[k2], 4)
    if ref is not None:
        r["S"] = round(float(np.mean([(r[f"mae_{k}"] / ref[f"mae_{k}"] + r[f"qs_{k}"] / ref[f"qs_{k}"]) / 2
                                      for k in S.ORIGINS])), 4)
    return r


def log_row(row):
    new = not os.path.exists(TSV)
    with open(TSV, "a") as fh:
        if new:
            fh.write("\t".join(COLS) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in COLS) + "\n")


def ref_metrics():
    t = pd.read_csv(TSV, sep="\t")
    r = t[(t.run == "ref_ols_wx") & (t.status == "ok")].iloc[-1]
    return {f"{m}_{k}": float(r[f"{m}_{k}"]) for m in ("mae", "qs") for k in S.ORIGINS}


# =========================================================================== torch model training
def tensors(fs, idx=None):
    import torch
    st = FT.stack(fs if idx is None else [fs[i] for i in idx])
    return {k: torch.tensor(v) for k, v in st.items()}


DEFAULT = {"arch": "wm", "d": 32, "assim": 1, "regime": 0, "steps": 1200, "lr": 3e-3, "bs": 64, "wd": 1e-4,
           "aux": 0.3, "wl48": 3.0, "mc": 0, "drop_hh": 0, "drop": 0.0, "cal": 0}


def train_model(cfg, items, labs, scales, seed):
    import torch
    import nets as N
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 2)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fs = [FT.feats(c, scales) for c in items]
    b = tensors(fs)
    if cfg.get("drop_hh"):
        b["g"][:, -3:-1] = 0.0
    y = torch.tensor(np.array([l["ds"] for l in labs]), dtype=torch.float32)
    sd = torch.tensor(scales["dev_sd"], dtype=torch.float32)
    m = N.GasNet(b["g"].shape[1], b["u"].shape[2], d=cfg["d"], rollout=(cfg["arch"] == "wm"),
                 assim=bool(cfg["assim"]), regime=bool(cfg["regime"]), drop=cfg.get("drop", 0.0))
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"])
    w = torch.tensor([cfg["wl48"], 1, 1, 1, 1, 1], dtype=torch.float32)
    n = len(y)
    g = torch.Generator().manual_seed(seed)
    m.train()
    for step in range(cfg["steps"]):
        i = torch.randint(0, n, (cfg["bs"],), generator=g)
        bb = {k: v[i] for k, v in b.items()}
        out = m(bb, sd)
        loss = N.pinball(out["q"], y[i], sd, w)
        if cfg["assim"] and cfg["aux"] > 0:
            loss = loss + cfg["aux"] * ((bb["dsprev"][:, 1:] - out["prev"]).abs() / sd[1:]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    m.eval()
    return m, N.n_params(m)


def fit(cfg, items, labs, scales, seed):
    """Train; if cfg['cal']: quantiles = median + empirical quantiles (per series, per origin kind) of the median's
    residuals on a time-ordered holdout = the last 20% of the training weeks, scored by a model trained on the
    first 80% (same seed/steps); the returned model is then retrained on 100% of the training weeks."""
    m, npar = train_model(cfg, items, labs, scales, seed)
    cal = None
    if cfg.get("cal"):
        ks = sorted(set(c["k"] for c in items))
        cut = ks[int(0.8 * len(ks))]
        a = [i for i, c in enumerate(items) if c["k"] < cut]
        h = [i for i, c in enumerate(items) if c["k"] >= cut]
        m80, _ = train_model(cfg, [items[i] for i in a], [labs[i] for i in a], scales, seed)
        qh = predict(m80, {**cfg, "mc": 0}, [items[i] for i in h], scales)
        res = np.array([labs[i]["ds"] for i in h]) - qh[:, :, QS.index(0.5)]
        cal = {}
        for kind in S.ORIGINS:
            sel = np.array([items[i]["kind"] == kind for i in h])
            cal[kind] = np.nanquantile(res[sel], QS, axis=0).T.tolist()       # [6, 13]
    return m, npar, cal


def apply_cal(q, items, cal):
    if cal is None:
        return q
    med = q[:, :, QS.index(0.5)]
    return np.stack([med[i][:, None] + np.array(cal[c["kind"]]) for i, c in enumerate(items)])


def predict(m, cfg, items, scales, fcsd=None, npaths=200, seed=0):
    import torch
    fs = [FT.feats(c, scales) for c in items]
    b = tensors(fs)
    if cfg.get("drop_hh"):
        b["g"][:, -3:-1] = 0.0
    sd = torch.tensor(scales["dev_sd"], dtype=torch.float32)
    with torch.no_grad():
        if not cfg.get("mc"):
            return m(b, sd)["q"].numpy().astype(np.float64)
        return mc_quantiles(m, b, sd, fcsd, npaths, seed)


def mc_quantiles(m, b, sd, fcsd, npaths, seed, shift=None):
    """Monte Carlo: npaths weather paths (forecast days perturbed with N(0, sd_lead) CPC error, held in common
    across regions) rolled through the model; each path's quantile function is sampled once per path draw
    (inverse-CDF interpolation) and the pooled sample gives the quantiles. shift [B,14,4] = scenario offset."""
    import torch
    rng = np.random.default_rng(seed)
    U = b["u"]
    lead = np.rint(U[..., 7].numpy() * 7).astype(int)          # 0 = observed
    fsd = np.asarray(fcsd)[lead]                                # [B,14,2]
    fsd[lead == 0] = 0.0
    B = U.shape[0]
    samples = []
    ql = np.array(QS)
    for p in range(npaths):
        eps = rng.standard_normal((B, 14, 2)) * fsd
        sh = np.zeros((B, 14, 4), np.float32)
        sh[..., :2] = eps
        if shift is not None:
            sh = sh + shift
        sh[..., 0] = np.maximum(sh[..., 0], -b["phys"][..., 0].numpy())   # degree days stay >= 0
        sh[..., 1] = np.maximum(sh[..., 1], -b["phys"][..., 1].numpy())
        q = m(b, sd, phys_shift=torch.tensor(sh))["q"].numpy()           # [B,6,13]
        uu = rng.uniform(0.025, 0.975, (B, 1))
        samples.append(np.stack([[np.interp(uu[i, 0], ql, q[i, s]) for s in range(6)] for i in range(B)]))
    smp = np.stack(samples, -1)                                  # [B,6,P]
    return np.quantile(smp, QS, axis=-1).transpose(1, 2, 0)


def run_bt(cfg, seed, tag, save_ckpt=False):
    z = load_cache()
    t0 = time.time()
    preds = {k: [] for k in S.ORIGINS}
    labs = {k: [] for k in S.ORIGINS}
    weeks = {k: [] for k in S.ORIGINS}
    npar = None
    for lo, tr, te in folds():
        items = [z["items"][i] for i in tr]
        assert max(z["lab"][c["k"]]["avail"] for c in items) < lo
        ltr = [z["lab"][c["k"]] for c in items]
        sc = scales_of(tr)
        m, npar, cal = fit(cfg, items, ltr, sc, seed)
        fcsd = fc_err_sd(tr, lo) if cfg.get("mc") else None
        tit = [z["items"][i] for i in te]
        q = apply_cal(predict(m, cfg, tit, sc, fcsd, seed=seed), tit, cal)
        for c, qq in zip(tit, q):
            preds[c["kind"]].append(qq)
            labs[c["kind"]].append(z["lab"][c["k"]])
            weeks[c["kind"]].append(c["k"])
    pred = {k: (np.array(preds[k]), labs[k]) for k in S.ORIGINS}
    np.savez(os.path.join(OUT, "preds", f"{tag}.npz"), **{f"q_{k}": pred[k][0] for k in S.ORIGINS},
             **{f"k_{k}": np.array(weeks[k]) for k in S.ORIGINS})
    try:
        ref = ref_metrics()
    except Exception:
        ref = None
    row = score_row(pred, ref)
    row.update(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run=tag, cfg=json.dumps(cfg, sort_keys=True), seed=seed,
               status="ok", n_params=npar, seconds=round(time.time() - t0, 1))
    return row


# =========================================================================== reference models (same folds)
def ref_predict(name, tr, te):
    """Linear references refit per fold on the fold's training weeks (out-of-sample, unlike the harness's
    st_ols_wx which is fitted once on all DEV weeks), plus the harness's unfitted baselines."""
    import models as M
    z = load_cache()
    it_tr, it_te = [z["items"][i] for i in tr], [z["items"][i] for i in te]
    if name in ("ref_persistence", "ref_climatology"):
        mm = M.st_persistence() if name == "ref_persistence" else M.st_climatology()
        return np.stack([mm.storage(it_te, QS, s) for s in S.SERIES], 1)
    sc = scales_of(tr)

    def X(c, s):
        an = M.wx_features(c)
        x = [1.0, an[0], an[1], c["dev_hist"][s][-1]]
        if name in ("ref_ols_seas", "ref_ridge_daily"):
            tw = pd.Timestamp(c["target_week"])
            a = 2 * np.pi * tw.dayofyear / 365.25
            x += [an[0] * np.cos(a), an[1] * np.cos(a), an[0] * np.sin(a), an[1] * np.sin(a), np.cos(a), np.sin(a)]
        if name == "ref_ridge_daily":
            f = FT.feats(c, sc)
            x += list(f["u"][7:, 4]) + list(f["u"][7:, 5]) + list(f["u"][:7, 4].sum(keepdims=True)) + list(f["g"])
        return np.nan_to_num(np.array(x, float))
    out = np.zeros((len(te), 6, len(QS)))
    for kind in S.ORIGINS:
        itr = [i for i, c in enumerate(it_tr) if c["kind"] == kind]
        ite = [i for i, c in enumerate(it_te) if c["kind"] == kind]
        for j, s in enumerate(S.SERIES):
            Xtr = np.array([X(it_tr[i], s) for i in itr])
            ytr = np.array([z["lab"][it_tr[i]["k"]]["dev"][j] for i in itr])
            ok = np.isfinite(ytr)
            Xtr, ytr = Xtr[ok], ytr[ok]
            lam = 1.0 * len(ytr) if name == "ref_ridge_daily" else 0.0
            mu, sg = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0) + 1e-9
            Z = np.c_[np.ones(len(Xtr)), (Xtr[:, 1:] - mu) / sg]
            P = np.eye(Z.shape[1]) * lam
            P[0, 0] = 0
            beta = np.linalg.solve(Z.T @ Z + P * 1e-2 if lam else Z.T @ Z, Z.T @ ytr)
            res = ytr - Z @ beta
            for i in ite:
                c = it_te[i]
                x = X(c, s)
                zz = np.r_[1.0, (x[1:] - mu) / sg]
                out[i, j] = c["avg5"][s] + zz @ beta + np.quantile(res, QS)
    return out


def run_refs(names):
    z = load_cache()
    for name in names:
        t0 = time.time()
        try:
            preds = {k: [] for k in S.ORIGINS}
            labs = {k: [] for k in S.ORIGINS}
            weeks = {k: [] for k in S.ORIGINS}
            for lo, tr, te in folds():
                q = ref_predict(name, tr, te)
                for i, qq in zip(te, q):
                    c = z["items"][i]
                    preds[c["kind"]].append(qq), labs[c["kind"]].append(z["lab"][c["k"]]), weeks[c["kind"]].append(c["k"])
            pred = {k: (np.array(preds[k]), labs[k]) for k in S.ORIGINS}
            np.savez(os.path.join(OUT, "preds", f"{name}.npz"), **{f"q_{k}": pred[k][0] for k in S.ORIGINS},
                     **{f"k_{k}": np.array(weeks[k]) for k in S.ORIGINS})
            try:
                ref = ref_metrics() if name != "ref_ols_wx" else None
            except Exception:
                ref = None
            row = score_row(pred, ref)
            if name == "ref_ols_wx":
                row["S"] = 1.0
            row.update(status="ok", note="reference")
        except Exception as e:
            row = {"status": "crash", "note": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        row.update(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run=name, cfg="{}", seed="", seconds=round(time.time() - t0, 1))
        log_row(row)
        print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "bt", "refs"])
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache()
    elif a.cmd == "refs":
        run_refs(["ref_ols_wx", "ref_ols_seas", "ref_ridge_daily", "ref_persistence", "ref_climatology"])
    else:
        cfg = {**DEFAULT, **json.loads(a.cfg)}
        t0 = time.time()
        try:
            row = run_bt(cfg, a.seed, a.tag)
        except Exception as e:
            traceback.print_exc()
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.tag, "cfg": json.dumps(cfg, sort_keys=True),
                   "seed": a.seed, "status": "crash", "note": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        row["note"] = (row.get("note", "") + " " + a.note).strip()
        log_row(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
