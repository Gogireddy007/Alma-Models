"""Architect B autoresearch loop: train a BNet config on the DEV-early inner folds and score P1 from the cache.

Inner folds (fit on the 16 training coins with label day <= fit_end; score all 24 coins on the next half-year):
  E1 fit <= 2022-12-31, test 2023-01-01..2023-06-30
  E2 fit <= 2023-06-30, test 2023-07-01..2023-12-31
  E3 fit <= 2023-12-31, test 2024-01-01..2024-06-30
S_B = mean P1_all over E1..E3 (row fold 'EM'); tie-break mean pinball. Nothing after 2024-06-30 is loaded.

  .venv/bin/python tools/fin/search_b/loop_b.py --name X --cfg '{"enc_d":"gru",...}' [--seed 0] [--folds E1,E2,E3]
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats_b as FB     # noqa: E402
import models_b as MB    # noqa: E402

FOLDS = {"E1": ("2022-12-31", "2023-01-01", "2023-06-30"),
         "E2": ("2023-06-30", "2023-07-01", "2023-12-31"),
         "E3": ("2023-12-31", "2024-01-01", "2024-06-30")}
TSV = os.path.join(FB.OUTB, "results_b.tsv")
COLS = ["time", "kind", "name", "config", "seed", "fold", "status", "P1_all", "P1_train", "P1_heldout", "P3_all",
        "E1_etf", "E3_etf", "SR_heldout", "SR_all", "SR_etf", "turn_heldout", "mdd_heldout", "note"]
_D = None


def cache():
    global _D
    if _D is None:
        _D = FB.build_cache()
        assert _D["days"].max() <= np.datetime64(FB.LOOP_END, "ns")
    return _D


def log_row(row):
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        row = dict(row, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        f.write("\t".join(_fmt(row.get(c, "")) for c in COLS) + "\n")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.5f}"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def tens(D, idx, cols=None, train=False):
    """train=True: features of the training coins built from the 16-coin context (cols index the training coins,
    which are the first 16 symbols; Xtr has only those 16 columns)."""
    cols = slice(None) if cols is None else cols
    g = lambda k: torch.from_numpy(np.ascontiguousarray(D[k][idx][:, cols]))
    X = torch.from_numpy(np.ascontiguousarray(D["Xtr"][idx][:, cols])) if train else g("X")
    return X, g("hz"), g("har"), g("aux")


def train_model(D, cfg, fit_end, seed, log=print):
    """Train one BNet on training coins, decision days with label day <= fit_end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    days = D["days"]
    fe = np.datetime64(pd.Timestamp(fit_end), "ns")
    tr_days = np.nonzero(days <= fe)[0]
    tr_days = tr_days[days[tr_days] >= np.datetime64("2020-06-01", "ns")]
    assert days[tr_days].max() <= fe
    has_next = set(tr_days[np.isin(tr_days + 1, tr_days)].tolist())
    cm = np.nonzero(D["train_mask"])[0]
    model = MB.BNet(cfg)
    steps, bs, lr = cfg.get("steps", 600), cfg.get("bs", 32), cfg.get("lr", 2e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    wp, wq, wb = cfg.get("wp", 1.0), cfg.get("wq", 1.0), cfg.get("wb", 0.3)
    t0 = time.time()
    model.train()
    for s in range(steps):
        di = rng.choice(tr_days, bs)
        if cfg.get("coin_drop", 0.25) > 0:                      # random subset of the training coins per batch
            keep = rng.random(len(cm)) > cfg.get("coin_drop", 0.25)
            keep[rng.integers(len(cm))] = True
            cols = cm[keep]
        else:
            cols = cm
        X, hz, har, aux = tens(D, di, cols, train=True)
        X, hz = X.clone(), hz.clone()
        etf = torch.from_numpy(rng.random(bs) < cfg.get("p_etf", 0.3))
        X[etf, :, :, 7:15] = 0.0
        hz[etf] = 0.0
        ok = torch.from_numpy(D["ok"][di][:, cols])
        tl = torch.from_numpy(D["tl"][di][:, cols])
        y = torch.from_numpy(D["y"][di][:, cols])
        mask = ok & torch.isfinite(tl)
        out = model(X, hz, har, aux, ok, gen=gen)
        lq = MB.pinball(out["q"], torch.nan_to_num(tl))[mask].mean()
        my = mask & torch.isfinite(y)
        if model.stoch:
            la = out["logits_all"]
            lp = Fn_bce(la[:, my], y[my].expand(la.shape[0], -1)) if my.any() else 0.0
        else:
            lp = Fn_bce(out["logit"][my], y[my]) if my.any() else 0.0
        loss = wq * lq + wp * lp
        if model.wm > 0 or model.stoch:
            blk = torch.from_numpy(D["blk"][di][:, cols])
            mb = torch.from_numpy(D["mblk"][di][:, cols]) > 0
            mb = mb & mask
            if mb.any():
                loss = loss + wb * (model.blk(out["next"])[mb] - torch.nan_to_num(blk[mb])).abs().mean()
            nx = np.array([d in has_next for d in di])
            if model.wm > 0 and nx.any():
                X2, hz2, har2, aux2 = tens(D, di[nx] + 1, cols, train=True)
                ok2 = torch.from_numpy(D["ok"][di[nx] + 1][:, cols])
                with torch.no_grad():
                    s2, _ = model.encode(X2, hz2, har2, aux2, ok2)
                m2 = mask[torch.from_numpy(nx)] & ok2
                if m2.any():
                    loss = loss + model.wm * ((out["next"][torch.from_numpy(nx)] - s2) ** 2)[m2].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 200 == 0 or s == steps - 1:
            log(f"  step {s} loss {float(loss):.4f} q {float(lq):.4f} {time.time() - t0:.0f}s")
    model.eval()
    return model


def Fn_bce(lg, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(lg, y)


def predict(models, D, idx, train=False):
    """-> p [n, m], q [n, m, 13] (log RV units) for cache day indices idx (all 24 coins). train=True: only the
    training coins, from their 16-coin-context features Xtr (held-out columns NaN) -- for in-sample policy inputs."""
    if train:
        tc = np.nonzero(D["train_mask"])[0]
        p16, q16 = _predict(models, D, idx, tc, True)
        p = np.full((len(idx), len(D["symbols"])), np.nan); q = np.full(p.shape + (13,), np.nan)
        p[:, tc], q[:, tc] = p16, q16
        return p, q
    return _predict(models, D, idx, None, False)


def _predict(models, D, idx, cols, train):
    P, Q = [], []
    with torch.no_grad():
        for k in range(0, len(idx), 64):
            ii = idx[k:k + 64]
            X, hz, har, aux = tens(D, ii, cols, train)
            sel = slice(None) if cols is None else cols
            ok = torch.from_numpy(D["ok"][ii][:, sel])
            ps, qs = [], []
            for mdl in models:
                g = torch.Generator().manual_seed(1234)
                o = mdl(X, hz, har, aux, ok, gen=g)
                ps.append(torch.sigmoid(o["logit"]).numpy()); qs.append(o["q"].numpy())
            p = np.mean(ps, 0)
            q = np.sort(np.mean(qs, 0), -1) + D["A"][ii][:, sel][..., None]
            p[~D["ok"][ii][:, sel]] = np.nan
            q[~D["ok"][ii][:, sel]] = np.nan
            P.append(p); Q.append(q)
    return np.concatenate(P), np.concatenate(Q)


I9 = [QS for QS in range(2, 11)]


def score(D, idx, p, q):
    y, lrv = D["y"][idx], D["lrv"][idx]
    tm = D["train_mask"]
    d = np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan)
    ok = np.isfinite(d) & np.isfinite(y)
    hit = (d == y) & ok
    res = {}
    for g, cols in (("all", np.ones_like(tm)), ("train", tm), ("heldout", ~tm)):
        res[g] = float(hit[:, cols].sum() / max(1, ok[:, cols].sum()))
    t = np.array([MB.QS[i] for i in I9])
    e = lrv[..., None] - q[..., I9]
    pb = np.maximum(t * e, (t - 1) * e).mean(-1)
    res["qs"] = float(np.nanmean(np.where(np.isfinite(lrv), pb, np.nan)))
    res["day_acc"] = np.where(ok.sum(1) > 0, hit.sum(1) / np.maximum(ok.sum(1), 1), np.nan)
    return res


def fold_idx(D, a, b):
    days = D["days"]
    return np.nonzero((days >= np.datetime64(pd.Timestamp(a), "ns")) & (days <= np.datetime64(pd.Timestamp(b), "ns")))[0]


def run_cfg(name, cfg, seed=0, folds=("E1", "E2", "E3"), kind="wm_b", log=print, save_preds=True):
    D = cache()
    rows, accs = [], []
    try:
        t0 = time.time()
        allres = {}
        for fk in folds:
            fe, a, b = FOLDS[fk]
            idx = fold_idx(D, a, b)
            members = [train_model(D, cfg, fe, seed + 1000 * k, log) for k in range(cfg.get("ens", 1))]
            p, q = predict(members, D, idx)
            r = score(D, idx, p, q)
            allres[fk] = r
            if save_preds:
                os.makedirs(os.path.join(FB.OUTB, "preds"), exist_ok=True)
                np.savez(os.path.join(FB.OUTB, "preds", f"{name}_s{seed}_{fk}.npz"), p=p, q=q, idx=idx)
            log_row({"kind": kind, "name": name, "config": cfg, "seed": seed, "fold": fk, "status": "ok",
                     "P1_all": r["all"], "P1_train": r["train"], "P1_heldout": r["heldout"], "P3_all": r["qs"],
                     "note": f"params {MB.nparams(members[0])}"})
        m = {k: float(np.mean([allres[f][k] for f in folds])) for k in ("all", "train", "heldout", "qs")}
        log_row({"kind": kind, "name": name, "config": cfg, "seed": seed, "fold": "EM", "status": "ok",
                 "P1_all": m["all"], "P1_train": m["train"], "P1_heldout": m["heldout"], "P3_all": m["qs"],
                 "note": f"{time.time() - t0:.0f}s " + " ".join(f"{f}={allres[f]['all']:.4f}" for f in folds)})
        log(f"{name} s{seed}: EM P1 {m['all']:.4f} (train {m['train']:.4f} ho {m['heldout']:.4f}) QS {m['qs']:.4f} "
            + " ".join(f"{f}={allres[f]['all']:.4f}" for f in folds) + f" {time.time() - t0:.0f}s")
        return m
    except Exception as ex:
        traceback.print_exc()
        log_row({"kind": kind, "name": name, "config": cfg, "seed": seed, "fold": "EM", "status": "crash",
                 "note": f"{type(ex).__name__}: {str(ex)[:200]}"})
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="E1,E2,E3")
    a = ap.parse_args()
    torch.set_num_threads(3)
    run_cfg(a.name, json.loads(a.cfg), a.seed, a.folds.split(","))


if __name__ == "__main__":
    main()
