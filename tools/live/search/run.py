"""
Alma-Live Gas search runner (docs/live_search.md 0.5-0.8). Every call appends to runs/live/search/results.tsv.

    nice -n 10 .venv/bin/python tools/live/search/run.py refs                       # ridge-QR + LightGBM refs
    nice -n 10 .venv/bin/python tools/live/search/run.py wm '{"arch":"struct"}' "note"
"""
import os
import sys
import json
import time
import uuid
import hashlib
import datetime as dt

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(LIVE))
sys.path.insert(0, LIVE)
torch.set_num_threads(2)

SRC = os.environ.get("LIVE_SRC", "wm_store")
WD = __import__(SRC)
import wm_core as C  # noqa: E402
import wm_harness as H  # noqa: E402

OUT = os.environ.get("LIVE_OUT", os.path.join(ROOT, "runs", "live", "search"))
PRED = os.path.join(OUT, "preds")
CK = os.path.join(OUT, "ckpt")
TSV = os.path.join(OUT, "results.tsv")
COLS = ["run_id", "utc", "kind", "family", "seed", "S_F", "rel_r1", "rel_r5", "rel_lrv5", "rel_dsurp",
        "S_A", "params", "steps", "locked", "leak", "status", "secs", "cfg", "note"]


LOCKED_FILES = [os.path.join(LIVE, "wm_harness.py"), os.path.join(LIVE, "wm_store.py"), os.path.abspath(__file__)]
LOCKED_PIN = os.path.join(OUT, "LOCKED.sha")


def locked_sha():
    h = hashlib.sha256()
    for p in LOCKED_FILES:
        h.update(open(p, "rb").read())
    h.update(open(WD.PANEL, "rb").read())
    return h.hexdigest()[:16]


def assert_locked():
    """LRT-17: the locked harness (+ frozen panel) must match the pin written at loop start."""
    cur = locked_sha()
    if os.path.exists(LOCKED_PIN):
        pin = open(LOCKED_PIN).read().strip()
        assert cur == pin, f"locked harness changed: {cur} != pin {pin}"
    return cur


def git_head():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def log_row(**kw):
    os.makedirs(OUT, exist_ok=True)
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(kw.get(c, "")) for c in COLS) + "\n")


def folds(P, L):
    return {Y: H.fold_rows(P, L, "dev", Y) for Y in WD.FOLD_YEARS}


# ------------------------------------------------------------------ flat features for the matched refs
def flat(P, rows, L=20):
    X = P["X"]
    a = X[rows]
    m5 = np.stack([X[i - 4:i + 1].mean(0) for i in rows])
    m20 = np.stack([X[i - L + 1:i + 1].mean(0) for i in rows])
    return np.concatenate([a, m5, m20, P["U"][rows].reshape(len(rows), -1)], 1)


def fit_linqr(F, Y, lam=1e-2, steps=600, seed=0):
    torch.manual_seed(seed)
    Ft, Yt = torch.tensor(F, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    W = torch.zeros(F.shape[1], Y.shape[1], C.NQ, requires_grad=True)
    b = torch.zeros(Y.shape[1], C.NQ, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.02)
    for _ in range(steps):
        q = torch.einsum("nf,ftq->ntq", Ft, W) + b
        loss, _ = C.pinball(q, Yt)
        loss = loss + lam * (W ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda G: (torch.einsum("nf,ftq->ntq", torch.tensor(G, dtype=torch.float32), W) + b).detach().numpy()


def refs(which=("linqr", "lgbm")):
    P = WD.load_panel()
    L = 20
    out = {}
    for Y, (tr, te) in folds(P, L).items():
        norm = H.Norm(P["X"], P["Y"], tr)
        Ftr, Fte = flat(P, tr, L), flat(P, te, L)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
        Ytr = norm.y(P["Y"][tr])
        res = {"te": te, "y": norm.y(P["Y"][te])}
        if "linqr" in which:
            res["linqr"] = np.sort(fit_linqr(Ftr, Ytr)(Fte), -1)
        if "lgbm" in which:
            import lightgbm as lgb
            q = np.zeros((len(te), C.NT, C.NQ))
            for j in range(C.NT):
                m = np.isfinite(Ytr[:, j])
                for k, a in enumerate(H.QL):
                    g = lgb.LGBMRegressor(objective="quantile", alpha=float(a), n_estimators=200, learning_rate=0.03,
                                          num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, verbose=-1, n_jobs=1, random_state=0)
                    g.fit(Ftr[m], Ytr[m, j])
                    q[:, j, k] = g.predict(Fte)
            res["lgbm"] = np.sort(q, -1)
        out[Y] = res
    os.makedirs(PRED, exist_ok=True)
    np.savez(os.path.join(PRED, "refs.npz"), **{f"{Y}_{k}": v for Y, r in out.items() for k, v in r.items()})
    rows = {}
    for name in which:
        rel = score({Y: out[Y][name] for Y in out}, out)
        rows[name] = rel
        log_row(run_id=f"ref_{name}", utc=dt.datetime.utcnow().isoformat(timespec="seconds"), kind="ref",
                family=name, seed=0, **rel, status="ok", note="feature-matched reference (not in K)")
    print(json.dumps(rows, indent=1))
    return out


def load_refs():
    z = np.load(os.path.join(PRED, "refs.npz"))
    out = {}
    for k in z.files:
        Y, name = k.split("_", 1)
        out.setdefault(int(Y), {})[name] = z[k]
    return out


def score(Q, R, ref="linqr"):
    """Q {fold: q [n, T, NQ]} standardized; R refs dict. Returns S_F and per-target relative QS."""
    rels = []
    for Y in sorted(Q):
        y = R[Y]["y"]
        a = np.nanmean(H.pinball_np(Q[Y], y), 0)
        b = np.nanmean(H.pinball_np(R[Y][ref], y), 0)
        rels.append(a / b)
    rels = np.mean(rels, 0)
    d = {f"rel_{t}": round(float(v), 4) for t, v in zip(WD.TARGETS, rels)}
    prim = [WD.TARGETS.index(t) for t in getattr(WD, "PRIMARY", WD.TARGETS)]
    d["S_F"] = round(float(rels[prim].mean()), 4)
    return d


# ------------------------------------------------------------------ per-checkpoint randomized-future test
def leak_check(m, P, R, te, norm, L, n=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(te, min(n, len(te)), replace=False)
    for r in rows:
        b = H.predict(m, P, np.array([r]), norm, L)[0]
        t = P["D"][r]
        j = WD.row_index(R, t)
        Q = WD.perturb_future(R, t, rng)
        w = WD.window_at(j, Q, L, "pert")
        if w is None:
            return False
        Xw = torch.tensor(norm.x(w[0]))[None]
        U = torch.tensor(w[1])[None]
        ST = torch.tensor(P["ST"][r])[None]
        with torch.no_grad():
            o = m(Xw, U, ST, sample=False).numpy()[0]
        if not np.array_equal(o, b):
            # panel window and rebuilt window must agree exactly
            print("LEAKFAIL", t, float(np.abs(o - b).max()))
            return False
    return True


def run_wm(cfg, note="", seeds=(0,), kind="wm", save=True, P=None, R=None, refs_=None):
    t0 = time.time()
    lsha = assert_locked()
    P = P or WD.load_panel()
    R = R or WD.load_raw()
    refs_ = refs_ or load_refs()
    L = cfg.get("L", 20)
    rid = cfg.get("run_id") or f"{kind}_{dt.datetime.utcnow().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    family = cfg.get("arch") + ("" if cfg.get("graph", 1) else "_nog")
    Q, leak_ok, npar, nsteps = {}, True, 0, []
    status = "ok"
    try:
        for Y, (tr, te) in folds(P, L).items():
            assert np.array_equal(te, refs_[Y]["te"])
            norm = H.Norm(P["X"], P["Y"], tr)
            qs = []
            cfg_f = cfg
            if "fold_steps" in cfg:      # LRT-16: twin matched on optimizer steps per fold
                cfg_f = dict(cfg, budget_s=None, steps=int(cfg["fold_steps"][str(Y)]))
            for s in seeds:
                m = C.train_model(cfg_f, P, tr, s, norm)
                npar = C.nparams(m)
                nsteps.append(getattr(m, "n_steps", 0))
                if not leak_check(m, P, R, te, norm, L, seed=s):
                    leak_ok = False
                    raise RuntimeError("leakfail")
                qs.append(H.predict(m, P, te, norm, L))
                if save:
                    os.makedirs(CK, exist_ok=True)
                    torch.save({"cfg": cfg, "state": m.state_dict(), "norm": norm.state()},
                               os.path.join(CK, f"{rid}_f{Y}_s{s}.pt"))
            Q[Y] = np.sort(np.mean(qs, 0), -1)
        sc = score(Q, refs_)
        np.savez(os.path.join(PRED, f"{rid}.npz"), **{str(Y): v for Y, v in Q.items()})
    except Exception as e:  # noqa: BLE001
        status = "leakfail" if not leak_ok else f"crash:{type(e).__name__}:{str(e)[:80]}"
        sc = {}
    log_row(run_id=rid, utc=dt.datetime.utcnow().isoformat(timespec="seconds"), kind=kind, family=family,
            seed="+".join(map(str, seeds)), **sc, params=npar, steps="/".join(map(str, nsteps)), locked=lsha, leak="pass" if leak_ok else "FAIL", status=status,
            secs=int(time.time() - t0), cfg=json.dumps(cfg, sort_keys=True), note=note)
    print(rid, status, sc, f"{int(time.time() - t0)}s", flush=True)
    return rid, sc, Q


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refs":
        refs()
    elif cmd == "exp":        # autoresearch: score wm_core.DEFAULT_CFG (the editable recipe)
        run_wm(dict(C.DEFAULT_CFG), sys.argv[2] if len(sys.argv) > 2 else "", kind="exp")
    elif cmd == "twin":       # direct twin of WM run <rid>: same params (+-5%) and same optimizer steps per fold
        rid = sys.argv[2]
        import csv
        row = [r for r in csv.DictReader(open(TSV), delimiter="\t") if r["run_id"] == rid][-1]
        wcfg = json.loads(row["cfg"])
        st = [int(x) for x in row["steps"].split("/")]
        nseed = len(row["seed"].split("+"))
        fs = {str(Y): st[i * nseed] for i, Y in enumerate(WD.FOLD_YEARS)}
        tc = C.matched_cfg({k: v for k, v in wcfg.items() if k not in ("fold_steps", "run_id")})
        tc["fold_steps"] = fs
        run_wm(tc, f"direct twin of {rid} (steps-matched)", kind="twin")
    elif cmd == "lock":       # pin the locked harness + panel at loop start
        open(LOCKED_PIN, "w").write(locked_sha())
        print("locked", open(LOCKED_PIN).read())
    elif cmd == "wm":
        cfg = json.loads(sys.argv[2])
        note = sys.argv[3] if len(sys.argv) > 3 else ""
        seeds = tuple(int(s) for s in os.environ.get("SEEDS", "0").split(","))
        run_wm(cfg, note, seeds)


def run_continual(rid, rule, note="", seeds=None):
    """apply an online rule to a logged WM run's fold checkpoints; logs one row (kind=continual)."""
    import glob
    import wm_continual as CL
    t0 = time.time()
    P = WD.load_panel()
    refs_ = load_refs()
    Qon, Qfr, info = {}, {}, []
    for Y, (tr, te) in folds(P, 20).items():
        norm = H.Norm(P["X"], P["Y"], tr)
        qo, qf = [], []
        for p in sorted(glob.glob(os.path.join(CK, f"{rid}_f{Y}_s*.pt"))):
            ck = torch.load(p, weights_only=False)
            m = C.WM(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
            a, b, inf = CL.run_fold(m, P, tr, te, norm, ck["cfg"].get("L", 20), rule)
            qo.append(a); qf.append(b); info.append(inf)
        Qon[Y] = np.sort(np.mean(qo, 0), -1); Qfr[Y] = np.sort(np.mean(qf, 0), -1)
    so, sf = score(Qon, refs_), score(Qfr, refs_)
    cid = f"cl_{dt.datetime.utcnow().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    np.savez(os.path.join(PRED, f"{cid}.npz"), **{str(Y): v for Y, v in Qon.items()})
    log_row(run_id=cid, utc=dt.datetime.utcnow().isoformat(timespec="seconds"), kind="continual", family="online",
            seed="", **so, status="ok", secs=int(time.time() - t0), leak="n/a",
            cfg=json.dumps(dict(rule, wm=rid), sort_keys=True),
            note=f"{note} frozen_S_F={sf['S_F']} S_C={round(so['S_F'] - sf['S_F'], 4)} rollbacks={sum(i['rollbacks'] for i in info)}")
    print(cid, "online", so, "frozen", sf["S_F"], f"{int(time.time() - t0)}s", flush=True)
    return cid, so, sf
