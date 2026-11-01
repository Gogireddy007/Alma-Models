"""Auditor tests of the architects' pipelines (DEV only; no training).

A1  architect A training set: rebuild runs/fin/search/cache/ds_2024-06-30.npz into a scratch dir (a) clean,
    (b) with every row dated > 2024-06-30 randomised, (c) with every held-out-coin row randomised -> all three must
    equal the cached file used by the search (also detects a stale cache).
B1  architect B features: features(ctx) of the TRAINING coins must not depend on the held-out coins' data
    (B's loop cache is built on all 24 coins; pooled market features would leak held-out data into training inputs).
I1  inference: for every fold checkpoint on disk (A: runs/fin/search/ckpt/*.pt), randomise (not just rescale; NaN
    flips included) every daily value of days >= D and every hourly value of hours >= D 00:00 -> bit-identical
    outputs; randomise day D-1 -> outputs change.
C1  continual: WMContinual on a fold checkpoint over 10 DEV days; panel randomised from day j -> decisions <= j
    bit-identical, later ones differ.
P1  independent recomputation of the fold P1 accuracy of every A prediction file from the raw parquet (labels
    computed here, not by harness.labels), compared with results.tsv.

  nice -n 10 .venv/bin/python tools/fin/redteam/rt_arch.py [--only A1,B1,I1,C1,P1]
"""
import os
import sys
import copy
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R    # noqa: E402
H, DT = R.H, R.DT
sys.path.insert(0, os.path.join(R.FIN, "search"))
sys.path.insert(0, os.path.join(R.FIN, "search_b"))
SCR = os.environ.get("RT_SCRATCH", "/tmp/fin_redteam")
DAYS = ["2024-08-01", "2024-12-02", "2025-03-03", "2025-06-30"]
LINES, OK = [], [True]


def rep(s, ok):
    OK[0] &= bool(ok)
    LINES.append(("OK   " if ok else "FAIL ") + s)
    print(LINES[-1], flush=True)


def rand_future(P, i, rng, from_prev=False):
    Q = copy.deepcopy(P)
    k = i - 1 if from_prev else i
    for f, a in Q.daily.items():
        sl = slice(k, k + 1) if from_prev else slice(k, None)
        x = a[sl]
        v = rng.lognormal(0, 1, x.shape) * (np.nanmedian(np.abs(a[:k])) if np.isfinite(a[:k]).any() else 1.0)
        v[rng.random(x.shape) < 0.05] = np.nan
        a[sl] = v
    if Q.hourly is not None and not from_prev:
        for f, a in Q.hourly.items():
            x = a[24 * i:]
            v = (rng.lognormal(0, 1, x.shape) * (np.nanmedian(np.abs(a[:24 * i])) + 1e-9)).astype(np.float32)
            v[rng.random(x.shape) < 0.05] = np.nan
            a[24 * i:] = v
    return Q


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a.shape == b.shape and np.array_equal(a, b)


def test_A1():
    import common as C
    import wm_core as WC
    orig = C.CACHE
    tr, _ = DT.universe()
    variants = [("legacy", "ds_2024-06-30.npz", lambda: C.build_dataset(C.train_panel(C.FOLD_FIT_END), C.FOLD_FIT_END))]
    if hasattr(WC, "W_MAX"):
        variants.append(("loop", f"ds_w{WC.W_MAX}_2024-06-30.npz",
                         lambda: C.build_dataset(C.cut_panel(H.Panel("crypto"), C.FOLD_FIT_END), C.FOLD_FIT_END,
                                                 coins=tr, wmax=WC.W_MAX)))
    for vname, fn, build in variants:
        cached = os.path.join(orig, fn)
        if not os.path.exists(cached):
            rep(f"A1 {vname}: no cache {fn}", True)
            continue
        ref = np.load(cached)
        for tag, kw in (("clean", {}), ("future", dict(cut=C.FOLD_FIT_END, mode="rand", seed=11)),
                        ("heldout", dict(heldout=True, mode="rand", seed=12))):
            C.CACHE = os.path.join(SCR, f"A1_{vname}_{tag}")
            os.makedirs(C.CACHE, exist_ok=True)
            p = os.path.join(C.CACHE, fn)
            if os.path.exists(p):
                os.remove(p)
            if kw:
                with R.perturbed_load(**kw):
                    D = build()
            else:
                D = build()
            keys = [k for k in ref.files if k in D]
            bad = [k for k in keys if not _eq(ref[k], D[k])]
            rep(f"A1 A dataset {vname} ({tag}) == cached search dataset {fn} (keys {len(keys)})"
                + ("" if not bad else f" -> differs in {bad}"), not bad)
    C.CACHE = orig


def test_B1():
    try:
        import feats_b as FB
    except Exception as e:
        rep(f"B1 skipped ({e})", True)
        return
    P = FB.loop_panel()
    _, ho = DT.universe()
    tr = np.array([s not in ho for s in P.symbols])
    with R.perturbed_load(heldout=True, mode="rand", seed=21):
        Q = FB.loop_panel()
    nd = {}
    for d in ["2023-02-01", "2023-09-01", "2024-03-01", "2024-06-30"]:
        i = P.index(d)
        f0 = FB.features(H.build_context(P, i))
        f1 = FB.features(H.build_context(Q, i))
        for k in ("X", "aux", "hz", "har", "A", "ref"):
            a, b = f0[k][tr], f1[k][tr]
            if not np.array_equal(a, b, equal_nan=True):
                if k == "X":
                    ch = [c for c in range(a.shape[-1]) if not np.array_equal(a[..., c], b[..., c], equal_nan=True)]
                    nd.setdefault(k, set()).update(ch)
                else:
                    nd.setdefault(k, set())
    rep(f"B1 B features of training coins independent of held-out data "
        f"{'' if not nd else '-> DIFFER: ' + str({k: sorted(v) for k, v in nd.items()})}", not nd)
    # same question for A's build (training panel has the 16 coins only) and for inference context composition
    import wm_core as WC
    i = P.index("2024-03-01")
    c0 = H.build_context(P, i)
    X0, _, _, _ = WC.make_features(c0["daily"], c0["hourly"], "crypto")
    c1 = H.build_context(P, i, symbols=[s for s, t in zip(P.symbols, tr) if t])
    X1, _, _, _ = WC.make_features(c1["daily"], c1["hourly"], "crypto")
    rep("B1b wm_core features of a coin depend on which OTHER coins are in the context (market factor): "
        "training (16 coins) vs evaluation (24 coins) inputs differ -> " +
        str(not np.array_equal(X0[tr], X1)), True)


def ckpts_A():
    fs = sorted(glob.glob(os.path.join(R.ROOT, "runs", "fin", "search", "ckpt", "*.pt")), key=os.path.getmtime)
    n = int(os.environ.get("RT_CKPT_LAST", "0"))
    return fs[-n:] if n else fs


def test_I1(rng):
    import wm_core as WC
    import wm_fin as WF
    import common as C
    P = C.dev_panel("crypto")
    E = C.dev_panel("etf")
    for ck in ckpts_A():
        if "agent" in os.path.basename(ck) or "ppo" in ck or "dqn" in ck:
            continue           # policies are tested through their own wrappers when present
        try:
            ms, meta = WC.load_members(ck)
        except Exception as e:
            rep(f"I1 {os.path.basename(ck)}: not a WM member file ({type(e).__name__})", True)
            continue
        fc = WF.WMForecaster(ms, meta)
        for mk, PP in (("crypto", P), ("etf", E)):
            for d in DAYS:
                i = PP.index(d)
                i = min(i, len(PP.days) - 1)
                o0 = R.out_of(H.run(PP, fc, [i], online=False))
                o1 = R.out_of(H.run(rand_future(PP, i, rng), fc, [i], online=False))
                o2 = R.out_of(H.run(rand_future(PP, i, rng, from_prev=True), fc, [i], online=False))
                rep(f"I1 {os.path.basename(ck)} {mk} {d}: future randomised -> identical; D-1 randomised -> changed",
                    R.same(o0, o1) and not R.same(o0, o2))


def test_C1(rng):
    import wm_core as WC
    import wm_fin as WF
    import common as C
    cks = [c for c in ckpts_A() if "_none_" in c]
    if not cks:
        rep("C1 skipped (no rollout ckpt)", True)
        return
    P = C.dev_panel("crypto")
    ms, meta = WC.load_members(cks[0])
    i0 = P.index("2025-01-06")
    days = list(range(i0, i0 + 10))
    j = i0 + 5
    r0 = H.run(P, WF.WMContinual(ms, meta), days)
    r1 = H.run(rand_future(P, j, rng), WF.WMContinual(ms, meta), days)
    o0, o1 = R.out_of(r0), R.out_of(r1)
    k = days.index(j)
    rep(f"C1 WMContinual({os.path.basename(cks[0])}): decisions <= perturbation day identical "
        f"(updates {r0['updates'].tolist()})", R.same(o0[:k + 1], o1[:k + 1]))
    rep("C1 WMContinual: later decisions differ (it does learn from revealed rows)", not R.same(o0[k + 1:], o1[k + 1:]))


def raw_labels():
    d = pd.read_parquet(DT.FILES["crypto_daily"])
    tr, ho = DT.universe()
    d = d[d["symbol"].isin(tr + ho)]
    lr = d.pivot(index="t", columns="symbol", values="rv")
    lr = lr.reindex(pd.date_range(lr.index.min(), lr.index.max(), freq="D"))[tr + ho]
    lr = np.log(lr.where(lr > 0))
    ref = lr.rolling(7, min_periods=7).mean().shift(1)            # mean of D-7..D-1 (all 7 required)
    y = (lr > ref).astype(float).where(lr.notna() & ref.notna() & (lr != ref))
    return y, tr + ho, ho


def test_P1():
    import common as C
    y, syms, ho = raw_labels()
    P = C.dev_panel("crypto")
    tsv = pd.read_csv(C.TSV, sep="\t")
    for f in sorted(glob.glob(os.path.join(C.OUT, "preds", "*_crypto.npz"))):
        z = np.load(f)
        if "p_up" not in z.files:
            continue
        name = os.path.basename(f)[:-len("_crypto.npz")]
        days = pd.DatetimeIndex(P.days[z["days"]])
        zs = list(z["symbols"])
        Y = y.loc[days, zs].to_numpy()
        pu = z["p_up"]
        if np.isfinite(pu).any():
            dd = np.where(np.isfinite(pu), (pu > 0.5).astype(float), np.nan)
        else:
            continue
        ok = np.isfinite(dd) & np.isfinite(Y)
        acc = float(((dd == Y) & ok).sum() / ok.sum())
        hoc = np.array([s in ho for s in zs])
        acc_ho = float(((dd == Y) & ok)[:, hoc].sum() / ok[:, hoc].sum())
        row = tsv[tsv["name"] == (name[4:] if name.startswith("ref_") else name)]
        row = row[row["kind"] == ("ref" if name.startswith("ref_") else row["kind"].iloc[0])] if len(row) else row
        logged = float(row["P1_all"].iloc[-1]) if len(row) and pd.notna(row["P1_all"].iloc[-1]) else np.nan
        rep(f"P1 recomputed {name}: acc {acc:.4f} (held-out {acc_ho:.4f}, n {int(ok.sum())}) vs results.tsv {logged:.4f}",
            not np.isfinite(logged) or abs(acc - logged) < 5e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="P1,B1,I1,C1,A1")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    for t in a.only.split(","):
        {"A1": test_A1, "B1": test_B1, "I1": lambda: test_I1(rng), "C1": lambda: test_C1(rng), "P1": test_P1, "P1b": test_P1b, "I1b": lambda: test_I1b(rng)}[t]()
    LINES.append("ALL OK" if OK[0] else "FAILURES")
    with open(os.path.join(R.HERE, "rt_arch.txt"), "a") as fh:
        fh.write(f"# {pd.Timestamp.now()} --only {a.only}\n" + "\n".join(LINES) + "\n")
    print(LINES[-1])



def test_P1b():
    """B: recompute every fold P1 in runs/fin/search_b/preds from raw labels; compare with results_b.tsv."""
    y, syms, ho = raw_labels()
    outb = os.path.join(R.ROOT, "runs", "fin", "search_b")
    zc = np.load(os.path.join(outb, "cache", "loop3_2024-06-30.npz" if os.path.exists(os.path.join(outb, "cache", "loop3_2024-06-30.npz")) else "loop_2024-06-30.npz"))
    days, csyms = pd.DatetimeIndex(zc["days"]), list(zc["symbols"])
    tsv = pd.read_csv(os.path.join(outb, "results_b.tsv"), sep="\t", dtype={"seed": str})
    n = 0
    for f in sorted(glob.glob(os.path.join(outb, "preds", "*_E[123].npz"))):
        z = np.load(f)
        if "p" not in z.files or "idx" not in z.files:
            continue
        base = os.path.basename(f)[:-4]
        name, fold = base.rsplit("_", 1)
        nm, seed = name.rsplit("_s", 1) if "_s" in name else (name, "0")
        Y = y.loc[days[z["idx"]], csyms].to_numpy()
        p = z["p"]
        d = np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan)
        ok = np.isfinite(d) & np.isfinite(Y)
        acc = float(((d == Y) & ok).sum() / max(1, ok.sum()))
        row = tsv[(tsv["name"] == nm) & (tsv["fold"] == fold) & (tsv["seed"].astype(str) == seed)]
        logged = float(row["P1_all"].iloc[-1]) if len(row) else np.nan
        if np.isfinite(logged):
            n += 1
            rep(f"P1b {nm} s{seed} {fold}: recomputed {acc:.4f} vs results_b {logged:.4f}", abs(acc - logged) < 5e-5)
        else:
            LINES.append(f"INFO P1b {base}: no matching results_b row (policy fc preds or unlogged): acc {acc:.4f}")
            print(LINES[-1])
    rep(f"P1b: {n} B fold scores re-derived", n > 0)



def test_I1b(rng):
    """B checkpoints (runs/fin/search_b/ckpt/*.pt) through wm_b_fin wrappers: forecaster, continual, voltgt agent."""
    import wm_b_fin as WB
    import common as C
    P = C.dev_panel("crypto")
    E = C.dev_panel("etf")
    for ck in sorted(glob.glob(os.path.join(R.ROOT, "runs", "fin", "search_b", "ckpt", "*.pt"))):
        rel = os.path.relpath(ck, R.ROOT)
        makers = {"fc": lambda: WB.BForecaster(*WB.load_ckpt(rel)),
                  "voltgt": lambda: WB.BVolTgt(WB.BForecaster(*WB.load_ckpt(rel)))}
        for kind, mk_ in makers.items():
            for mk, PP in (("crypto", P), ("etf", E)):
                for d in DAYS:
                    i = min(PP.index(d), len(PP.days) - 1)
                    o0 = R.out_of(H.run(PP, mk_(), [i], online=False))
                    o1 = R.out_of(H.run(rand_future(PP, i, rng), mk_(), [i], online=False))
                    o2 = R.out_of(H.run(rand_future(PP, i, rng, from_prev=True), mk_(), [i], online=False))
                    rep(f"I1b {os.path.basename(ck)} {kind} {mk} {d}: future randomised -> identical; D-1 -> changed",
                        R.same(o0, o1) and not R.same(o0, o2))
        if hasattr(WB, "BContinual"):
            i0 = P.index("2025-01-06"); days = list(range(i0, i0 + 8)); j = i0 + 4
            r0 = H.run(P, WB.BContinual(*WB.load_ckpt(rel)), days)
            r1 = H.run(rand_future(P, j, rng), WB.BContinual(*WB.load_ckpt(rel)), days)
            k = days.index(j)
            rep(f"I1b {os.path.basename(ck)} BContinual: decisions <= perturbation day identical",
                R.same(R.out_of(r0)[:k + 1], R.out_of(r1)[:k + 1]))


if __name__ == "__main__":
    main()
