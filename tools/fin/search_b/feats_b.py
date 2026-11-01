"""Architect B: feature extraction (shared by training cache and wrappers) and the DEV-early loop cache.

Features are computed ONLY from harness-context-shaped dicts (ctx["daily"], ctx["hourly"]), so training samples
(built with harness.build_context) and inference use identical code and information sets.

Loop cache: the crypto Panel is cut at decision day LOOP_END = 2024-06-30 right after loading (asserted), so no
value dated after 2024-06-30 is ever in memory in the search loop (B's honesty rule). Targets (labels of day D,
asset return fill(D)->fill(D+1), funding of D, intraday blocks of D) are stored separately and only used as targets.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIN))
sys.path.insert(0, FIN)
import wm_core as WC     # noqa: E402

LOOP_END = pd.Timestamp("2024-06-30")
OUTB = os.path.join(ROOT, "runs", "fin", "search_b")
CACHEB = os.path.join(OUTB, "cache")
NH = 168                 # hours of intraday context (last 7 days)
NHC = 4                  # hourly channels: log rv5 (daily units) - A, ret / sqrt(day var), taker - 0.5, mask
NHAR = 15                # HAR block: d1, w5, m22, ref7, q90 mean, |r1|, r1, park1 (all relative to A where log)
                         # + one-hot weekday of the decision day D (calendar, known at decision; not an asset id)


def hourly_feats(hourly, A, m):
    """hourly[f] [Lh, m] (hours ending <= D 00:00) or None -> [m, NH, NHC] float32."""
    out = np.zeros((m, NH, NHC), np.float32)
    if hourly is None:
        return out
    rv5 = WC._tail(np.asarray(hourly["rv5"], float), NH)
    ret = WC._tail(np.asarray(hourly["ret"], float), NH)
    tk = WC._tail(np.asarray(hourly["taker_ratio"], float), NH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lrv = np.log(np.maximum(24 * rv5, 1e-12)) - A[None, :]
        good = np.isfinite(rv5) & (rv5 > 0)
        r = ret / np.exp(A / 2)[None, :] * np.sqrt(24)
    out[:, :, 0] = np.clip(np.where(good, lrv, 0.0), -8, 6).T
    out[:, :, 1] = np.clip(np.nan_to_num(r), -6, 6).T
    out[:, :, 2] = np.clip(np.nan_to_num((tk - 0.5) * 10), -3, 3).T
    out[:, :, 3] = good.T
    return out


def har_feats(daily, A):
    lr = WC._tail(np.asarray(daily["logrv"], float), 90)
    r = WC._tail(np.asarray(daily["ret"], float), 2)[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d1 = lr[-1]
        w5 = np.nanmean(lr[-5:], 0)
        m22 = np.nanmean(lr[-22:], 0)
        ref = np.nanmean(lr[-7:], 0)
        q90 = np.nanmean(lr, 0)
        rn = r / np.exp(A / 2)
        hi = WC._tail(np.asarray(daily["high"], float), 1)[-1]
        lo = WC._tail(np.asarray(daily["low"], float), 1)[-1]
        pk = np.log(np.maximum(np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12)) ** 2 / (4 * np.log(2)), 1e-12))
    H = np.stack([d1 - A, w5 - A, m22 - A, ref - A, q90 - A, np.abs(rn), rn, pk - A], 1)
    H = np.clip(np.nan_to_num(H), -8, 8)
    return H.astype(np.float32)


def features(ctx):
    """-> dict X [m,W,NF], aux [m,3], A [m], ok [m], hz [m,NH,NHC], har [m,NHAR], ref [m] (mean log RV D-7..D-1)."""
    X, aux, A, ok = WC.make_features(ctx["daily"], ctx.get("hourly"), ctx["market"])
    m = X.shape[0]
    hz = hourly_feats(ctx.get("hourly"), A, m)
    har = har_feats(ctx["daily"], A)
    dow = np.zeros((m, 7), np.float32)
    dow[:, pd.Timestamp(ctx["day"]).day_of_week] = 1.0
    har = np.concatenate([har, dow], 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.nanmean(np.asarray(ctx["daily"]["logrv"], float)[-7:], 0)
        rr = WC._tail(np.asarray(ctx["daily"]["ret"], float), 60)
        mom = np.stack([np.nansum(rr[-k:], 0) for k in (1, 7, 14, 30, 60)], 1)
        fu = np.nan_to_num(WC._tail(np.asarray(ctx["daily"]["funding"], float), 1)[-1]) if "funding" in ctx["daily"] \
            else np.zeros(m)
    return {"X": X, "aux": aux, "A": A.astype(np.float64), "ok": ok, "hz": hz, "har": har, "ref": ref,
            "mom": np.nan_to_num(mom), "fu_last": fu}


def loop_panel():
    import harness as H
    sys.path.insert(0, os.path.join(FIN, "search"))
    import common as C
    P = C.cut_panel(H.Panel("crypto"), LOOP_END)
    assert P.days.max() <= np.datetime64(LOOP_END, "ns")
    return P


def build_cache(first="2020-03-01", end=None):
    """Per decision day (all 24 coins) features + targets, days first..2024-06-30. Panel cut at 2024-06-30.
    end='2025-06-30' builds the FREEZE-REFIT cache (DEV panel cut at 2025-06-30); used only by holdout_b --refit,
    after selection (no score is computed from it)."""
    global LOOP_END
    if end is not None:
        assert pd.Timestamp(end) <= pd.Timestamp("2025-06-30")
        old, LOOP_END = LOOP_END, pd.Timestamp(end)
        try:
            return _build(first, os.path.join(CACHEB, f"refit4_{pd.Timestamp(end).date()}.npz"))
        finally:
            LOOP_END = old
    return _build(first, os.path.join(CACHEB, "loop4_2024-06-30.npz"))


def _build(first, path):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    import harness as H
    os.makedirs(CACHEB, exist_ok=True)
    P = loop_panel()
    i0 = max(P.index(first), 100)
    tr_syms = [s for s in P.symbols if P.group[s] == "train"]
    assert tr_syms == P.symbols[:len(tr_syms)]
    keys = ("Xtr", "X", "aux", "A", "ok", "hz", "har", "ref", "mom", "fu_last", "tl", "y", "yr", "lrv", "blk", "mblk", "r", "fund", "alive")
    out = {k: [] for k in keys}
    days = []
    t0 = time.time()
    for i in range(i0, len(P.days)):
        ctx = H.build_context(P, i)
        F = features(ctx)
        # RT-4: training samples' features from a context restricted to the 16 training coins (the market-factor
        # channels X[...,5:7] are cross-sectional means; held-out coins must never enter a training input)
        Ft = features(H.build_context(P, i, symbols=tr_syms))
        out["Xtr"].append(Ft["X"])
        lab = P.labels(i)
        r, f, _ = P.asset_return(i)
        rv5 = P.hourly["rv5"][24 * i:24 * (i + 1)].astype(float)
        blk = rv5.reshape(WC.SUB, 24 // WC.SUB, -1).sum(1).T
        good = np.isfinite(blk).all(1) & (blk > 0).all(1)
        lb = np.where(good[:, None], np.log(np.maximum(WC.SUB * blk, 1e-12)) - F["A"][:, None], np.nan)
        for k in ("X", "aux", "A", "ok", "hz", "har", "ref", "mom", "fu_last"):
            out[k].append(F[k])
        out["tl"].append(lab["logrv"] - F["A"]); out["y"].append(lab["y_vol"]); out["yr"].append(lab["y_ret"])
        out["lrv"].append(lab["logrv"]); out["blk"].append(lb); out["mblk"].append(good.astype(float))
        out["r"].append(r); out["fund"].append(f); out["alive"].append(np.isfinite(P.daily["fill"][i]))
        days.append(P.days[i])
    D = {k: np.stack(v).astype(np.float32 if k not in ("A", "ref", "lrv", "r", "fund", "mom", "fu_last") else np.float64)
         for k, v in out.items()}
    D["ok"] = D["ok"].astype(bool); D["alive"] = D["alive"].astype(bool)
    D["days"] = np.array(days, dtype="datetime64[ns]")
    D["train_mask"] = np.array([P.group[s] == "train" for s in P.symbols])
    D["symbols"] = np.array(P.symbols)
    D["cost"] = P.cost.astype(np.float64)
    assert D["days"].max() <= np.datetime64(LOOP_END, "ns")
    np.savez(path, **D)
    print(f"cache {path}: {len(days)} days, {time.time() - t0:.0f}s", flush=True)
    return D


if __name__ == "__main__":
    build_cache()
