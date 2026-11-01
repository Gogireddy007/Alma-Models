"""Shared search utilities (architect). DEV only: every panel is cut at decision day 2025-06-30 right after loading."""
import os
import sys
import json
import time
import hashlib
import contextlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIN))
sys.path.insert(0, FIN)
import harness as H      # noqa: E402
import data as DT        # noqa: E402
import metrics as MT     # noqa: E402
import wm_core as WC     # noqa: E402

DEV_END = pd.Timestamp("2025-06-30")
FOLD_FIT_END = pd.Timestamp("2024-06-30")
FOLD_BT = (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30"))
OUT = os.path.join(ROOT, "runs", "fin", "search")
CK = os.path.join(OUT, "ckpt")
CACHE = os.path.join(OUT, "cache")
TSV = os.path.join(OUT, "results.tsv")
I9 = [H.QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
for _d in (OUT, CK, CACHE, os.path.join(OUT, "logs")):
    os.makedirs(_d, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cut_panel(P, end=DEV_END, symbols=None):
    """Truncate a Panel in place to decision days <= end (and optionally to `symbols`)."""
    e = int(np.searchsorted(P.days, np.datetime64(end, "ns"), side="right"))
    keep = list(P.symbols) if symbols is None else [s for s in P.symbols if s in symbols]
    js = [P.symbols.index(s) for s in keep]
    for f in P.daily:
        P.daily[f] = P.daily[f][:e][:, js]
    if P.hourly is not None:
        for f in P.hourly:
            P.hourly[f] = P.hourly[f][:24 * e][:, js]
    P.days, P.symbols = P.days[:e], keep
    P.last_bar, P.alive, P.cost = P.last_bar[:e][:, js], P.alive[:e][:, js], P.cost[js]
    P.group = {s: P.group[s] for s in keep}
    assert P.days.max() <= np.datetime64(end, "ns")
    return P


def dev_panel(market="crypto"):
    return cut_panel(H.Panel(market))


def train_panel(fit_end):
    """Training coins only, labels <= fit_end (<= 2025-06-30 asserted)."""
    assert pd.Timestamp(fit_end) <= DEV_END
    P = H.Panel("crypto")
    tr, _ = DT.universe()
    return cut_panel(P, fit_end, tr)


@contextlib.contextmanager
def fit_end_patch(fit_end):
    """Lead factories cut their fitting data at data.TRAIN_END (models.train_view): temporarily move it earlier."""
    old = DT.TRAIN_END
    assert pd.Timestamp(fit_end) <= old
    DT.TRAIN_END = pd.Timestamp(fit_end)
    try:
        yield
    finally:
        DT.TRAIN_END = old


def bt_days(P, a=FOLD_BT[0], b=FOLD_BT[1]):
    sel = (P.days >= np.datetime64(a, "ns")) & (P.days <= np.datetime64(b, "ns"))
    return np.nonzero(sel)[0]


# --------------------------------------------------------------------------- training dataset
def build_dataset(P, fit_end, first="2020-03-01", coins=None, wmax=None):
    """Samples for decision days i with label day <= fit_end, for `coins` (default: all of P's symbols, which for a
    train_panel are the 16 training coins). Features via build_context restricted to `coins` (identical information
    set and market factor as a training-coin panel). wmax=None: legacy W=14 cache; wmax=28: loop cache (models slice
    the last W days, identical to computing with W directly). Cached."""
    fit_end = pd.Timestamp(fit_end)
    tag = "" if wmax is None else f"w{wmax}_"
    path = os.path.join(CACHE, f"ds_{tag}{fit_end.date()}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    coins = list(P.symbols) if coins is None else list(coins)
    cj = np.array([P.symbols.index(s) for s in coins])
    i0 = max(P.index(first), 60)
    i1 = int(np.searchsorted(P.days, np.datetime64(fit_end, "ns"), side="right"))
    out = {k: [] for k in ("X", "aux", "A", "tl", "y", "yr", "rn", "blk", "mblk", "coin", "day", "cost")}
    t0 = time.time()
    for i in range(i0, i1):
        assert P.days[i] <= np.datetime64(fit_end, "ns")
        ctx = H.build_context(P, i, ctx_days=95, symbols=coins)
        if wmax is None:
            X, aux, A, ok = WC.make_features(ctx["daily"], ctx["hourly"], "crypto")
        else:
            X, aux, A, ok = WC.make_features(ctx["daily"], ctx["hourly"], "crypto", W=wmax)
        lab = P.labels(i)
        rv5 = P.hourly["rv5"][24 * i:24 * (i + 1)][:, cj].astype(float)    # label-day intraday (target only)
        blk = rv5.reshape(WC.NB, 24 // WC.NB, -1).sum(1).T                  # [m, 4]
        good = np.isfinite(blk).all(1) & (blk > 0).all(1)
        lb = np.log(np.maximum(WC.NB * blk, 1e-12)) - A[:, None]
        lr_, yv, yr, rt = lab["logrv"][cj], lab["y_vol"][cj], lab["y_ret"][cj], lab["ret"][cj]
        tl = lr_ - A
        keep = ok & np.isfinite(tl)
        for j in np.nonzero(keep)[0]:
            out["X"].append(X[j]); out["aux"].append(aux[j]); out["A"].append(A[j])
            out["tl"].append(tl[j]); out["y"].append(yv[j]); out["yr"].append(yr[j])
            out["rn"].append(np.clip(rt[j] / np.exp(A[j] / 2), -6, 6) if np.isfinite(rt[j]) else np.nan)
            out["blk"].append(np.where(good[j], lb[j], np.nan)); out["mblk"].append(float(good[j]))
            out["coin"].append(cj[j]); out["day"].append(i); out["cost"].append(P.cost[cj[j]])
    D = {k: np.asarray(v, np.float32 if k not in ("coin", "day") else np.int64) for k, v in out.items()}
    D["fit_end"] = np.array(str(fit_end.date()))
    D["day_str"] = P.days[D["day"]].astype("datetime64[D]").astype(str)
    add_fill_rewards(D, P)
    np.savez(path, **D)
    print(f"dataset {path}: {len(D['X'])} samples, {time.time() - t0:.0f}s", flush=True)
    return D


def add_fill_rewards(D, P):
    """RT-8: realised harness-style trading outcome of each sample (training-side label only): rn_fill =
    (fill(D+1)/fill(D) - 1) / exp(A/2) and fund_real = funding settled in (D 00:00, D+1 00:00]. Both are complete by
    D+1 00:10; subset() blanks them for the fold's last label day so nothing after fit_end is used."""
    i, j = D["day"], D["coin"]
    f0 = P.daily["fill"][i, j]
    f1 = np.where(i + 1 < len(P.days), P.daily["fill"][np.minimum(i + 1, len(P.days) - 1), j], np.nan)
    r = (f1 / f0 - 1) / np.exp(D["A"].astype(float) / 2)
    D["rn_fill"] = np.clip(r, -6, 6).astype(np.float32)
    D["fund_real"] = np.nan_to_num(P.daily["funding"][i, j]).astype(np.float32)
    return D


def subset(D, fit_end):
    """samples whose label day <= fit_end (asserted)."""
    keep = D["day_str"] <= str(pd.Timestamp(fit_end).date())
    S = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v) for k, v in D.items()}
    if "rn_fill" in S:
        last = S["day_str"] == str(pd.Timestamp(fit_end).date())
        S["rn_fill"] = np.where(last, np.nan, S["rn_fill"]).astype(np.float32)
        S["fund_real"] = np.where(last, 0.0, S["fund_real"]).astype(np.float32)
        assert not np.isfinite(S["rn_fill"][last]).any(), "fill-based reward of the last label day must be purged"
    assert max(S["day_str"].tolist()) <= str(pd.Timestamp(fit_end).date())
    return S


# --------------------------------------------------------------------------- metrics on the fold
def p1_p3(P, res):
    """res: harness.run output (forecaster). Returns dict by group of P1 acc and P3 QS + day-level series."""
    days = res["days"]
    L = [P.labels(int(i)) for i in days]
    Y = np.stack([l["y_vol"] for l in L]); LR = np.stack([l["logrv"] for l in L]); REF = np.stack([l["ref"] for l in L])
    groups = {"all": list(P.symbols)}
    if P.market == "crypto":
        groups["train"] = [s for s in P.symbols if P.group[s] == "train"]
        groups["heldout"] = [s for s in P.symbols if P.group[s] == "heldout"]
    out = {}
    for g, syms in groups.items():
        js = [res["symbols"].index(s) for s in syms]
        pu, q = res["p_up"][:, js], res["q"][:, js]
        y, lr, ref = Y[:, js], LR[:, js], REF[:, js]
        if np.isfinite(pu).any():
            d = np.where(np.isfinite(pu), (pu > 0.5).astype(float), np.nan)
        else:
            med = q[:, :, H.QS.index(0.5)]
            d = np.where(np.isfinite(med) & np.isfinite(ref), (med > ref).astype(float), np.nan)
        ok = np.isfinite(d) & np.isfinite(y)
        hit = np.where(ok, (d == y).astype(float), 0.0)
        den = ok.sum(1)
        dayhit = np.where(den > 0, hit.sum(1) / np.maximum(den, 1), np.nan)
        okq = np.isfinite(lr) & np.isfinite(q[:, :, I9]).all(-1)
        qs = np.full(lr.shape, np.nan)
        if okq.any():
            qs[okq] = MT.pinball(lr[okq], q[:, :, I9][okq], [H.QS[i] for i in I9])
        out[g] = {"acc": float(hit.sum() / max(1, ok.sum())), "n": int(ok.sum()),
                  "qs": float(np.nanmean(qs)) if okq.any() else np.nan,
                  "day_acc": dayhit, "day_qs": np.nanmean(np.where(okq, qs, np.nan), 1) if okq.any() else None}
    return out


def p2(P, res, cost_mult=1.0):
    out = {}
    groups = {"all": list(P.symbols)}
    if P.market == "crypto":
        groups["heldout"] = [s for s in P.symbols if P.group[s] == "heldout"]
        groups["train"] = [s for s in P.symbols if P.group[s] == "train"]
    days, pos = res["days"], res["pos"]
    if len(days) and int(days[-1]) == len(P.days) - 1:     # exit fill outside the panel: drop (aligned with B)
        days, pos = days[:-1], pos[:-1]
    for g, syms in groups.items():
        js = [res["symbols"].index(s) for s in syms]
        pl = H.pnl(P, days, pos[:, js], symbols=syms, cost_mult=cost_mult)
        s = MT.trading_summary(pl["net"], pl["turnover"], P.periods)
        s["net"] = pl["net"]
        s["sr_period"] = float(np.mean(pl["net"]) / np.std(pl["net"], ddof=1)) if np.std(pl["net"]) > 0 else 0.0
        out[g] = s
    return out


def log_row(row):
    cols = ["time", "kind", "name", "config", "seed", "fold", "status", "P1_all", "P1_train", "P1_heldout", "P3_all",
            "E1_etf", "E3_etf", "SR_heldout", "SR_all", "SR_etf", "turn_heldout", "mdd_heldout", "note"]
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(cols) + "\n")
        row = dict(row, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        f.write("\t".join(_fmt(row.get(c, "")) for c in cols) + "\n")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.5f}"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def save_preds(name, res):
    p = os.path.join(OUT, "preds", f"{name}.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)}, symbols=np.array(res["symbols"]))
    return p
