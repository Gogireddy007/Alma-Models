"""Post-hoc quantile recalibration for train_wm.py, fitted on TRAIN series only (calibration engineer, 2026-10-31).

Used by train_wm.py behind TSAR_WM_RECAL (default 0 = off, bit-identical).

Map. For a forecast q[n, NQ, h] (after paths / TTA), with median m = q[:, NQ // 2]:
    q'_k[t] = m[t] + c[g, b(t), k] * (q_k[t] - m[t])         (then sorted; the TTA non-negative clip re-applied)
c is a positive scale per quantile level k, per lead bucket b(t) (leads 1-8, 9-16, 17-32, 33-64, 65-128, 129-256,
257+) and, in mode `freqlead`, per frequency group g (sub = sub-daily, day = daily, low = weekly and coarser). The
median is never moved, so the point forecast and MASE are unchanged. In mode `lead`, g is pooled ("all").

Fit. Each c[g, b, k] minimises the weighted pinball loss at level k over a grid of 57 log-spaced scales in [0.5, 4]
(1.0 is on the grid), on calibration windows built ONLY from bench TRAIN series (after TSAR_GIFT_CLEAN and
TRAIN_DROP; the two sources are under Leakage): from every dataset up to N series (fixed rng, independent of
TSAR_SEED), each split into context y[:-H] (last 2048 points, as the bench) and target y[-H:], H = CAL_H[group].
Each window is scaled by the mean absolute first difference of its context; every dataset weighs the same in every
(group, bucket) cell. A cell with fewer than MIN_WIN windows is not fitted and falls back to the pooled cell
(freqlead fits both); leads past the longest fitted bucket use the longest fitted pooled bucket. Pinball-optimal
scales calibrate the quantiles on the calibration set and directly minimise the wql-type loss there.

Leakage. VAL and TEST are never read here. The windows come from bench TRAIN series only. Two sources (train_wm):
the training pool itself (in-sample targets: the model trained on whole series, so the tails were inside training
windows), or, under TSAR_DEV=1, the DEV families that devsplit.holdout removed from training (out-of-family for that
model). A DEV-fitted map is saved (save) and applied to another checkpoint with TSAR_WM_RECAL=file (load).
"""
import json
import os

import numpy as np

CAL_H = {"sub": 288, "day": 56, "low": 13}      # target length by frequency group (VAL max: 288 / 28 / 13)
MIN_CTX = 32                                   # observed context points a calibration window needs
EDGES = np.array([8, 16, 32, 64, 128, 256])     # lead bucket upper edges; bucket 6 = 257+
NB = len(EDGES) + 1
GRID = np.exp(np.linspace(np.log(0.5), np.log(4.0), 57))
MIN_WIN = 30
SEED = 20261031


def freq_group(freq):
    f = str(freq).upper()
    if f.endswith("MIN") or f.endswith("T") or f in ("H", "S") or f.endswith("H") and f[:-1].isdigit():
        return "sub"
    if f == "D" or f.endswith("D") and f[:-1].isdigit():
        return "day"
    return "low"


def cal_len(y, freq):
    """Calibration target length for series y, or 0 if y is too short to split."""
    H = CAL_H[freq_group(freq)]
    if len(y) < H + MIN_CTX or np.isfinite(y[:-H]).sum() < MIN_CTX or not np.isfinite(y[-H:]).any():
        return 0
    return H


def lead_bucket(h):
    return np.searchsorted(EDGES, np.arange(1, h + 1), side="left")


def windows(cal_src, freq_of, n_per):
    """{dataset: (contexts, targets, H)} from TRAIN series only, N per dataset, fixed rng."""
    rng = np.random.default_rng(SEED)
    out = {}
    for d in sorted(cal_src):
        ys = [y for y in cal_src[d] if cal_len(y, freq_of[d])]
        if not ys:
            continue
        pick = rng.permutation(len(ys))[:n_per]
        H = CAL_H[freq_group(freq_of[d])]
        out[d] = ([np.asarray(ys[i][:-H][-2048:], dtype=np.float32) for i in pick],
                  np.stack([np.asarray(ys[i][-H:], dtype=np.float32) for i in pick]), H)
    return out


def _scale(c):
    c = c[np.isfinite(c)][-256:]
    s = np.abs(np.diff(c)).mean() if len(c) > 1 else 0.0
    if not s > 0:
        s = np.abs(c).mean() if len(c) else 0.0
    return s if s > 0 else 1.0


def _pinball(tau, y, p):
    e = y - p
    return np.maximum(tau * e, (tau - 1) * e)


def fit(preds, wins, freq_of, taus, mode, clip_fn=None):
    """preds: {dataset: q[n, NQ, H]} from the full predict(); returns (map, report lines)."""
    taus = np.asarray(taus, dtype=np.float64)
    NQ, mid = len(taus), len(taus) // 2
    cells = {}                                          # (g, b) -> list of per-dataset (y, m, dev, clip, w)
    for d, q in preds.items():
        ctxs, tgt, H = wins[d]
        groups = ("all", freq_group(freq_of[d])) if mode == "freqlead" else ("all",)
        q = np.sort(q.astype(np.float64), axis=1)
        s = np.array([_scale(c) for c in ctxs])[:, None]
        clip = clip_fn(ctxs) if clip_fn else np.zeros(len(ctxs), bool)
        b = lead_bucket(H)
        for bb in np.unique(b):
            sel = b == bb
            y = tgt[:, sel].astype(np.float64) / s
            ok = np.isfinite(y)
            if not ok.any():
                continue
            m = q[:, mid, sel] / s
            dev = (q[:, :, sel] - q[:, mid:mid + 1, sel]) / s[:, :, None]
            cl = np.broadcast_to(clip[:, None], y.shape)
            w = 1.0 / ok.sum()                          # each dataset weighs 1 in every cell
            for g in groups:                            # freqlead also fits the pooled cell (its fallback)
                cells.setdefault((g, int(bb)), []).append((y[ok], m[ok], dev.transpose(0, 2, 1)[ok], cl[ok], w,
                                                             len(ctxs)))
    cmap, rep = {}, []
    for key in sorted(cells):
        parts = cells[key]
        nwin = sum(p[5] for p in parts)
        if nwin < MIN_WIN:
            continue
        c = np.ones(NQ)
        y = np.concatenate([p[0] for p in parts]); m = np.concatenate([p[1] for p in parts])
        dev = np.concatenate([p[2] for p in parts]); cl = np.concatenate([p[3] for p in parts])
        w = np.concatenate([np.full(len(p[0]), p[4]) for p in parts])
        for k in range(NQ):
            if k == mid:
                continue
            P = m[None] + GRID[:, None] * dev[None, :, k]
            P = np.where(cl[None], np.maximum(P, 0), P) if clip_fn else P   # = the raw-space clip, as s > 0
            L = (_pinball(taus[k], y[None], P) * w[None]).sum(1)
            c[k] = GRID[int(np.argmin(L))]
        cmap[key] = c
        lo, hi = 0, NQ - 1
        cov0 = ((y >= m + dev[:, lo]) & (y <= m + dev[:, hi])).astype(float)
        cov1 = ((y >= m + c[lo] * dev[:, lo]) & (y <= m + c[hi] * dev[:, hi])).astype(float)
        rep.append(f"recal {key[0]:>4s} leads {_bname(key[1]):>8s}: {len(parts):2d} datasets {nwin:5d} windows  "
                   f"cov80 {np.average(cov0, weights=w):.3f} -> {np.average(cov1, weights=w):.3f}  "
                   f"c=[{' '.join(f'{v:.2f}' for v in c)}]")
    if not any(k[0] == "all" for k in cmap):
        raise SystemExit("TSAR_WM_RECAL: no calibration cell had enough windows")
    return cmap, rep


def _bname(b):
    lo = 1 if b == 0 else EDGES[b - 1] + 1
    return f"{lo}-{EDGES[b]}" if b < len(EDGES) else f"{lo}+"


def _lookup(cmap, g, b):
    """Scale vector for group g, bucket b: the (g, b) cell, else the pooled ("all", b) cell, else the longest pooled
    bucket below b (leads past the fitted range), else the first pooled bucket."""
    if (g, b) in cmap:
        return cmap[(g, b)]
    fitted = sorted(k[1] for k in cmap if k[0] == "all")
    below = [x for x in fitted if x <= b]
    return cmap[("all", below[-1] if below else fitted[0])]


def apply(q, freq, cmap, mode, clip=None):
    """q[n, NQ, h] -> recalibrated q (sorted). clip: bool[n] series to clamp at 0 (TTA clip), or None."""
    n, NQ, h = q.shape
    mid = NQ // 2
    g = freq_group(freq) if mode == "freqlead" else "all"
    b = lead_bucket(h)
    C = np.stack([_lookup(cmap, g, int(bb)) for bb in range(NB)])          # [NB, NQ]
    c = C[b].T[None]                                                       # [1, NQ, h]
    qs = np.sort(q, axis=1)
    m = qs[:, mid:mid + 1]
    out = m + c * (qs - m)
    if clip is not None:
        out = np.where(clip[:, None, None], np.maximum(out, 0), out)
    return np.ascontiguousarray(np.sort(out, axis=1), dtype=np.float32)


def save(path, cmap, mode, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {"mode": mode, "edges": EDGES.tolist(), "cal_h": CAL_H, "grid": [float(GRID[0]), float(GRID[-1]), len(GRID)],
           "cells": {f"{g}|{b}": [float(v) for v in c] for (g, b), c in sorted(cmap.items())}, **meta}
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(path + ".tmp", path)


def load(path):
    doc = json.load(open(path))
    if doc["edges"] != EDGES.tolist() or doc["cal_h"] != CAL_H:
        raise SystemExit(f"{path}: lead buckets / calibration lengths differ from tools/wmrecal.py")
    cmap = {(k.split("|")[0], int(k.split("|")[1])): np.asarray(v) for k, v in doc["cells"].items()}
    return cmap, doc
