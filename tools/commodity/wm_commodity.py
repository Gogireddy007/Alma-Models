"""Harness registration of the frozen commodity world model and its direct twin (docs/commodity_search.md).

predict(context, horizon, quantiles) -> [len(quantiles), horizon] raw levels. Uses nothing but `context`
(1-D levels, oldest first): it is cropped to the model's context length, floored at 1.0 before the log (the same
floor used in training; it only matters around WTI 2020-04-20), and the model's forecast of the log change is
mapped back to levels. Deterministic: torch is re-seeded on every call. Sample-based models give every requested
level; quantile-head models give the 9 native levels 0.1..0.9 and NaN elsewhere. Horizons beyond the trained
rollout (H_MAX) are NaN (see docs/commodity_search.md: the candidate is scored at h <= its trained horizon).

The frozen checkpoints are listed in FROZEN below (path + sha256, verified at load).
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

# name -> list of (checkpoint relative to ROOT, sha256); several = seed ensemble (quantile average).
# Filled when the candidate is frozen (also written to runs/commodity/FROZEN.json).
FROZEN = {
    "wm_sv": [
        [
            "runs/commodity/search/ckpt/f_sv_none_s0.pt",
            "392a58a3a3596a1e8e71fe98061458e246bc5ab29a3326807f69af49edb8f2e2"
        ],
        [
            "runs/commodity/search/ckpt/f_sv_none_s1.pt",
            "19a14e94e3834c3f35b42bd4bf9903c3d635548685401c8a78a9bdd15299ca27"
        ],
        [
            "runs/commodity/search/ckpt/f_sv_none_s2.pt",
            "9d65eb3f6931001ef798c614a7a9cf6bcacf993755c106d07aa0c5461e13fdb1"
        ]
    ],
    "wm_sv_direct": [
        [
            "runs/commodity/search/ckpt/f_sv_direct_s0.pt",
            "8da818568e4e7ff344765ad31080c3e6236793c124d448f9faff91269dd80295"
        ],
        [
            "runs/commodity/search/ckpt/f_sv_direct_s1.pt",
            "173c123ce6dfedafcf4a1531d15ae9c457deb8618bdae694f41ba17511604953"
        ],
        [
            "runs/commodity/search/ckpt/f_sv_direct_s2.pt",
            "e92697cf7a90a7a5aa7caaacafe84fd319fae2a1852ecf941d5219c473927390"
        ]
    ]
}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_model(path, sha=None):
    import torch
    if sha is not None:
        got = _sha(path)
        assert got == sha, f"checkpoint {path}: sha256 {got} != frozen {sha}"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["arch"] == "alma":
        import wm_alma
        m = wm_alma.AlmaSSM(abl=ck["abl"], init=None, **ck["kw"])
    else:
        import wm_models
        m = wm_models.ARCHS[ck["arch"]](abl=ck["abl"], **ck["kw"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def make_predict_ens(members, seed=0):
    """members: list of (abs path, sha). Quantile average of the members' level forecasts (log-space average)."""
    preds = [make_predict(p, s, seed) for p, s in members]

    def batch(ctxs, h, qs):
        return np.exp(np.mean([np.log(p.batch(ctxs, h, qs)) for p in preds], 0))

    def predict(context, horizon, quantiles):
        return batch([context], horizon, quantiles)[0]

    predict.batch = batch
    predict.meta = {"members": [p.meta for p in preds], "combine": "mean of log-level quantiles"}
    return predict


def make_predict(path, sha=None, seed=0):
    import torch
    m, ck = load_model(path, sha)
    L = ck["L"]
    hmax = 64 if ck["arch"] == "alma" else m.Hmax
    assert ck.get("cols", [0]) == [0], "cross-commodity models need a multivariate context (not in the harness API)"

    def batch(ctxs, h, qs):
        qs = [float(x) for x in qs]
        n = len(ctxs)
        out = np.full((n, len(qs), h), np.nan)
        lp = np.full((n, L + 1, 1), np.nan, dtype=np.float32)
        last = np.empty(n)
        for i, c in enumerate(ctxs):
            c = np.log(np.maximum(np.asarray(c, dtype=np.float64)[-(L + 1):], 1.0))
            lp[i, L + 1 - len(c):, 0] = c
            last[i] = c[-1]
        hh = min(h, hmax)
        torch.manual_seed(seed)
        with torch.no_grad():
            x = torch.from_numpy(lp)
            if hasattr(m, "samples"):
                for i in range(n):                    # re-seed per origin: forecasts never depend on batch composition
                    torch.manual_seed(seed)
                    sam = m.samples(x[i:i + 1], hh).numpy()                   # [M, 1, hh] log change
                    out[i, :, :hh] = np.quantile(sam[:, 0], qs, axis=0)
            else:
                q9 = m.quantiles(x, hh).numpy()                               # [n, 9, hh]
                native = [round(0.1 * k, 1) for k in range(1, 10)]
                for j, q in enumerate(qs):
                    if round(q, 4) in native:
                        out[:, j, :hh] = np.sort(q9, 1)[:, native.index(round(q, 4))]
        return np.exp(last[:, None, None] + out)

    def predict(context, horizon, quantiles):
        return batch([context], horizon, quantiles)[0]

    predict.batch = batch
    predict.meta = {"ckpt": os.path.relpath(path, ROOT), "ckpt_sha256": _sha(path), "arch": ck["arch"],
                    "abl": ck["abl"], "context": L, "trained_horizon": hmax, "train_cut": ck["cut"]}
    return predict


def register(REGISTRY):
    for name, members in FROZEN.items():
        mem = [(os.path.join(ROOT, p), s) for p, s in members]
        REGISTRY[name] = (lambda mm: (lambda: make_predict_ens(mm)))(mem)
