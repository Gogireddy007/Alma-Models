"""Score seed ensembles (quantile average of saved DEV-backtest predictions) and WM-vs-twin differences with a
circular 7-day block bootstrap over target days. Usage: ens_eval.py WM_a,WM_b,WM_c TWIN_a,TWIN_b,TWIN_c"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_data as W
from run import score, BT, OUT
D = W.PowerData("dev")


def ens(names):
    zs = [np.load(os.path.join(OUT, "pred", n + ".npz")) for n in names]
    assert all((z["idx"] == zs[0]["idx"]).all() for z in zs)
    return np.sort(np.mean([z["q"] for z in zs], 0), 1), zs[0]["idx"]


def per_day(idx, q):
    """Per-day components so the bootstrap does not recompute naive_epf: p1 hit (nan on label ties), p2 hits/count,
    QS of model and of naive_epf."""
    from run import naive_epf, LV9
    import wm_models as M
    y, y1 = D.P[idx], D.P[idx - 1]
    med = q[:, 6]
    lab = y.mean(1) > y1.mean(1)
    h1 = ((med.mean(1) > y1.mean(1)) == lab).astype(float)
    h1[y.mean(1) == y1.mean(1)] = np.nan
    tie = y == y1
    h2 = (((med > y1) == (y > y1)) & ~tie).sum(1)
    n2 = (~tie).sum(1)
    ql = M.QL.numpy()[LV9][None, :, None]
    e = y[:, None] - q[:, LV9]
    qs = np.maximum(ql * e, (ql - 1) * e).mean((1, 2))
    if "ref" not in CACHE:
        r = naive_epf(D, idx); er = y[:, None] - r[:, LV9]
        CACHE["ref"] = np.maximum(ql * er, (ql - 1) * er).mean((1, 2))
    return h1, h2, n2, qs, CACHE["ref"]


CACHE = {}


def S_of(c, sel):
    h1, h2, n2, qs, qr = (x[sel] for x in c)
    return 0.35 * np.nanmean(h1) + 0.35 * h2.sum() / n2.sum() + 0.30 * (1 - qs.mean() / qr.mean())


def boot(idx, qa, qb=None, B=2000, blk=7, seed=0):
    g = np.random.default_rng(seed); n = len(idx); out = []
    ca = per_day(idx, qa); cb = per_day(idx, qb) if qb is not None else None
    for _ in range(B):
        st = g.integers(0, n, n // blk + 1)
        sel = (st[:, None] + np.arange(blk)[None]).ravel()[:n] % n
        out.append(S_of(ca, sel) - (S_of(cb, sel) if cb is not None else 0))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


if __name__ == "__main__":
    a = sys.argv[1].split(","); qa, idx = ens(a)
    res = {"wm": a, "wm_score": score(D, idx, qa), "wm_S_90ci": boot(idx, qa, B=2000)}
    if len(sys.argv) > 2:
        b = sys.argv[2].split(","); qb, _ = ens(b)
        res.update(twin=b, twin_score=score(D, idx, qb), diff_S_90ci=boot(idx, qa, qb, B=2000))
    print(json.dumps(res, indent=1))
