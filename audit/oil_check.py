"""Independent re-score of the frozen oil model (wm_sv) on the TEST year.

Own data slicing (raw FRED csv), own pinball/QS, own random-walk + GARCH-free EWMA baselines, own leak test:
every value at or after the origin is replaced by noise and the forecast must not change.
Only the model wrapper (tools/commodity/wm_commodity.py) is reused, because it is the model.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "commodity"))
import wm_commodity as W  # noqa: E402

QS9 = np.round(np.arange(0.1, 0.91, 0.1), 2)
CTX = 1024


def load_wti():
    df = pd.read_csv(os.path.join(REPO, "data/commodity/raw/DCOILWTICO.csv"))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    return pd.to_datetime(df["date"]).to_numpy(), df["v"].to_numpy(float)


def pinball(y, q):
    d = y[:, None] - q
    return np.mean(np.maximum(QS9 * d, (QS9 - 1) * d), 1)


def rw(ctx, h, win=250, ewma=None):
    lp = np.log(ctx[ctx > 0])
    r = np.diff(lp)
    if ewma:
        w = ewma ** np.arange(len(r))[::-1]
        s = np.sqrt(np.sum(w * r ** 2) / w.sum())
    else:
        s = np.std(r[-win:], ddof=1)
    return lp[-1] + norm.ppf(QS9) * s * np.sqrt(h)


def main():
    d, v = load_wti()
    test = np.where(d >= np.datetime64("2025-11-01"))[0]
    models = {n: W.make_predict_ens([(os.path.join(REPO, p), s) for p, s in m]) for n, m in W.FROZEN.items()}
    rows = []
    for h in (1, 5, 20):
        oi = test[test + h - 1 < len(v)]                     # origin i: context v[:i], target v[i+h-1]
        y = np.log(v[oi + h - 1])
        res = {}
        for n, p in models.items():
            q = np.log(p.batch([v[max(0, i - CTX):i] for i in oi], h, QS9)[:, :, h - 1])
            res[n] = pinball(y, q)
        res["rw250"] = pinball(y, np.stack([rw(v[:i], h) for i in oi]))
        res["ewma94"] = pinball(y, np.stack([rw(v[:i], h, ewma=0.94) for i in oi]))
        for n, l in res.items():
            rows.append((h, n, len(oi), l.mean(), 1 - l.mean() / res["rw250"].mean()))
    out = pd.DataFrame(rows, columns=["h", "model", "n", "QS", "skill_vs_rw"])
    print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # leak test: scramble everything at/after the origin; forecast must be identical
    rng = np.random.default_rng(0)
    p = models["wm_sv"]
    bad = 0
    for i in test[::10]:
        a = p.batch([v[max(0, i - CTX):i]], 20, QS9)
        vv = v.copy()
        vv[i:] = rng.uniform(5, 500, len(vv) - i)
        b = p.batch([vv[max(0, i - CTX):i]], 20, QS9)
        bad += not np.allclose(a, b, equal_nan=True)
    print(f"leak test (future scrambled): {bad} of {len(test[::10])} origins changed")
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oil_check.csv"), index=False)


if __name__ == "__main__":
    main()
