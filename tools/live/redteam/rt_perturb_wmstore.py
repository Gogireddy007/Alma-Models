"""Red-team randomized-future perturbation of the architect's store-based feature layer tools/live/wm_store.py.

wm_store reads data/live/pit directly (not via Store.asof), so it gets its own test. For each sampled trading day d
(t = common.decision_time(d)): every row with published_at > t in every series is randomised AND future rows are
inserted (older obs re-published after t: a fake later 'first print' / later close / later split / later CPC issue
for days in the forecast window / later observed day); features_at(d) must be bit-identical. Positive controls:
a row published 1 s before t (new storage print, new UNG close) must change the features.

    nice -n 10 .venv/bin/python tools/live/redteam/rt_perturb_wmstore.py [n]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as CM  # noqa: E402
import wm_store as WS  # noqa: E402

SER = ["UNG", "NG=F", "l48", "chg", "cot_mm_long", "cot_mm_short", "cot_oi", "UNG.split", "NG=F.split"]


def _ins(s, obs, v, pa):
    o = np.concatenate([s["obs"], np.array(obs, dtype="datetime64[D]")])
    vv = np.concatenate([s["v"], np.array(v, float)])
    p = np.concatenate([s["pa"], np.array(pa, dtype="datetime64[ns]")])
    i = np.lexsort((o, p))
    return {"obs": o[i], "v": vv[i], "pa": p[i]}


def perturb(R, d, rng):
    t = R["dt"][np.searchsorted(R["days"], d)]
    Q = dict(R)
    late = t + np.timedelta64(1, "s")
    for k in SER:
        s = dict(R[k])
        s["v"] = s["v"].copy()
        m = s["pa"] > t
        s["v"][m] = rng.uniform(0.1, 5000, m.sum())
        if len(s["obs"]):
            past = s["obs"][s["pa"] <= t]
            if len(past):
                pick = past[-3:]
                s = _ins(s, pick, rng.uniform(0.1, 5000, len(pick)), [late] * len(pick))
        Q[k] = s
    for key, dcol in (("fc", "day"), ("ob", "date")):
        s = {c: v.copy() for c, v in R[key].items()}
        m = s["pa"] > t
        for c in ("hdd", "cdd"):
            s[c] = s[c].astype(float)
            s[c][m] = rng.uniform(0, 60, m.sum())
        # insert: a forecast issue dated d (issue == d) published after t, and an observed value for d-1 after t
        n = 7
        add = {"hdd": rng.uniform(0, 60, n), "cdd": rng.uniform(0, 30, n), "pa": np.array([late] * n)}
        if key == "fc":
            add["issue"] = np.array([d] * n, dtype="datetime64[D]")
            add["day"] = np.array([d + np.timedelta64(j, "D") for j in range(1, n + 1)])
        else:
            add["date"] = np.array([d - np.timedelta64(j, "D") for j in range(1, n + 1)])
        s = {c: np.concatenate([s[c], add[c].astype(s[c].dtype)]) for c in s}
        Q[key] = s
    return Q


def main(n=40, seed=3):
    R = WS.load_raw()
    rng = np.random.default_rng(seed)
    P = WS.load_panel()
    D = P["D"]
    pick = list(rng.choice(D[D <= np.datetime64("2025-10-31")], n, replace=False))
    for x in ("2025-12-29", "2025-12-31", "2026-04-23", "2026-03-09", "2024-11-04", "2019-02-08", "2026-01-30",
              "2026-01-29", "2024-01-24", "2024-01-25", "2018-01-05", "2026-10-22", "2026-10-23"):
        x = np.datetime64(x)
        if (D == x).any():
            pick.append(x)
    bad = []
    for d in pick:
        a = WS.features_at(d, R)
        WS._NC.clear()
        b = WS.features_at(d, perturb(R, d, rng), nkey="rt")
        WS._NC.clear()
        if a is None or b is None or not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
            bad.append((str(d), None if a is None or b is None else
                        [WS.FEATS[i] for i in np.where(a[0] != b[0])[0]] +
                        sorted(set(WS.UFEATS[i] for i in np.where(a[1] != b[1])[1]))))
    print(f"rt_perturb_wmstore: {len(pick) - len(bad)}/{len(pick)} identical (randomised + inserted future rows)")
    for x in bad:
        print("  CHANGED", x)
    d = np.datetime64("2023-03-16")
    t = R["dt"][np.searchsorted(R["days"], d)]
    a = WS.features_at(d, R)
    Q = dict(R)
    s = R["chg"]
    Q["chg"] = _ins(s, [s["obs"][s["pa"] <= t][-1] + np.timedelta64(7, "D")], [999.0], [t - np.timedelta64(1, "s")])
    Q["l48"] = _ins(R["l48"], [s["obs"][s["pa"] <= t][-1] + np.timedelta64(7, "D")], [3000.0], [t - np.timedelta64(1, "s")])
    b = WS.features_at(d, Q, nkey="pc")
    print("positive control (storage print at t-1s) changes:", not np.array_equal(a[0], b[0]))
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
