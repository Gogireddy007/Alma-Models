"""Red-team perturbation test of the architect's interim adapter tools/live/wm_data.py, using the RED-TEAM's own
availability rules (not the adapter's), so a wrong availability key in the adapter shows up as a change.

Availability used here (conservative, from rt_ref):
  price HH spot of trade date d: d (16:00 ET), i.e. accepted adapter convention (see LRT-2 for the live caveat)
  storage week W: rt_ref.eia_release_utc(W) (DST-correct, holiday-name paired) must be <= t 16:00 ET
  CPC observed day d: d+1 (end of day) -> only days <= t-1
  CPC forecast issue: HTTP Last-Modified <= t 16:00 ET (issue date alone NOT trusted)
Every row not available at t is replaced by random values; rows are also appended beyond the end (inserted
future). features_at(j) must be bit-identical. Positive controls: last available report / price of t perturbed.

    nice -n 10 .venv/bin/python tools/live/redteam/rt_perturb_wm.py [n]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rt_ref as RR  # noqa: E402
import wm_data as W  # noqa: E402


def avail_tables(R):
    rel = np.array([np.datetime64(RR.eia_release_utc(pd.Timestamp(w))[0].tz_localize(None), "s")
                    for w in R["s_week"]])
    return {"s_rel": rel}


def perturb(R, A, t, rng):
    tc = W.close_utc(t)
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    # HH spot: red-team availability = next Wednesday 17:00 ET after the trade date's week (rt_ref), stricter of the
    # two keys (adapter's p_avail if present)
    pav = np.array([np.datetime64(RR.hh_spot_eia_avail_utc(pd.Timestamp(d)).tz_localize(None), "s")
                    for d in Q["p_date"]]) if "p_av_rt" not in A else A["p_av_rt"]
    A["p_av_rt"] = pav
    m = (pav > tc + np.timedelta64(0, "s")) & (Q["p_date"] > t - np.timedelta64(14, "D"))
    Q["p"][m | (Q["p_date"] > t)] = rng.uniform(0.5, 20, (m | (Q["p_date"] > t)).sum())
    m = A["s_rel"] > tc
    for k in ["s_lvl", "s_ds", "s_avg5", "s_dev"]:
        Q[k][m] = rng.normal(0, 300, m.sum())
    m = Q["o_date"] >= t
    Q["o_hdd"][m] = rng.uniform(0, 60, m.sum())
    Q["o_cdd"][m] = rng.uniform(0, 30, m.sum())
    m = Q["f_lm"] > tc
    Q["f_hdd"][m] = rng.uniform(0, 60, m.sum())
    Q["f_cdd"][m] = rng.uniform(0, 30, m.sum())
    return Q, int((A["s_rel"] > tc).sum() - (R["s_avail"] > t).sum()), int((m & (R["f_issue"] < t)).sum())


def main(n=60, seed=1):
    R = W.load_raw()
    A = avail_tables(R)
    rng = np.random.default_rng(seed)
    js = np.where((R["p_date"] >= np.datetime64("2015-06-01")) & (R["p_date"] <= np.datetime64("2024-06-30")))[0]
    # targeted days: storage release days (+/-1), DST switch weeks, holiday weeks, late-LM CPC issue days
    rd = set(pd.DatetimeIndex(R["s_avail"]).normalize())
    tgt = [j for j in js if pd.Timestamp(R["p_date"][j]) in rd]
    late = np.unique(R["f_issue"][R["f_lm"] > np.array([W.close_utc(d + np.timedelta64(1, "D"))
                                                         for d in R["f_issue"]])])
    late_days = set(pd.DatetimeIndex(late + np.timedelta64(1, "D")))
    tgt_late = [j for j in js if pd.Timestamp(R["p_date"][j]) in late_days]
    sample = list(rng.choice(js, n, replace=False)) + list(rng.choice(tgt, min(20, len(tgt)), replace=False)) + \
        tgt_late[:20]
    bad = []
    for j in sample:
        t = R["p_date"][j]
        a = W.features_at(j, R)
        if a is None:
            bad.append((str(t), ["features None"], [], 0, 0))
            continue
        Q, extra_s, extra_f = perturb(R, A, t, rng)
        W._NCACHE.clear()
        b = W.features_at(j, Q, nkey="rt")
        W._NCACHE.clear()
        same = b is not None and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        if not same:
            dx = [W.FEATS[i] for i in np.where(a[0] != b[0])[0]] if b is not None else ["None"]
            du = sorted(set(W.UFEATS[i] for i in np.where(a[1] != b[1])[1])) if b is not None else []
            bad.append((str(t), dx, du, extra_s, extra_f))
    print(f"rt_perturb_wm: {len(sample) - len(bad)}/{len(sample)} identical "
          f"(random {n}, release days {min(20, len(tgt))}, late-LM CPC days {len(tgt_late[:20])})")
    for b in bad:
        print("  CHANGED", b)
    # positive controls
    j = js[len(js) // 2]
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    jl = np.searchsorted(R["p_avail"], R["p_date"][j], side="right") - 1 if "p_avail" in R else j
    Q["p"][jl] *= 1.1
    c1 = not np.array_equal(W.features_at(j, R)[0], W.features_at(j, Q)[0])
    t = R["p_date"][j]
    i = np.where(R["s_avail"] <= t)[0][-1]
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    Q["s_ds"][i] += 50
    c2 = not np.array_equal(W.features_at(j, R)[0], W.features_at(j, Q)[0])
    print(f"positive controls: price(t) changes {c1}; last released report changes {c2}")
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
