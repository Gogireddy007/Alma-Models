"""Red-team randomized-future perturbation of the data lead's Store.asof (tools/live/store.py).

For each sampled decision time t = common.decision_time(day):
  P1 every record with published_at > t gets a random value, a random-but-later published_at shuffle, and
     extra FUTURE records are inserted (a 'revised' vintage of the latest visible obs of every series, published
     t + 1 s .. t + 30 d; brand-new obs rows; a fake split published after t);
  -> ctx_hash, features and every ctx frame must be bit-identical.
  P2 (live=True) every record with fetched_at > t randomised (published_at untouched) -> live ctx identical.
  Positive controls: a record published 1 s before t inserted (new storage revision, new px bar) -> must change.
Days: random DEV/VAL days + storage release days (Thu/holiday) + DST switch days + COT Fridays + roll days.

    nice -n 10 .venv/bin/python tools/live/redteam/rt_perturb_store.py [n_random]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import store as ST  # noqa: E402


def frames():
    out = {}
    for s in ST.SOURCES:
        p = os.path.join(C.PIT, f"{s}.csv")
        if os.path.exists(p):
            out[s] = pd.read_csv(p, dtype={"series": str, "obs": str, "vintage": str, "extra": str},
                                 keep_default_na=False, na_values={"value": ["", "nan", "NaN"]})
            out[s]["value"] = out[s]["value"].astype(float)   # same dtype as perturbed copies
    return out


def perturb(F, t, rng, live=False):
    G = {}
    tn = pd.Timestamp(t)
    for s, df in F.items():
        df = df.copy()
        df["value"] = df["value"].astype(float)
        pa = pd.to_datetime(df["published_at"], utc=True)
        fa = pd.to_datetime(df["fetched_at"], utc=True)
        m = (fa > tn) if live else (pa > tn)
        df.loc[m, "value"] = rng.normal(0, 1000, int(m.sum()))
        if not live:
            # shuffle future published_at among future rows (still > t)
            if m.sum() > 1:
                df.loc[m, "published_at"] = rng.permutation(df.loc[m, "published_at"].to_numpy())
            # inserted future records: revised vintage of visible obs + new obs + fake split
            vis = df[~m]
            if len(vis):
                last = vis.drop_duplicates("series", keep="last").copy()
                k = len(last)
                last["value"] = rng.normal(0, 1000, k)
                last["published_at"] = [C.iso(tn + pd.Timedelta(seconds=int(x))) for x in rng.integers(1, 30 * 86400, k)]
                last["fetched_at"] = C.iso(tn + pd.Timedelta(days=31))
                last["vintage"] = "revised"
                new = last.copy()
                new["obs"] = (tn + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                df = pd.concat([df, last, new], ignore_index=True)
            if s == "px":
                df = pd.concat([df, pd.DataFrame([{"source": "px", "series": "UNG.split",
                                                   "obs": (tn + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                                                   "value": 0.25, "published_at": C.iso(tn + pd.Timedelta(days=2)),
                                                   "fetched_at": C.iso(tn + pd.Timedelta(days=3)),
                                                   "vintage": "snapshot", "extra": ""}])], ignore_index=True)
        G[s] = df
    return G


def frame_diff(c1, c2):
    out = []
    for k in c1:
        a, b = c1[k], c2.get(k)
        if isinstance(a, pd.DataFrame):
            if not a.astype(str).equals(b.astype(str)):
                out.append(k)
        elif k == "px":
            out += [f"px.{s}" for s in set(a) | set(b) if s not in a or s not in b or not a[s].equals(b[s])]
    return out


try:
    import wm_store as WMS  # architect feature layer (pure function of ctx) -- included in the comparison
except Exception:  # noqa: BLE001
    WMS = None


def snapshot(ctx):
    feats = ctx["features"]
    f = {k: (None if not np.isfinite(v) else v) for k, v in feats.items()}
    if WMS is not None and hasattr(WMS, "features_from_ctx"):
        fu = WMS.features_from_ctx(ctx)
        f["__wm_x"] = None if fu is None else fu[0].tobytes()
        f["__wm_u"] = None if fu is None else fu[1].tobytes()
        f["__next_storage"] = str(ctx.get("next_storage"))
    return ctx["meta"]["input_sha256"], f


def days_sample(n, rng):
    td = [d for d in C.trading_days("2016-01-01", "2025-10-31")]
    pick = list(rng.choice(td, n, replace=False))
    extra = [pd.Timestamp(x) for x in ("2025-12-29", "2025-12-31", "2025-11-14", "2025-11-26", "2026-04-23",
                                       "2026-03-09", "2026-03-06", "2025-11-03", "2025-10-31", "2024-11-04",
                                       "2024-03-11", "2019-02-08", "2026-01-30", "2026-01-29", "2025-11-21",
                                       "2024-06-25", "2024-06-26", "2024-06-27", "2026-10-28", "2026-10-29")]
    thursdays = [d for d in td if d.weekday() == 3]
    pick += list(rng.choice(thursdays, 8, replace=False))
    return [pd.Timestamp(d) for d in pick] + [d for d in extra if C.is_bday(d)]


def main(n=12, seed=7):
    rng = np.random.default_rng(seed)
    F = frames()
    S = ST.Store(frames=F)
    bad, ok = [], 0
    for day in days_sample(n, rng):
        t = C.decision_time(day)
        ca, cb = S.asof(t), ST.Store(frames=perturb(F, t, rng)).asof(t)
        a, b = snapshot(ca), snapshot(cb)
        if a[0] != b[0] or a[1] != b[1]:
            diff = [k for k in a[1] if a[1][k] != b[1][k]]
            bad.append(("P1", str(day.date()), a[0] == b[0], diff, frame_diff(ca, cb)))
        else:
            ok += 1
    print(f"P1 published_at>t randomised+inserted: {ok}/{ok + len(bad)} identical")
    for x in bad:
        print("  CHANGED", x)
    # P2 live
    ok2, bad2 = 0, []
    for day in [pd.Timestamp("2026-10-29"), pd.Timestamp("2026-10-30")]:
        t = C.decision_time(day)
        a = snapshot(S.asof(t, live=True))
        b = snapshot(ST.Store(frames=perturb(F, t, rng, live=True)).asof(t, live=True))
        (bad2.append(str(day.date())) if a != b else None) or (ok2 := ok2 + (a == b))
    print(f"P2 live fetched_at>t randomised: {ok2}/2 identical {bad2}")
    # positive controls
    day = pd.Timestamp("2024-03-14")
    t = C.decision_time(day)
    a = snapshot(S.asof(t))
    G = {k: v.copy() for k, v in F.items()}
    r = G["storage"][G["storage"].series == "l48"].iloc[[-1]].copy()
    ctx = S.asof(t)
    w = ctx["storage"].index[-1]
    r["obs"], r["value"], r["vintage"] = w.strftime("%Y-%m-%d"), 9999.0, "revised"
    r["published_at"] = C.iso(t - pd.Timedelta(seconds=1))
    G["storage"] = pd.concat([G["storage"], r], ignore_index=True)
    b = snapshot(ST.Store(frames=G).asof(t))
    print(f"positive control (revision published t-1s) changes ctx: {a[0] != b[0]}; features: {a[1] != b[1]}")
    return bad, bad2


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
