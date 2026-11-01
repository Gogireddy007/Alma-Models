"""Red-team published_at audit of data/live/pit/*.csv against independent references (rt_ref, rt_vintage).
A record EARLIER than the reference is a candidate leak (counted); later = conservative (fine).

    nice -n 10 .venv/bin/python tools/live/redteam/rt_pubaudit.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rt_ref as R  # noqa: E402
import rt_vintage as V  # noqa: E402

PIT = os.path.join(R.ROOT, "data/live/pit")
# CFTC post-shutdown catch-up (last backlog release); Tuesdays from shutdown start up to ~2 weeks before catch-up
COT_BACKLOG = [("2013-10-01", "2013-11-22"), ("2018-12-22", "2019-03-08"), ("2025-10-01", "2026-01-30")]


def load(s):
    p = os.path.join(PIT, f"{s}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["pa"] = pd.to_datetime(df["published_at"], utc=True)
    df["fa"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df


def report(name, early, n, ex=None):
    print(f"[{name}] rows {n}: EARLIER than reference {int(early.sum())}")
    if ex is not None and early.any():
        print(ex[early].head(8).to_string())


def audit_storage():
    df = load("storage")
    if df is None:
        return
    print("[storage] fa<pa rows:", int((df.fa < df.pa).sum()))
    fp = df[df.vintage.isin(["first_release", "current_backfill"]) & ~df.series.str.endswith("chg_printed")].copy()
    fp["ref"] = [R.eia_release_utc(pd.Timestamp(w))[0] for w in fp.obs]
    report("storage first print vs eia_release_utc", fp.pa < fp.ref, len(fp), fp[["series", "obs", "pa", "ref"]])
    T = V.table()
    l48 = df[df.series == "l48"]
    # value check: first print equals rt_vintage first; revised equals current with revision time >= reference
    f = l48[l48.vintage == "first_release"].set_index(pd.to_datetime(l48[l48.vintage == "first_release"].obs))
    j = f.join(T[["first", "current", "revised_at"]], how="inner")
    print("[storage l48 first_release] value != rt_vintage first:", int((j.value.astype(float) != j["first"]).sum()),
          "of", len(j))
    rv = l48[l48.vintage == "revised"]
    rv = rv.set_index(pd.to_datetime(rv.obs)).join(T[["current", "revised_at"]], how="left")
    bad = rv.pa < rv.revised_at
    print(f"[storage l48 revised] rows {len(rv)}; value != current {int((rv.value.astype(float) != rv.current).sum())};"
          f" published earlier than rt_vintage revised_at {int(bad.sum())}")
    if bad.any():
        print(rv[bad][["value", "pa", "revised_at"]].head(8).to_string())
    # weeks where rt_vintage has a revision but the store has none (would leave first print forever: conservative)
    miss = set(T.index[T.revised_at.notna()]) - set(rv.index)
    print("[storage l48] revisions in rt_vintage missing from store:", len(miss))
    # store as-of for the resubmission window: pick t in (2025-11-03, 2026-04-22]
    ch = df[df.series == "l48.chg_printed"].set_index(pd.to_datetime(df[df.series == "l48.chg_printed"].obs))
    rc = pd.Series({w: V.reported_change(T, w) for w in ch.index if w in T.index and w >= pd.Timestamp("2015-07-01")})
    d = (ch.value.astype(float).reindex(rc.index) - rc).abs()
    print("[storage l48.chg_printed] vs rt_vintage reported_change: mismatches", int((d > 1e-9).sum()), "of", len(rc),
          d[d > 1e-9].head(5).to_dict())


def audit_cot():
    df = load("cot")
    if df is None:
        return
    df["tue"] = pd.to_datetime(df.obs)
    ref = {}
    for tue in df.tue.unique():
        r = R.cot_release_utc(tue)
        for a, c in COT_BACKLOG:
            if pd.Timestamp(a) <= tue <= pd.Timestamp(c) - pd.Timedelta(days=14):
                r = max(r, R._et(pd.Timestamp(c), 15, 30))
        ref[tue] = r
    df["ref"] = df.tue.map(ref)
    early = df.pa < df.ref
    g = df[early].groupby("obs").agg(pa=("pa", "first"), ref=("ref", "first"))
    print(f"[cot] rows {len(df)}: EARLIER than reference {int(early.sum())} ({len(g)} report dates)")
    if len(g):
        g["days_early"] = (g.ref - g.pa).dt.total_seconds() / 86400
        print(g.groupby(pd.to_datetime(g.index).year).days_early.agg(["count", "max"]).to_string())


def audit_px():
    df = load("px")
    if df is None:
        return
    df["ref"] = [R._et(pd.Timestamp(o), 16, 0) for o in df.obs] if len(df) < 0 else pd.NaT
    obs = pd.to_datetime(df.obs.unique())
    m = {o.strftime("%Y-%m-%d"): R._et(o, 16, 0) for o in obs}       # ETF close; NG Globex closes 17:00 ET
    df["ref"] = df.obs.map(m)
    bars = ~df.series.str.endswith(".split")
    early = bars & (df.pa < df.ref)
    print(f"[px] bar rows {int(bars.sum())}: published before the 16:00 ET close of the session {int(early.sum())};"
          f" fa<pa {int((df.fa < df.pa).sum())}")
    ng = df[df.series.str.startswith("NG=F.") & bars]
    print("[px] NG=F bars published before 17:00 ET (Globex close):",
          int((ng.pa < ng.obs.map(lambda o: R._et(pd.Timestamp(o), 17, 0))).sum()))


def audit_edgar():
    for s in ("edgar_fact", "edgar_filing"):
        df = load(s)
        if df is None:
            continue
        per = pd.to_datetime(df.obs, errors="coerce").dt.tz_localize("UTC")
        same = (df.pa - per).dt.total_seconds().abs() < 86400 * 2
        print(f"[{s}] rows {len(df)}; published within 2 days of obs (period end/filing day) {int(same.sum())};"
              f" published before obs {int((df.pa < per).sum())}")


if __name__ == "__main__":
    audit_storage()
    audit_cot()
    audit_px()
    audit_edgar()
