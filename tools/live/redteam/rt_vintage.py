"""Red-team independent EIA storage vintage reference (no data-lead code).

Builds, from the raw EIA files only, the Lower-48 level that a live trader could have known at time t for each week:
  * first print (revisions.xls original_data, every week 2015-06-19 .. 2026-04-10; later weeks: current = first print
    as no revision is recorded) published at eia_release_utc(week);
  * revision published at the release of the NEXT week's report (EIA practice: the explanation row is the revised
    week; the 2026-04-10 row announces the 33-week resubmission 2025-08-29 .. 2026-04-10, published with the report
    for the week ending 2026-04-17).
  Intermediate vintages (e.g. 2026-03-27: 1865 -> 1861 -> 1851) are not in the files; we keep first print until the
  big resubmission and flag the gap.

  asof_level(week, t)    -> level known at t (NaN if not yet released)
  reported_change(week)  -> net change as printed on the release day = FP(week) - level(week-1) known at that release
                            (differs from FP(W) - FP(W-1) when W-1 was revised in the same report)
Run: python tools/live/redteam/rt_vintage.py  (prints the exposed-window weeks whose current value != first print)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rt_ref as R  # noqa: E402

REV = os.path.join(R.ROOT, "data/gas/raw/ngs_revisions.xls")
HIST = os.path.join(R.ROOT, "data/gas/storage.csv")   # used ONLY for the current-vintage column (l48_current)
RESUB_FIRST, RESUB_LAST = pd.Timestamp("2025-08-29"), pd.Timestamp("2026-04-10")


def table():
    import warnings
    warnings.simplefilter("ignore")
    x = pd.read_excel(REV, sheet_name="original_data", header=1)
    x["week"] = pd.to_datetime(x["Week ending"], errors="coerce")
    x = x.dropna(subset=["week"]).set_index("week")
    cur = pd.read_csv(HIST, parse_dates=["week"]).set_index("week")["l48_current"]
    t = pd.DataFrame({"current": cur})
    t["first"] = x["Total Lower 48"].astype(float).reindex(t.index)
    t["first"] = t["first"].fillna(t["current"].where(t.index > x.index.max()))
    t["release"] = [R.eia_release_utc(w)[0] for w in t.index]
    t["release_rule"] = [R.eia_release_utc(w)[1] for w in t.index]
    # revision publication: release of the following week (big resubmission: release of week 2026-04-17)
    nxt = t["release"].shift(-1)
    t["revised_at"] = np.where(t["first"] != t["current"], nxt, pd.NaT)
    m = (t.index >= RESUB_FIRST) & (t.index <= RESUB_LAST) & (t["first"] != t["current"])
    t.loc[m, "revised_at"] = R.eia_release_utc(pd.Timestamp("2026-04-17"))[0]
    t["revised_at"] = pd.to_datetime(t["revised_at"], utc=True)
    return t


def asof_level(t_tab, week, t):
    r = t_tab.loc[pd.Timestamp(week)]
    t = pd.Timestamp(t)
    if t < r.release:
        return np.nan
    if pd.notna(r.revised_at) and t >= r.revised_at:
        return r.current
    return r["first"]


def reported_change(t_tab, week):
    w = pd.Timestamp(week)
    rel = t_tab.loc[w, "release"]
    return t_tab.loc[w, "first"] - asof_level(t_tab, w - pd.Timedelta(days=7), rel)


if __name__ == "__main__":
    T = table()
    ex = T[(T.index >= "2025-08-01") & (T["first"] != T["current"])]
    print(f"weeks with current != first print since 2025-08: {len(ex)}; mean diff {(ex.current - ex['first']).mean():.1f}")
    print(ex[["first", "current", "release", "revised_at"]].to_string())
    rc = pd.Series({w: reported_change(T, w) for w in T.index[T.index >= "2015-07-01"]})
    naive = T["first"].diff().reindex(rc.index)
    bad = rc[(rc - naive).abs() > 0]
    print(f"weeks where printed net change != FP(W)-FP(W-1): {len(bad)}  e.g.",
          {str(k.date()): (float(rc[k]), float(naive[k])) for k in bad.index[-3:]})
    print("release rules:", T["release_rule"].value_counts().to_dict())
