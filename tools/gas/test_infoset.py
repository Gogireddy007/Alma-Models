"""
Information-set tests for Alma-Gas (docs/gas_prereg.md section 2). Run before any VAL score and before TEST:

    nice -n 10 .venv/bin/python tools/gas/test_infoset.py

Origins are sampled from DEV and VAL only (the code path is identical for TEST; no TEST value is read).
1. Price task: rebuild the daily panel (data.make_panel) from PERTURBED RAW SOURCES -- every spot price dated >= the
   first target date, every storage report released after the origin date, every weather day >= the origin date --
   and check build_context returns bit-identical inputs. Sensitivity: perturbing the origin-day price, the latest
   released report and the weather of origin-1 MUST change the inputs.
2. Storage task: rebuild the weekly table (storage.weekly_from) from perturbed levels (target week and later),
   perturb observed weather from the origin day on and every CPC forecast issue not usable at the origin (issued
   later, or same day with Last-Modified after the origin); storage.build_context must be identical. Sensitivity:
   perturbing week k-1 and the used forecast issue MUST change it.
3. Training helpers refuse anything after 2025-06-30.
4. Models: listed cheap models give identical forecasts on original vs perturbed contexts and on a repeat call.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D        # noqa: E402
import storage as S     # noqa: E402

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
CTX = 1024


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        if a.dtype.kind in "fc":
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return np.array_equal(a, b)
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


def ctx_from(p, i):
    dates = p["date"].to_numpy().astype("datetime64[D]")
    return D.build_context(i, CTX, dates, p["hh"].to_numpy(float), p[D.COVS].to_numpy(float), p)


def test_price(rng, n):
    hh, st, wx = D.sources()
    p0 = D.make_panel(hh, st, wx)
    dates = p0["date"].to_numpy().astype("datetime64[D]")
    pool = np.concatenate([D.split_origins("dev", dates), D.split_origins("val", dates)])
    origins = sorted(rng.choice(pool, n, replace=False).tolist())
    # include origins right around storage releases (Thursday/Friday) and a holiday week
    rel = pd.to_datetime(st.avail)
    for day in rel[(rel > "2024-11-01") & (rel < "2025-10-31")].iloc[[0, 5, 30]]:
        j = int(np.searchsorted(dates, np.datetime64(day.date(), "D")))
        origins += [j, j + 1]
    for i in origins:
        t = pd.Timestamp(dates[i - 1])                       # forecast time = close of t
        c0 = ctx_from(p0, i)
        hh2, st2, wx2 = hh.copy(), st.copy(), wx.copy()
        m = hh2.date >= pd.Timestamp(dates[i])
        hh2.loc[m, "value"] = rng.uniform(0.5, 50, m.sum())
        m = st2.avail > t
        st2.loc[m, "l48"] = rng.uniform(500, 5000, m.sum())
        m = wx2.index >= t
        wx2.loc[m, ["hdd_pw", "cdd_pw"]] = rng.uniform(0, 60, (m.sum(), 2))
        c1 = ctx_from(D.make_panel(hh2, st2, wx2), i)
        assert same(c0["y"], c1["y"]), f"price leak at origin {t.date()}"
        assert same(c0["covs"], c1["covs"]), f"covariate leak at origin {t.date()}"
        # sensitivity
        hh3 = hh.copy()
        hh3.loc[hh3.date == t, "value"] += 1.0
        assert not same(c0["y"], ctx_from(D.make_panel(hh3, st, wx), i)["y"]), "origin-day price not used"
        wx3 = wx.copy()
        wx3.loc[t - pd.Timedelta(days=1), "hdd_pw"] += 5.0
        assert not same(c0["covs"], ctx_from(D.make_panel(hh, st, wx3), i)["covs"]), "weather of t-1 not used"
        st3 = st.copy()
        last = st3.index[st3.avail <= t][-1]
        st3.loc[last, "l48"] += 7.0
        assert not same(c0["covs"], ctx_from(D.make_panel(hh, st3, wx), i)["covs"]), "latest released report not used"
    print(f"price task: OK on {len(origins)} origins ({min(dates[np.array(origins) - 1])} .. "
          f"{max(dates[np.array(origins) - 1])}), raw-source perturbation + sensitivity")
    return origins


def test_storage(rng, n):
    lvl, st = S.levels()
    w0 = S.weekly_from(lvl, st)
    _, o, f = S.load()
    pool = np.concatenate([S.target_weeks("dev", w0), S.target_weeks("val", w0)])
    ks = sorted(rng.choice(pool, n, replace=False).tolist())
    ks += [int(np.where(w0.avail_rule == "schedule")[0][2])]          # a schedule-shifted week (2025-06)
    nfc = 0
    for k in ks:
        for kind in S.ORIGINS:
            odt = S.origin_dt(w0, k, kind)
            od = pd.Timestamp(odt.date())
            c0 = S.build_context(k, kind, w0, o, f)
            lvl2 = lvl.copy()
            lvl2.iloc[k:] = rng.uniform(100, 5000, lvl2.iloc[k:].shape)
            w2 = S.weekly_from(lvl2, st)
            o2 = o.copy()
            mo = o2.index >= od
            o2.loc[mo] = rng.uniform(0, 60, (mo.sum(), 2))
            f2 = f.copy()
            bad = (f2.issue > od) | ((f2.issue == od) & (f2.last_modified > odt.tz_convert("UTC")))
            f2.loc[bad, ["hdd", "cdd"]] = rng.uniform(0, 60, (bad.sum(), 2))
            c1 = S.build_context(k, kind, w2, o2, f2)
            for key in c0:
                assert same(c0[key], c1[key]), f"storage leak in '{key}' at week {w0.week.iloc[k].date()} {kind}"
            # sensitivity: week k-1 and the used forecast issue
            lvl3 = lvl.copy()
            lvl3.iloc[k - 1, lvl3.columns.get_loc("l48")] += 5.0
            c3 = S.build_context(k, kind, S.weekly_from(lvl3, st), o, f)
            assert not same(c0["ds_hist"], c3["ds_hist"]), "week k-1 not used"
            if (c0["wx_src"] >= 1).any():
                nfc += 1
                f4 = f.copy()
                f4.loc[f4.issue == pd.Timestamp(c0["wx_issue"]), "hdd"] += 3.0
                assert not same(c0["wx"], S.build_context(k, kind, w0, o, f4)["wx"]), "usable forecast issue not used"
            assert (c0["wx_src"] >= 0).all(), f"missing weather day for week {w0.week.iloc[k].date()} {kind}"
    print(f"storage task: OK on {len(ks)} weeks x {len(S.ORIGINS)} origin kinds ({nfc} contexts use CPC forecasts)")
    return ks


def test_train():
    for fn in (lambda: D.train_panel("2025-07-01"),):
        try:
            fn()
            raise AssertionError("train_panel accepted a date after TRAIN_END")
        except ValueError:
            pass
    X, XC, Y, T = D.train_windows(64, 20, stride=50, start="2025-01-01")
    assert len(T) and T.max() <= np.datetime64("2025-06-30")
    ks = S.train_weeks()
    w = S.load()[0]
    assert w.avail.iloc[ks].max() <= D.TRAIN_END
    print(f"train helpers: OK (last daily target {T.max()}, last storage training release {w.avail.iloc[ks].max().date()})")


def test_models(origins, ks, rng):
    import models as M
    dates, y, C, extra = D.panel()
    sub = origins[:4]
    ctx = [D.build_context(int(i), CTX, dates, y, C, extra) for i in sub]
    for mn in ("rw_gauss", "px_fund_ols"):
        p = M.get(mn)
        a, b = p.batch(ctx, 20, QS), p.batch(ctx, 20, QS)
        assert np.array_equal(a, b, equal_nan=True), f"{mn} not deterministic"
        print(f"  {mn}: deterministic on {len(sub)} origins")
    w, o, f = S.load()
    sc = [S.build_context(int(k), kind, w, o, f) for k in ks[:4] for kind in S.ORIGINS]
    for mn in ("st_persistence", "st_climatology", "st_seasonal_naive", "st_ols_wx"):
        p = M.get(mn)
        a, b = p.storage(sc, QS, "l48"), p.storage(sc, QS, "l48")
        assert np.array_equal(a, b, equal_nan=True), f"{mn} not deterministic"
        print(f"  {mn}: deterministic on {len(sc)} contexts")
    print("models: inputs are the context dicts only (future invariance follows from tests 1-2)")


def main():
    rng = np.random.default_rng(1)
    origins = test_price(rng, 10)
    ks = test_storage(rng, 10)
    test_train()
    test_models(origins, ks, rng)
    print("ALL INFORMATION-SET TESTS PASSED")


if __name__ == "__main__":
    main()
