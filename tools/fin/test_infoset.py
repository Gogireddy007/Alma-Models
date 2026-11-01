"""
Information-set tests for Alma-Fin (docs/fin_prereg.md section 2). Run before any score and before TEST:

    nice -n 10 .venv/bin/python tools/fin/test_infoset.py [--models har,lgbm,tsmom_30] [--n 12]

1. Perturbation: for sampled decision days D (DEV and VAL; TEST rows are sealed), multiply every daily value of days
   >= D and every hourly value of hours starting >= D 00:00 by U(0.3, 3); build_context must be bit-identical.
2. Sensitivity: perturbing day D-1 (daily) and the hour 23:00-24:00 of D-1 MUST change the context.
3. Raw re-derivation: for sampled (coin, day) the fill = (O+H+L+C)/4 of the raw 00:05 5-min bar, the label RV =
   sum of squared 5-min log returns 00:05..23:55 from the raw zips, and the fill bar opens >= decision + 5 min.
4. P&L timing: the return of decision D uses fill(D) and fill(D+1) only (perturbing days >= D+2 leaves it unchanged).
5. Training view: train_view() stops at 2025-06-30 and holds only the 16 training coins.
6. Data-access guard: a model that opens a file under data/fin/ during predict raises PermissionError.
7. Models (each listed): forecasts on the original vs future-perturbed panel are identical and repeat calls are
   identical. Online runner: a toy continual model's decisions up to the perturbation day are bit-identical, its
   observe() never receives a row dated >= its next decision, and its update count is logged.
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H     # noqa: E402
import data as DT       # noqa: E402


def perturbed(P, i, rng, lo=0.3, hi=3.0):
    Q = H.Panel.__new__(H.Panel)
    Q.__dict__.update(P.__dict__)
    Q.daily = {f: a.copy() for f, a in P.daily.items()}
    for f in Q.daily:
        Q.daily[f][i:] *= rng.uniform(lo, hi, Q.daily[f][i:].shape)
    if P.hourly is not None:
        Q.hourly = {f: a.copy() for f, a in P.hourly.items()}
        for f in Q.hourly:
            Q.hourly[f][24 * i:] *= rng.uniform(lo, hi, Q.hourly[f][24 * i:].shape).astype(np.float32)
    return Q


def same(c0, c1):
    for f in c0["daily"]:
        if not np.array_equal(c0["daily"][f], c1["daily"][f], equal_nan=True):
            return False
    for f in c0.get("hourly", {}):
        if not np.array_equal(c0["hourly"][f], c1["hourly"][f], equal_nan=True):
            return False
    return True


class ToyOnline:
    """Continual toy forecaster: EWMA of log RV per asset, updated in observe()."""
    kind = "forecaster"

    def __init__(self):
        self.mu, self.seen = None, []

    def predict(self, ctx):
        if self.mu is None:
            self.mu = np.nan_to_num(np.nanmean(ctx["daily"]["logrv"][-30:], 0), nan=-8.0)
        self.last_decision = ctx["decision_time"]
        return {"q": self.mu[:, None] + np.array(H.QS)[None, :] - 0.5}

    def observe(self, rec):
        assert rec["day"] < self.last_decision + pd.Timedelta(days=1)
        self.seen.append(rec["day"])
        x = rec["row"]["logrv"]
        self.mu = np.where(np.isfinite(x), 0.9 * self.mu + 0.1 * x, self.mu)
        return 1


class Snooper:
    kind = "forecaster"

    def predict(self, ctx):
        open(os.path.join(DT.DATA, "crypto_daily.parquet"), "rb").read(10)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="majority,rv_persist,har,harx,har_roll,garch,lgbm,lgbm_asset,har_dow,lgbm_dow,har_intra_dow,buy_hold,tsmom_1,tsmom_7,tsmom_30,meanrev_1,voltgt_tsmom,lgbm_policy")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    P = H.Panel("crypto")
    pool = np.concatenate([P.split_days("dev"), P.split_days("val")])
    days = sorted(int(i) for i in rng.choice(pool, a.n, replace=False))
    days[-1] = int(P.split_days("val")[-1])
    for i in days:
        c0 = H.build_context(P, i)
        c1 = H.build_context(perturbed(P, i, rng), i)
        assert same(c0, c1), f"leak at {P.days[i]}"
        Q = perturbed(P, i - 1, rng)
        for f in Q.daily:
            Q.daily[f][i:] = P.daily[f][i:]
        assert not same(c0, H.build_context(Q, i)), f"D-1 daily not used at {P.days[i]}"
        Q = H.Panel.__new__(H.Panel)
        Q.__dict__.update(P.__dict__)
        Q.hourly = {f: x.copy() for f, x in P.hourly.items()}
        Q.hourly["close"][24 * i - 1] += 1.0
        assert not same(c0, H.build_context(Q, i)), "last hour before decision not used"
        assert c0["hours"][-1] + np.timedelta64(1, "h") == np.datetime64(c0["decision_time"], "ns")
        assert pd.Timestamp(c0["days"][-1]) + pd.Timedelta(days=1) == c0["decision_time"]
    print(f"1-2 perturbation/sensitivity: OK on {len(days)} decision days "
          f"({pd.Timestamp(P.days[days[0]]).date()} .. {pd.Timestamp(P.days[days[-1]]).date()})")
    E = H.Panel("etf")
    for i in rng.choice(np.concatenate([E.split_days("dev"), E.split_days("val")]), 6, replace=False):
        i = int(i)
        c0 = H.build_context(E, i)
        assert same(c0, H.build_context(perturbed(E, i, rng), i)), f"ETF leak at {E.days[i]}"
        assert pd.Timestamp(c0["days"][-1]) == pd.Timestamp(E.days[i - 1])
    print("1b ETF perturbation: OK on 6 sessions (context = sessions <= D-1)")
    # 3 raw re-derivation
    checked = 0
    for s in rng.choice(P.symbols, 4, replace=False):
        fs = sorted(glob.glob(os.path.join(DT.RAW, "um", "klines_5m", s, f"{s}-5m-202[45]-0[1-9].zip")))
        f = fs[int(rng.integers(len(fs)))]
        raw = DT._read_kline_zip(f).sort_values("t").reset_index(drop=True)
        for day in rng.choice(raw["t"].dt.floor("D").unique()[1:-1], 2, replace=False):
            day = pd.Timestamp(day)
            i, j = P.index(day), P.symbols.index(s)
            b = raw[raw["t"] == day + pd.Timedelta(minutes=5)].iloc[0]
            assert b["t"] >= day + pd.Timedelta(minutes=5), "fill bar not after decision + 1 bar"
            fill = (b["open"] + b["high"] + b["low"] + b["close"]) / 4
            assert np.isclose(P.daily["fill"][i, j], fill, rtol=1e-12), (s, day)
            w = raw[(raw["t"] >= day) & (raw["t"] < day + pd.Timedelta(days=1))]
            prev_close = raw[raw["t"] < day + pd.Timedelta(minutes=5)]["close"].iloc[-1]
            lab = w[w["t"] >= day + pd.Timedelta(minutes=5)]
            c = np.concatenate([[prev_close], lab["close"].to_numpy(float)])
            rv = float((np.diff(np.log(c)) ** 2).sum())
            if len(lab) >= DT.MIN_BARS:
                assert np.isclose(P.daily["rv"][i, j], rv, rtol=1e-9), (s, day, P.daily["rv"][i, j], rv)
            checked += 1
    print(f"3 raw re-derivation of fill and RV: OK ({checked} coin-days)")
    # 4 P&L timing
    i = int(P.split_days("val")[10])
    r0 = P.asset_return(i)[0]
    Q = perturbed(P, i + 2, rng)
    assert np.array_equal(r0, Q.asset_return(i)[0], equal_nan=True)
    Q = perturbed(P, i + 1, rng)
    assert not np.array_equal(r0, Q.asset_return(i)[0], equal_nan=True), "fill(D+1) not used"
    print("4 P&L timing: OK (uses fill(D), fill(D+1) only)")
    # 5 training view
    import models as M
    T = M.train_view()
    tr, _ = DT.universe()
    assert T.days.max() == np.datetime64("2025-06-30", "ns") and T.symbols == tr
    print(f"5 train_view: OK (last day {pd.Timestamp(T.days.max()).date()}, {len(T.symbols)} training coins)")
    # 6 guard
    try:
        H.run(P, Snooper(), P.split_days("val")[:1])
        raise AssertionError("data-access guard did not fire")
    except PermissionError:
        pass
    class Snooper2(Snooper):
        def predict(self, ctx):
            open(os.path.join(H.ROOT, "runs", "fin", "label_corr.json"), "rb").read(10)
            return {}
    try:
        H.run(P, Snooper2(), P.split_days("val")[:1])
        raise AssertionError("data-access guard did not fire on runs/fin")
    except PermissionError:
        pass
    print("6 data-access guard: OK (open() under data/fin/ and runs/fin/ refused during predict)")
    # 7 models
    sub = [int(x) for x in P.split_days("val")[[0, 30, 60, 90]]]
    for mn in [m for m in a.models.split(",") if m]:
        model = M.get(mn)
        outs = []
        for variant in range(3):
            res = []
            for i in sub:
                PP = perturbed(P, i, rng) if variant == 1 else P
                o = H.run(PP, model, [i], online=False)
                res.append(o["pos"] if "pos" in o else np.concatenate([o["p_up"], o["q"].reshape(1, -1)], 1))
            outs.append(np.concatenate(res))
        assert np.array_equal(outs[0], outs[1], equal_nan=True), f"{mn}: changed when the future was perturbed"
        assert np.array_equal(outs[0], outs[2], equal_nan=True), f"{mn}: not deterministic"
        print(f"7 model {mn}: future-invariant and deterministic on {len(sub)} days")
    vd = P.split_days("val")[:40]
    cut = int(vd[25])
    o0 = H.run(P, ToyOnline(), vd)
    o1 = H.run(perturbed(P, cut, rng), ToyOnline(), vd)
    k = int(np.nonzero(vd == cut)[0][0])
    assert np.array_equal(o0["q"][:k + 1], o1["q"][:k + 1], equal_nan=True), "online decisions changed before cut"
    assert not np.array_equal(o0["q"][k + 1:], o1["q"][k + 1:], equal_nan=True), "online model ignores updates"
    assert o0["updates"].sum() == len(vd)
    print(f"7 online runner: OK (decisions <= cut bit-identical, {int(o0['updates'].sum())} updates logged)")
    print("ALL INFORMATION-SET TESTS PASSED")


if __name__ == "__main__":
    main()
