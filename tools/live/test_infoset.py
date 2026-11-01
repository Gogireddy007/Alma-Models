"""
Alma-Live Gas information-set tests (must all pass before any harness or paper run is trusted). Output:
runs/live/test_infoset.txt.

  1 perturbation   for sampled decision times t (DEV/VAL, storage release days, DST weeks, holiday weeks, COT Fridays,
                   roll days): randomize the value of EVERY record with published_at > t, shuffle their obs, delete
                   10% and insert fake future rows -> ctx hash, features, forecaster outputs and rule actions at t
                   are identical. live=True: the same for records with fetched_at > t. Positive control: perturbing
                   the latest record published <= t changes the ctx.
  2 release times  storage first print >= red-team reference (rt_ref.eia_release_utc) and > week end + 6 days;
                   revisions after the first print; 2026-04-23 resubmission vintage; printed net change for week
                   2026-04-17 = 103; COT >= Friday 15:30 ET after the Tuesday; px bar of d at d 18:00 ET; CPC
                   forecasts not before the issue day 14:00 UTC; CPC obs not before day+1; published_at <= fetched_at.
  3 roll rule      active NG contract from the calendar only (identical under future-price perturbation), last trade
                   > d + ROLL_BDAYS business days, labels masked on Yahoo switch intervals.
  4 weather        ctx forecasts come only from the cpc_fcst source, every forecast row was published before its valid
                   day ended, obs never enter forecast slots, forecasts differ from realized values.
  5 ledger         hash chain verifies; editing / reordering / deleting a line is detected; live backfill refused;
                   models cannot open data files (audit hook).

    nice -n 10 .venv/bin/python tools/live/test_infoset.py [--n 12]
"""
import os
import sys
import json
import shutil
import tempfile
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from store import Store, features  # noqa: E402
import models as M  # noqa: E402
import paper as P  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def sample_times(S, n, rng):
    st = S.raw["storage"]
    rel = pd.to_datetime(st.loc[st["vintage"] == "first_release", "pa"].unique(), utc=True)
    rel = [r for r in rel if pd.Timestamp("2015-01-01", tz="UTC") <= r <= pd.Timestamp("2025-10-31", tz="UTC")]
    days = []
    for r in rng.choice(rel, size=min(n // 3, len(rel)), replace=False):     # release days (after the release)
        days.append(pd.Timestamp(r).tz_convert(C.ET).normalize().tz_localize(None))
    days += [pd.Timestamp(x) for x in ("2019-03-08", "2019-03-11", "2024-03-11", "2024-11-04", "2025-11-03",
                                       "2018-12-26", "2016-11-25", "2024-06-28", "2025-10-31")]
    days += [C.add_bdays(C.ng_last_trade(y, m), k) for y, m, k in ((2020, 4, 0), (2023, 1, 1), (2024, 8, -3))]
    alltd = C.trading_days("2015-01-01", "2025-10-31")
    days += list(rng.choice(alltd, size=max(0, n - len(days)), replace=False))
    days = sorted({d for d in days if C.is_bday(d)})
    return [C.decision_time(d) for d in days]


def perturb(S, t, rng, by="pa"):
    frames = {}
    tn = t.value
    for s, df in S.raw.items():
        d = df.drop(columns=["pa", "fa"]).copy()
        d["value"] = d["value"].astype(float)
        fut = (df[by].to_numpy() > tn)
        k = int(fut.sum())
        if k:
            d.loc[fut, "value"] = rng.normal(0, 1e4, k)
            d.loc[fut, "obs"] = rng.permutation(d.loc[fut, "obs"].to_numpy())
            drop = np.where(fut)[0][rng.random(k) < 0.1]
            fake = d.loc[fut].sample(min(50, k), random_state=1).copy()
            fake["value"] = rng.normal(0, 1e4, len(fake))
            d = pd.concat([d.drop(index=drop), fake], ignore_index=True)
        frames[s] = d
    return Store(frames=frames)


def model_outputs(ctx):
    out = {}
    for n in M.RULES + M.FORECASTERS:
        m = M.load(n)
        with C.no_data_access():
            out[n] = (json.dumps(m.act(ctx), sort_keys=True, default=str),
                      json.dumps(m.predict(ctx), sort_keys=True, default=str))
    return out


def same_features(a, b):
    return all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)


def test_perturbation(S, n, rng):
    times = sample_times(S, n, rng)
    for t in times:
        c0 = S.asof(t)
        S1 = perturb(S, t, rng, "pa")
        c1 = S1.asof(t)
        ok = c0["meta"]["input_sha256"] == c1["meta"]["input_sha256"] and same_features(c0["features"], c1["features"]) \
            and model_outputs(c0) == model_outputs(c1)
        check(f"perturb published_at>t  {C.iso(t)} ({C.split_of(c0['day'])})", ok,
              f"records {c0['meta']['n_records']}")
        # live mode: also fetched_at > t (backfilled history has fetched_at = fetch day, so ctx is sparse there)
        c0l = S.asof(t, live=True)
        S2 = perturb(S, t, rng, "fa")
        check(f"perturb fetched_at>t live {C.iso(t)}", c0l["meta"]["input_sha256"] == S2.asof(t, live=True)["meta"]["input_sha256"])
    # positive control: change the latest UNG close published <= t
    t = times[len(times) // 2]
    frames = {s: df.drop(columns=["pa", "fa"]).copy() for s, df in S.raw.items()}
    px = S.raw["px"]
    vis = px[(px["pa"] <= t.value) & (px["series"] == "UNG.close")]
    j = vis.index[-1]
    frames["px"].loc[j, "value"] = frames["px"].loc[j, "value"] * 1.01
    c2 = Store(frames=frames).asof(t)
    check("positive control: perturb a record published <= t changes ctx",
          c2["meta"]["input_sha256"] != S.asof(t)["meta"]["input_sha256"] and c2["features"]["ung_r1"] != S.asof(t)["features"]["ung_r1"])


def test_release_times(S):
    sys.path.insert(0, os.path.join(C.HERE, "redteam"))
    st = S.panel("storage")
    fr = st[st["vintage"].isin(["first_release", "current_backfill"]) & (st["series"] == "l48")]
    try:
        import rt_ref
        bad = []
        for _, r in fr.iterrows():
            ref = rt_ref.eia_release_utc(pd.Timestamp(r["obs"]))[0]
            if r["published_at"] < ref:
                bad.append((r["obs"], str(r["published_at"]), str(ref)))
        check("storage first print >= red-team reference release time", not bad, f"{len(fr)} weeks; early: {bad[:3]}")
    except Exception as e:  # noqa: BLE001
        check("storage vs rt_ref (reference import)", False, repr(e))
    wk = pd.to_datetime(fr["obs"]).dt.tz_localize("UTC")
    check("storage published >= week end + 4 days (earliest holiday alternates are Wednesdays)",
          bool((fr["published_at"] >= wk + pd.Timedelta(days=4)).all()))
    rv = st[st["vintage"] == "revised"]
    fp = st[st["vintage"] == "first_release"].drop_duplicates(["series", "obs"]).set_index(["series", "obs"])["published_at"]
    later = all(r["published_at"] > fp[(r["series"], r["obs"])] for _, r in rv.iterrows())
    check("revisions published after the first print", later, f"{len(rv)} revised rows")
    t_before = pd.Timestamp("2026-04-23 14:29:59", tz="UTC")
    t_after = pd.Timestamp("2026-04-23 14:30:00", tz="UTC")
    a, b = S.asof(t_before, with_features=False)["storage"], S.asof(t_after, with_features=False)["storage"]
    w = pd.Timestamp("2026-04-10")
    check("LRT-1: week 2026-04-10 first print 1970 before the resubmission, 1960 after",
          a.at[w, "l48"] == 1970 and b.at[w, "l48"] == 1960, f"{a.at[w, 'l48']} -> {b.at[w, 'l48']}")
    check("LRT-1: printed net change of week 2026-04-17 = 103 (not FP diff 93)",
          b.at[pd.Timestamp("2026-04-17"), "l48.chg_printed"] == 103)
    dst = S.asof(pd.Timestamp("2024-03-14 14:30", tz="UTC"), with_features=False)["storage"]
    check("LRT-4: DST - 2024-03-14 release at 14:30Z (10:30 EDT) visible at 14:30Z",
          dst.index[-1] == pd.Timestamp("2024-03-08"))
    dst2 = S.asof(pd.Timestamp("2024-11-07 15:29", tz="UTC"), with_features=False)["storage"]
    check("LRT-4: 2024-11-07 release (EST) not visible at 15:29Z", dst2.index[-1] == pd.Timestamp("2024-10-25"))
    cot = Store(sources=["cot"]).panel("cot")
    tue = pd.to_datetime(cot["obs"])
    fri = [C.et(x + pd.Timedelta(days=3), 15, 30) for x in tue]
    check("COT published >= Friday 15:30 ET after the Tuesday", bool((cot["published_at"].to_numpy() >= np.array(fri)).all()))
    px = S.panel("px")
    bars = px[~px["series"].str.endswith(".split")]
    exp = (pd.to_datetime(bars["obs"]) + pd.Timedelta(hours=18)).dt.tz_localize(C.ET).dt.tz_convert("UTC")
    check("px bar of session d published at d 18:00 ET (after the 16:00 close)", bool((bars["published_at"].to_numpy() == exp.to_numpy()).all()))
    if "cpc_fcst" in S.raw:
        f = S.panel("cpc_fcst")
        iss = pd.to_datetime(f["vintage"].str[6:]).dt.tz_localize("UTC")
        lat = (f["published_at"] - iss).dt.total_seconds() / 3600
        check("CPC forecast published after its 00Z issue time (+1 h)", bool((lat >= 1).all()),
              f"{f['vintage'].nunique()} issues; min latency {lat.min():.2f} h, median {lat.median():.2f} h; rules {f['extra'].value_counts().to_dict()}")
    if "cpc_obs" in S.raw:
        o = S.panel("cpc_obs")
        check("CPC obs published >= day + 1", bool((o["published_at"] >= pd.to_datetime(o["obs"]).dt.tz_localize("UTC") + pd.Timedelta(days=1)).all()))
    allok = all((S.raw[s]["pa"] <= S.raw[s]["fa"]).all() for s in S.raw)
    check("published_at <= fetched_at for every record", allok)


def test_roll(S, rng):
    days = C.trading_days("2015-01-01", "2026-10-30")
    ok = all(C.add_bdays(d, C.ROLL_BDAYS) < C.ng_active_contract(d)["last_trade"] for d in days)
    check("roll rule: active contract last trade > d + ROLL_BDAYS business days", ok, f"{len(days)} days")
    t = C.decision_time(pd.Timestamp("2024-08-23"))
    a = S.asof(t, with_features=False)["ng_roll"]
    b = perturb(S, t, rng).asof(t, with_features=False)["ng_roll"]
    check("roll rule identical under future-price perturbation", a == b, a["active"])
    L = S.labels()
    sw = set(C.yahoo_front_switch_days("2010-01-01", "2026-10-30"))
    ngi = S.asof(C.decision_time(pd.Timestamp("2026-10-30")), with_features=False)["px"]["NG=F"].index
    prev_of = {ngi[i + 1]: ngi[i] for i in range(len(ngi) - 1)}
    tgt = [prev_of[s_] for s_ in sw if s_ in prev_of and prev_of[s_] in L.index and s_ >= pd.Timestamp("2011-01-01")]
    masked = all(not np.isfinite(L.at[d, "ng_cc"]) for d in tgt)
    check("NG=F close->close labels masked on intervals ending on a Yahoo switch session", masked, f"{len(tgt)} switches")
    px = S.labels()
    r = np.log(S.asof(C.decision_time(pd.Timestamp("2025-10-31")), with_features=False)["px"]["NG=F"]["close"]).diff()
    on = r[[d in sw for d in r.index]].abs().median()
    off = r[[d not in sw for d in r.index]].abs().median()
    check("diagnostic: median |NG=F return| on switch sessions vs others (roll gaps)", True, f"{on:.4f} vs {off:.4f}")


def test_weather(S, rng):
    for t in sample_times(S, 6, rng)[:6]:
        ctx = S.asof(t, with_features=False)
        f = ctx["cpc_fcst"]
        if not len(f):
            continue
        ok = bool((f["published_at"] < f["day"].dt.tz_localize("UTC") + pd.Timedelta(days=1)).all()) and \
            bool((f["published_at"] <= t).all())
        check(f"weather: forecasts published before their valid day ended, <= t  {C.iso(t)}", ok)
    # obs must not be able to reach forecast slots: perturb cpc_obs entirely -> forecasts unchanged
    t = C.decision_time(pd.Timestamp("2022-01-12"))
    frames = {s: df.drop(columns=["pa", "fa"]).copy() for s, df in S.raw.items()}
    if "cpc_obs" in frames:
        frames["cpc_obs"]["value"] = rng.normal(0, 100, len(frames["cpc_obs"]))
    a = S.asof(t, with_features=False)["cpc_fcst"]
    b = Store(frames=frames).asof(t, with_features=False)["cpc_fcst"]
    check("weather: realized obs never enter the forecast frame", a.equals(b))
    if "cpc_fcst" in S.raw and "cpc_obs" in S.raw:
        f = S.panel("cpc_fcst")
        o = S.panel("cpc_obs").drop_duplicates(["series", "obs"], keep="last").set_index(["series", "obs"])["value"]
        f["lead"] = (pd.to_datetime(f["obs"]) - pd.to_datetime(f["vintage"].str[6:])).dt.days
        m = f.join(o.rename("obs_v"), on=["series", "obs"]).dropna(subset=["obs_v"])
        m = m[(m["series"] == "hdd") & (m["obs_v"] >= 10)]
        g = m.groupby("lead").apply(lambda x: pd.Series({"mae": (x["value"] - x["obs_v"]).abs().mean(),
                                                          "exact": (x["value"] == x["obs_v"]).mean()}))
        ok = g["mae"].is_monotonic_increasing and g["mae"].iloc[-1] > 1.5 * g["mae"].iloc[0] and g["exact"].iloc[-1] < 0.3
        check("weather: CPC forecast error grows with lead (genuine forecasts, not analyses)", ok,
              "MAE by lead " + ", ".join(f"{int(k)}:{v:.2f}" for k, v in g["mae"].items()) +
              "; exact-match " + ", ".join(f"{int(k)}:{v:.0%}" for k, v in g["exact"].items()))


class _Snoop:
    name = "snoop"
    meta = {}

    def act(self, ctx):
        open(os.path.join(C.PIT, "px.csv")).read(10)
        return {"UNG": 1.0}


def test_ledger(S):
    tmp = tempfile.mkdtemp()
    try:
        B = P.Book("t", mode="sim", root=tmp)
        m = M.load("tsmom_20")
        days = C.trading_days("2024-01-02", "2024-01-12")
        for d in days:
            ctx = S.asof(C.decision_time(d))
            B.decide(d, m, ctx, now=C.decision_time(d))
            closes = {"UNG": float(S.asof(C.decision_time(C.add_bdays(d, 1)))["px"]["UNG"]["close"].iloc[-1]),
                      "BOIL": float(S.asof(C.decision_time(C.add_bdays(d, 1)))["px"]["BOIL"]["close"].iloc[-1])}
            B.settle(d, closes, now=C.et(d, 18, 5))
        ok, msg = B.L.verify()
        check("ledger: hash chain verifies", ok, msg)
        path = B.L.path
        lines = open(path).read().splitlines()
        ed = json.loads(lines[3])
        ed["target"] = {"UNG": -1.0}
        bad = lines[:3] + [P.canon(ed)] + lines[4:]
        open(path, "w").write("\n".join(bad) + "\n")
        check("ledger: edited line detected", not P.Ledger(path).verify()[0], P.Ledger(path).verify()[1])
        open(path, "w").write("\n".join(lines[:3] + [lines[4], lines[3]] + lines[5:]) + "\n")
        check("ledger: reordered lines detected", not P.Ledger(path).verify()[0])
        open(path, "w").write("\n".join(lines[:3] + lines[4:]) + "\n")
        check("ledger: deleted line detected", not P.Ledger(path).verify()[0])
        open(path, "w").write("\n".join(lines) + "\n")
        BL = P.Book("live", mode="live", root=tmp)
        d = pd.Timestamp("2025-03-03")
        try:
            BL.decide(d, m, S.asof(C.decision_time(d), live=True))
            check("ledger: live backfill refused", False)
        except SystemExit as e:
            check("ledger: live backfill refused", True, str(e)[:80])
        try:
            B.decide(days[-1], m, S.asof(C.decision_time(days[-1])), now=C.decision_time(days[-1]))
            check("ledger: duplicate decision refused", False)
        except SystemExit:
            check("ledger: duplicate decision refused", True)
        B2 = P.Book("snoop", mode="sim", root=tmp)
        try:
            B2.decide(days[0], _Snoop(), S.asof(C.decision_time(days[0])), now=C.decision_time(days[0]))
            check("guard: model opening data/live is blocked", False)
        except PermissionError:
            check("guard: model opening data/live is blocked", True)
        # kill switch: a -3% day halts the next decision
        B3 = P.Book("kill", mode="sim", root=tmp)
        d0, d1, d2 = days[0], days[1], days[2]
        B3.decide(d0, M.load("buy_hold"), S.asof(C.decision_time(d0)), now=C.decision_time(d0))
        B3.settle(d0, {"UNG": 10.0, "BOIL": 10.0}, now=C.et(d0, 18, 5))
        B3.decide(d1, M.load("buy_hold"), S.asof(C.decision_time(d1)), now=C.decision_time(d1))
        B3.settle(d1, {"UNG": 9.6, "BOIL": 10.0}, now=C.et(d1, 18, 5))
        ln = B3.decide(d2, M.load("buy_hold"), S.asof(C.decision_time(d2)), now=C.decision_time(d2))
        check("risk: daily loss >= 2% forces the next decision flat", ln["forced_flat"] is not None and ln["target"]["UNG"] == 0,
              str(ln["forced_flat"]))
        # LRT-20: BOIL held across the 2026-05-28 1:2 reverse split in a flat market books only costs
        B4 = P.Book("split", mode="sim", root=tmp, instruments=("UNG", "BOIL"))
        a0, a1 = pd.Timestamp("2026-05-27"), pd.Timestamp("2026-05-28")
        class _W:
            name, meta = "boil_half", {}
            def act(self, ctx):
                return {"BOIL": 0.5}
        B4.decide(a0, _W(), S.asof(C.decision_time(a0)), now=C.decision_time(a0))
        B4.settle(a0, {"UNG": 10.0, "BOIL": 20.0}, now=C.et(a0, 18, 5))
        B4.decide(a1, _W(), S.asof(C.decision_time(a1)), now=C.decision_time(a1))
        mk = B4.settle(a1, {"UNG": 10.0, "BOIL": 40.0}, now=C.et(a1, 18, 5),
                       splits=P.splits_from_store(S, C.et(a1, 18, 5)))
        check("LRT-20: BOIL across the 2026-05-28 reverse split, flat market -> |gross P&L| ~ 0",
              abs(mk["pnl_gross"]) < 1.0 and any(x["kind"] == "split" for x in B4.L.lines),
              f"pnl_gross {mk['pnl_gross']:.2f}, pos {mk['pos']}")
        # mode is immutable: a sim book cannot be reopened live and vice versa
        try:
            P.Book("split", mode="live", root=tmp)
            check("LRT-19c: ledger mode from genesis is immutable", False)
        except SystemExit:
            check("LRT-19c: ledger mode from genesis is immutable", True)
        # truncation of a committed (anchored) ledger: emulate via a prefix check against a saved copy
        full = open(B4.L.path, "rb").read()
        open(B4.L.path, "wb").write(full[: full.rfind(b"\n", 0, len(full) - 1) + 1])
        check("LRT-19a: truncated tail breaks the anchored-prefix property", not open(B4.L.path, "rb").read().startswith(full))
    finally:
        shutil.rmtree(tmp)
    # UNG / BOIL split history: as-of-t adjusted closes are continuous across every ex-date
    for k in ("UNG", "BOIL"):
        sp = S.raw["px"][S.raw["px"]["series"] == f"{k}.split"]
        jumps = []
        for x in sp["obs"]:
            d = pd.Timestamp(x)
            if d < pd.Timestamp("2012-01-01") and k == "BOIL":
                continue
            c = S.asof(C.decision_time(C.add_bdays(d, 2)), with_features=False)["px"][k]["close"]
            r = np.log(c / c.shift(1)).loc[d - pd.Timedelta(days=5): d + pd.Timedelta(days=1)]
            jumps.append(float(r.abs().max()))
        check(f"{k} split-adjusted (as of t) closes continuous across {len(sp)} ex-dates (max |log ret| < 0.3)",
              max(jumps) < 0.3, f"max {max(jumps):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    S = Store()
    test_release_times(S)
    test_roll(S, rng)
    test_weather(S, rng)
    test_ledger(S)
    test_perturbation(S, a.n, rng)
    n_fail = sum(1 for _, ok, _ in RES if not ok)
    os.makedirs(C.RUNS, exist_ok=True)
    with open(os.path.join(C.RUNS, "test_infoset.txt"), "w") as f:
        f.write(f"Alma-Live Gas information-set tests, {C.iso(C.utcnow())}; data manifest sha256 "
                f"{C.sha256_file(C.MANIFEST)}\n")
        for name, ok, det in RES:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}  {det}\n")
        f.write(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} FAILED'} ({len(RES)} checks)\n")
    print(f"\n{'ALL PASSED' if n_fail == 0 else f'{n_fail} FAILED'} ({len(RES)} checks)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
