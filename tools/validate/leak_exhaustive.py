"""V3 — exhaustive future-scramble leak tests, with positive controls.

    .venv/bin/python tools/validate/leak_exhaustive.py [--projects oil,power,gas,fin]
    # -> runs/validation/leak/RESULTS.json + .md

For EVERY October-November 2026 origin (not a sample) and every frozen world model (and its twin / blend), the data
from the forecast target time onward is corrupted in four ways:
    noise    random values on a realistic scale
    nan      every future value missing
    extreme  every future value multiplied by 1000
    shift    the future replaced by a copy of the past (plausible but wrong values)
and the forecast must stay bit-for-bit (allclose 1e-9) unchanged.

Positive controls: the same procedure is run with a deliberately BUGGY context builder that lets one future value
in (an off-by-one). The test must flag every one of those, otherwise it would be blind.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs", "validation", "leak")
LO, HI = "2026-10-01", "2026-11-30"
KINDS = ("noise", "nan", "extreme", "shift")
RES = []


def gas_nocache(p):
    """The frozen gas wrappers memoise storage() by (target_week, origin) only, not by content. A test that re-queries
    the same week with altered inputs must clear every cache first, or it silently gets the old answer
    (found by the V3 positive control; tools/gas/wm_gas.py itself is frozen and is not changed)."""
    for o in [p] + list(getattr(p, "ens", [])):
        for attr in ("_cache", "_raw"):           # wm_gas: _cache; wm_gas_ens: _cache and the persistent _raw
            if hasattr(o, attr):
                setattr(o, attr, {})
    return p


def corrupt(a, start, kind, rng, past=None):
    """Return a copy of array a with a[start:] corrupted (axis 0 = time)."""
    b = np.array(a, dtype=float, copy=True)
    n = b.shape[0] - start
    if n <= 0:
        return b
    if kind == "noise":
        scale = np.nanstd(b[:start]) or 1.0
        b[start:] = np.nanmean(b[:start]) + rng.normal(0, 3 * scale, b[start:].shape)
    elif kind == "nan":
        b[start:] = np.nan
    elif kind == "extreme":
        b[start:] = b[start:] * 1000.0
    elif kind == "shift":
        src = b[max(0, start - n):start] if past is None else past
        b[start:] = np.resize(src, b[start:].shape)
    return b


def record(project, model, kind, n, changed, control=False):
    ok = (changed > 0) if control else (changed == 0)
    RES.append({"project": project, "model": model, "corruption": kind, "n_forecasts": n, "changed": changed,
                "positive_control": control, "ok": ok})
    tag = "CONTROL" if control else "TEST   "
    print(f"[{'OK ' if ok else 'BAD'}] {tag} {project:5s} {model:18s} {kind:8s} {changed}/{n} changed", flush=True)


def same(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- oil
def oil():
    sys.path.insert(0, os.path.join(ROOT, "tools/commodity"))
    import wm_commodity as W
    df = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    d, v = pd.to_datetime(df.date).to_numpy(), df.v.to_numpy(float)
    oi = np.where((d >= np.datetime64(LO)) & (d <= np.datetime64(HI)))[0]
    mods = {n: W.make_predict_ens([(os.path.join(ROOT, p), s) for p, s in m]) for n, m in W.FROZEN.items()}
    rng = np.random.default_rng(11)
    for mn, p in mods.items():
        base = {i: p.batch([v[max(0, i - 1024):i]], 20, [0.1, 0.5, 0.9]) for i in oi}
        for kind in KINDS:
            ch = 0
            for i in oi:
                vv = corrupt(v, i, kind, rng)
                ch += not same(base[i], p.batch([vv[max(0, i - 1024):i]], 20, [0.1, 0.5, 0.9]))
            record("oil", mn, kind, len(oi), ch)
        ch = 0                                    # positive control: off-by-one context v[:i+1] (sees the target day)
        for i in oi[:-1]:
            a = p.batch([v[max(0, i - 1023):i + 1]], 20, [0.1, 0.5, 0.9])
            vv = corrupt(v, i, "noise", rng)
            ch += not same(a, p.batch([vv[max(0, i - 1023):i + 1]], 20, [0.1, 0.5, 0.9]))
        record("oil", mn, "off-by-one", len(oi) - 1, ch, control=True)


# --------------------------------------------------------------------------- power
def power():
    sys.path.insert(0, os.path.join(ROOT, "tools/power"))
    import data as D
    import models as M
    dates, P, C, flags = D.panel()
    idx = [i for i, x in enumerate(dates) if np.datetime64(LO) <= x <= np.datetime64(HI) and np.isfinite(P[i]).sum() >= 23]
    rng = np.random.default_rng(12)

    def fc(p, c):
        return p.batch([c["prices"]], [c["covs"]], 24, [0.1, 0.5, 0.9],
                       dates=[{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]}])
    for mn in ("wm_power", "wm_power_direct"):
        p = M.get(mn)
        base = {i: fc(p, D.build_context(i, 736, dates, P, C, flags)) for i in idx}
        for kind in KINDS:
            ch = 0
            for i in idx:
                P2, C2 = corrupt(P, i, kind, rng), corrupt(C, i - 1, kind, rng)     # fundamentals of D-1 unknown too
                ch += not same(base[i], fc(p, D.build_context(i, 736, dates, P2, C2, flags)))
            record("power", mn, kind, len(idx), ch)
        ch = 0                                    # positive control: buggy context that includes day D's prices
        for i in idx[:-1]:
            c = D.build_context(i + 1, 736, dates, P, C, flags)
            a = fc(p, c)
            P2 = corrupt(P, i, "noise", rng)
            ch += not same(a, fc(p, D.build_context(i + 1, 736, dates, P2, C, flags)))
        record("power", mn, "includes-D", len(idx) - 1, ch, control=True)


# --------------------------------------------------------------------------- gas
def gas():
    sys.path.insert(0, os.path.join(ROOT, "tools/gas"))
    import storage as S
    import models as M
    W0, O0, F0 = S.load()
    ks = np.where((W0.avail >= LO) & (W0.avail <= HI) & W0.ds_l48.notna())[0]
    rng = np.random.default_rng(13)
    mods = {mn: M.get(mn) for mn in ("wm_gas", "wm_gas_direct", "wm_gas_ens")}
    stor_cols = [c for c in W0.columns if c.startswith(("ds_", "lvl_", "dev_"))]
    for mn, p in mods.items():
        for kind in KINDS:
            ch, n = 0, 0
            for k in ks:
                for orig in S.ORIGINS:
                    odt = S.origin_dt(W0, k, orig)
                    od = pd.Timestamp(odt.date())
                    a = gas_nocache(p).storage([S.build_context(int(k), orig, W0, O0, F0)], [0.1, 0.5, 0.9], "l48")
                    W1, O1, F1 = W0.copy(), O0.copy(), F0.copy()
                    W1[stor_cols] = corrupt(W0[stor_cols].to_numpy(), int(k), kind, rng)
                    io = int(np.searchsorted(O0.index, od))
                    O1[["hdd", "cdd"]] = corrupt(O0[["hdd", "cdd"]].to_numpy(), io, kind, rng)
                    late = ((F1.issue > od) | ((F1.issue == od) & (F1.last_modified > odt.tz_convert("UTC")))).to_numpy()
                    fv = F0[["hdd", "cdd"]].to_numpy(float).copy()
                    fv[late] = corrupt(fv[late], 0, "noise" if kind == "shift" else kind, rng)
                    F1[["hdd", "cdd"]] = fv
                    b = gas_nocache(p).storage([S.build_context(int(k), orig, W1, O1, F1)], [0.1, 0.5, 0.9], "l48")
                    ch += not same(a, b)
                    n += 1
            record("gas", mn, kind, n, ch)
        ch, n = 0, 0                              # positive control: target-week weather taken from OBSERVATIONS
        for k in ks:
            c = S.build_context(int(k), "thu", W0, O0, F0)
            days = pd.to_datetime(c["wx_days"])
            leak = dict(c, wx=O0.reindex(days)[["hdd", "cdd"]].to_numpy(float))
            O1 = O0.copy()
            O1.loc[O1.index >= days[0], ["hdd", "cdd"]] += rng.normal(0, 8, ((O1.index >= days[0]).sum(), 2))
            leak2 = dict(c, wx=O1.reindex(days)[["hdd", "cdd"]].to_numpy(float))
            ch += not same(gas_nocache(p).storage([leak], [0.1, 0.5, 0.9], "l48"), gas_nocache(p).storage([leak2], [0.1, 0.5, 0.9], "l48"))
            n += 1
        record("gas", mn, "observed-wx", n, ch, control=True)


# --------------------------------------------------------------------------- fin
def fin():
    os.environ["ALMA_FIN_TEST_GO"] = "1"
    sys.path.insert(0, os.path.join(ROOT, "tools/fin"))
    import data as DT
    import harness as H
    import models as M
    d0, h0 = DT.load("crypto_daily", True), DT.load("crypto_hourly", True)
    P0 = H.Panel("crypto", frames={"daily": d0, "hourly": h0})
    days = [i for i, x in enumerate(P0.days) if np.datetime64(LO, "ns") <= x <= np.datetime64(HI, "ns")]
    rng = np.random.default_rng(14)
    mods = {mn: M.get(mn) for mn in ("wm_fin", "wm_direct")}
    base = {mn: H.run(P0, p, days)["q"] for mn, p in mods.items()}
    dcols = [c for c in ("open", "high", "low", "close", "rv", "funding_sum", "quote_volume", "volume",
                         "taker_buy_quote_volume", "fill", "n_lab") if c in d0]
    hcols = [c for c in ("open", "high", "low", "close", "rv5", "quote_volume", "volume", "taker_buy_quote_volume",
                         "funding", "n5") if c in h0]
    td, th = pd.to_datetime(d0.t), pd.to_datetime(h0.t)
    for kind in KINDS:
        ch = {mn: 0 for mn in mods}
        for j, i in enumerate(days):
            D = pd.Timestamp(P0.days[i])
            d1, h1 = d0.copy(), h0.copy()
            md, mh = (td >= D).to_numpy(), (th >= D).to_numpy()
            for col in dcols:
                d1.loc[md, col] = corrupt(np.r_[d0[col].to_numpy(float)[~md][-50:], d0[col].to_numpy(float)[md]], 50, kind, rng)[50:]
            for col in hcols:
                h1.loc[mh, col] = corrupt(np.r_[h0[col].to_numpy(float)[~mh][-500:], h0[col].to_numpy(float)[mh]], 500, kind, rng)[500:]
            if kind == "nan":                    # a NaN 'close' would also drop the row from the calendar; keep rows
                d1.loc[md, "close"] = d0.loc[md, "close"]
            P1 = H.Panel("crypto", frames={"daily": d1, "hourly": h1})
            for mn, p in mods.items():
                ch[mn] += not same(base[mn][j], H.run(P1, p, [i])["q"][0])
        for mn in mods:
            record("fin", mn, kind, len(days), ch[mn])
    for mn, p in mods.items():                   # positive control: context of D+1 (includes day D's own row)
        ch = 0
        for i in days[:-1][::3]:
            a = H.run(P0, p, [i + 1])["q"][0]
            D = pd.Timestamp(P0.days[i])
            d1 = d0.copy()
            m = (td == D).to_numpy()
            d1.loc[m, "rv"] = d0.loc[m, "rv"].to_numpy() * 5
            d1.loc[m, "close"] = d0.loc[m, "close"].to_numpy() * 1.3
            P1 = H.Panel("crypto", frames={"daily": d1, "hourly": h0})
            ch += not same(a, H.run(P1, p, [i + 1])["q"][0])
        record("fin", mn, "includes-D", len(days[:-1][::3]), ch, control=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", default="oil,power,gas,fin")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for p in a.projects.split(","):
        {"oil": oil, "power": power, "gas": gas, "fin": fin}[p]()
    path = os.path.join(OUT, "RESULTS.json")
    old = [r for r in (json.load(open(path)) if os.path.exists(path) else []) if r["project"] not in a.projects.split(",")]
    allr = old + RES
    json.dump(allr, open(path, "w"), indent=1)
    L = ["# V3 — Exhaustive leak tests (every Oct-Nov 2026 origin) with positive controls\n",
         "TEST rows: future corrupted, forecast must NOT change (changed = 0). CONTROL rows: a deliberately buggy "
         "context builder lets one future value in, and the same test MUST flag it (changed > 0).\n",
         "| project | model | corruption | forecasts | changed | kind | result |", "|---|---|---|---|---|---|---|"]
    for r in allr:
        L.append(f"| {r['project']} | {r['model']} | {r['corruption']} | {r['n_forecasts']} | {r['changed']} | "
                 f"{'positive control' if r['positive_control'] else 'leak test'} | {'**as expected**' if r['ok'] else '**NOT as expected**'} |")
    n_ok = sum(r["ok"] for r in allr)
    n_fc = sum(r["n_forecasts"] for r in allr if not r["positive_control"])
    L.append(f"\n**{n_ok}/{len(allr)} rows as expected; {n_fc} corrupted-future forecasts checked.**")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_ok}/{len(allr)} as expected; {n_fc} forecasts")


if __name__ == "__main__":
    main()
