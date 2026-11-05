"""Independent integrity audit of the Oct/Nov 2026 deployment (runs/deploy_2026_10_11/). Exit code 1 on any FAIL.

    .venv/bin/python tools/deploy/audit_deploy.py            # -> runs/deploy_2026_10_11/AUDIT.json + AUDIT.md

Written to be independent of the deploy scripts: it does NOT import tools/deploy/common.py or reuse their scoring.
It reads the saved forecasts (preds/*.npz) and the RAW downloaded source files, and checks:

  A. pins        every frozen checkpoint matches winning_models/*/SHA256SUMS and tools/forward/FREEZE.json; every frozen
                 wrapper file matches FREEZE.json; git shows no edits to frozen model code or checkpoints.
  B. no-fit      the deploy scripts contain no optimizer / backward / fit call on an Alma model; every results file
                 reports unchanged parameter hashes (and 0 continual updates for fin).
  C. truth       the realised values used for scoring are re-derived from the RAW files with separate parsing code:
                 WTI from FRED csv; DE-LU prices from Energy-Charts JSON (15-min -> local hourly means); gas storage
                 changes from EIA ngshistory.xls; crypto log-RV from the Binance 5-minute zips.
  D. re-score    every headline metric in the results JSON is recomputed from the saved forecasts with an
                 independent pinball implementation; any difference > 1e-9 (relative) fails.
  E. leakage     future-scramble tests on each world model: every value at/after the forecast target time is
                 replaced by noise (prices, storage, weather, RV, funding ...) and the forecast must not change.
  F. sample size every 'better/worse' verdict must rest on >= 10 independent scoring units.
  G. sealing     sha256 of every forecast file; ex-ante forecasts are listed with their target publication dates.
"""
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.environ.get("ALMA_DEPLOY_OUT") or os.path.join(ROOT, "runs", "deploy_2026_10_11")   # daily runs: runs/deploy_daily/<date>
DAILY = os.path.basename(os.path.dirname(os.path.abspath(RUN))) == "deploy_daily"   # daily runs score 'corrected' fin only
FIN_VARIANTS = ("corrected",) if DAILY else ("pinned", "corrected")
LV = np.round(np.arange(0.1, 0.91, 0.1), 2)
QS13 = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
CHECKS = []


def gas_nocache(p):
    """The frozen gas wrappers memoise storage() by (target_week, origin) only, not by content. A test that re-queries
    the same week with altered inputs must clear every cache first, or it silently gets the old answer
    (found by the V3 positive control; tools/gas/wm_gas.py itself is frozen and is not changed)."""
    for o in [p] + list(getattr(p, "ens", [])):
        for attr in ("_cache", "_raw"):           # wm_gas: _cache; wm_gas_ens: _cache and the persistent _raw
            if hasattr(o, attr):
                setattr(o, attr, {})
    return p


def rerun_check(project, model, saved, fresh, detail=""):
    """R: a saved forecast must be exactly what the frozen model produces when re-run on the same inputs (so a
    fabricated forecast file with consistent metrics is still caught)."""
    saved, fresh = np.asarray(saved, float), np.asarray(fresh, float)
    ok = saved.shape == fresh.shape and np.allclose(saved, fresh, rtol=1e-5, atol=1e-6, equal_nan=True)
    diff = float(np.nanmax(np.abs(saved - fresh))) if saved.shape == fresh.shape else float("inf")
    check("R", f"{project}: saved {model} forecasts = fresh re-run of the frozen model", ok,
          f"{detail}max |diff| {diff:.1e}")


def check(section, name, ok, detail=""):
    CHECKS.append({"section": section, "check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} {name} {detail}", flush=True)
    return ok


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pinball(y, q9):
    d = np.asarray(y, float)[..., None] - np.asarray(q9, float)
    return np.where(d >= 0, LV * d, (LV - 1) * d).mean(-1)


def close(a, b, tol=1e-9):
    a, b = float(a), float(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def load(project, name):
    p = os.path.join(RUN, project, name)
    return json.load(open(p)) if os.path.exists(p) else None


# =========================================================================== A. pins
def audit_pins():
    freeze = json.load(open(os.path.join(ROOT, "tools/forward/FREEZE.json")))["sha256"]
    n = 0
    for sums in glob.glob(os.path.join(ROOT, "winning_models/*/checkpoints/SHA256SUMS")):
        d = os.path.dirname(sums)
        for line in open(sums):
            if not line.strip():
                continue
            s, f = line.split()
            p = os.path.join(d, f.lstrip("*"))
            rel = os.path.relpath(p, ROOT)
            ok = sha(p) == s and (rel not in freeze or freeze[rel] == s)
            n += 1
            if not ok:
                check("A", f"checkpoint {rel}", False, "sha256 mismatch")
    check("A", "all winning_models checkpoints match SHA256SUMS and FREEZE.json", True, f"{n} files")
    bad = [p for p, s in freeze.items() if sha(os.path.join(ROOT, p)) != s]
    check("A", "all FREEZE.json files (code + checkpoints) unchanged", not bad, f"{len(freeze)} files, changed: {bad}")
    diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "tools/gas", "tools/power", "tools/fin",
                           "tools/commodity", "winning_models", "tools/forward"], cwd=ROOT, capture_output=True,
                          text=True).stdout.split()
    check("A", "git: no edits to frozen model code / checkpoints / forward freeze", not diff, f"{diff}")
    # oil wrapper pins (not in FREEZE.json): wm_commodity.FROZEN shas == winning_models SHA256SUMS
    src = open(os.path.join(ROOT, "tools/commodity/wm_commodity.py")).read()
    shas = set(re.findall(r'"([0-9a-f]{64})"', src))
    pins = {l.split()[0] for l in open(os.path.join(ROOT, "winning_models/oil_wm_sv/checkpoints/SHA256SUMS")) if l.strip()}
    check("A", "oil wrapper pins == oil SHA256SUMS", shas == pins, f"{len(shas)} pins")


# =========================================================================== B. no-fit
def audit_nofit():
    bad = []
    for f in glob.glob(os.path.join(ROOT, "tools/deploy/deploy_*.py")):
        s = open(f).read()
        for pat in (r"torch\.optim\.\w+\(", r"\.backward\(", r"\.zero_grad\(", r"optimizer", r"\.train\(\)"):
            if re.search(pat, s):
                bad.append((os.path.basename(f), pat))
    check("B", "deploy scripts contain no optimizer/backward/train() call", not bad, f"{bad}")
    for proj, files in (("oil", ["results.json"]), ("power", ["results.json"]), ("gas", ["results.json"]),
                        ("fin", [f"results_{v}.json" for v in FIN_VARIANTS])):
        for fn in files:
            r = load(proj, fn)
            if r is None:
                check("B", f"{proj}/{fn} exists", False)
                continue
            skipped = {m: v["skipped"][:80] for m, v in r.get("models", {}).items() if isinstance(v, dict) and v.get("skipped")}
            check("B", f"{proj}/{fn}: every registered model ran (no competitor silently skipped)", not skipped,
                  f"{len(r.get('models', {}))} models; skipped: {skipped}")
            pw = r.get("param_watch", [])
            check("B", f"{proj}/{fn}: torch parameters unchanged after deployment",
                  len(pw) > 0 and all(x["unchanged"] for x in pw), f"{len(pw)} modules hashed")
            if proj == "fin":
                upd = {m: v.get("updates") for m, v in r["models"].items() if "updates" in v}
                check("B", f"{proj}/{fn}: zero online updates", all(u == 0 for u in upd.values()), f"{upd}")


# =========================================================================== C + D + F per project
def verdict_sizes(node, path=""):
    """Collect every (path, verdict, n_eff) in a results tree."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("verdict_"):
                out.append((path, v, node.get("n_eff")))
            else:
                out += verdict_sizes(v, f"{path}/{k}")
    return out


def audit_sizes(proj, r):
    vs = verdict_sizes(r)
    bad = [(p, v, n) for p, v, n in vs if v in ("cand_better", "cand_worse") and (n is None or n < 10)]
    check("F", f"{proj}: every significance verdict rests on >= 10 independent units", not bad,
          f"{len(vs)} verdicts, {sum(v in ('cand_better', 'cand_worse') for _, v, _ in vs)} significant, bad={bad[:3]}")


def audit_oil():
    r = load("oil", "results.json")
    z = np.load(os.path.join(ROOT, r["preds_file"]))
    check("G", "oil preds sha256 matches results", sha(os.path.join(ROOT, r["preds_file"])) == r["preds_sha256"])
    raw = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    raw.columns = ["date", "v"]
    raw["v"] = pd.to_numeric(raw["v"], errors="coerce")
    raw = raw.dropna().reset_index(drop=True)
    dates, v = pd.to_datetime(raw.date).to_numpy(), raw.v.to_numpy(float)
    oi = z["origins"]
    # origin date recorded = last context day -> must equal raw date at oi-1, and context must end before target
    od = np.array([str(pd.Timestamp(dates[i - 1]).date()) for i in oi])
    check("C", "oil: origin dates re-derived from raw FRED csv", np.array_equal(od, z["origin_dates"]))
    nbad, ntot = 0, 0
    for wn, w in r["windows"].items():
        lo, hi = {"oct_2026": ("2026-10-01", "2026-10-31"), "nov_2026": ("2026-11-01", "2026-11-30")}[wn]
        for hk, x in w.items():
            if not x.get("n"):
                continue
            h = int(hk[1:])
            t = oi + h - 1
            ok = t < len(v)
            td = np.where(ok, dates[np.minimum(t, len(v) - 1)], np.datetime64("NaT"))
            sel = ok & (td >= np.datetime64(lo)) & (td <= np.datetime64(hi))
            y = np.log(v[t[sel]])
            for mn, s in x["models"].items():
                q = np.log(z[f"q_{mn}"][sel][:, 2:11, h - 1])
                ntot += 1
                nbad += not close(pinball(y, q).mean(), s["qs"])
            check("C", f"oil {wn} {hk}: n targets from raw csv", int(sel.sum()) == x["n"], f"n={x['n']}")
    check("D", "oil: every QS recomputed from saved forecasts", nbad == 0, f"{ntot - nbad}/{ntot} match")
    audit_sizes("oil", r)
    # leak test: scramble everything at/after the origin
    sys.path.insert(0, os.path.join(ROOT, "tools/commodity"))
    import wm_commodity as W
    mods = {n: W.make_predict_ens([(os.path.join(ROOT, p), s) for p, s in m]) for n, m in W.FROZEN.items()}
    rng = np.random.default_rng(1)
    changed, n = 0, 0
    for i in oi[oi < len(v)][::3]:
        for mn, p in mods.items():
            a = p.batch([v[max(0, i - 1024):i]], 20, list(LV))
            vv = v.copy()
            vv[i:] = rng.uniform(5, 500, len(vv) - i)
            b = p.batch([vv[max(0, i - 1024):i]], 20, list(LV))
            changed += not np.allclose(a, b, equal_nan=True)
            n += 1
    check("E", "oil: future-scramble leak test (wm_sv, wm_sv_direct)", changed == 0, f"{changed}/{n} forecasts changed")
    js = list(range(0, len(oi), 5)) + [len(oi) - 1]                      # every 5th origin + the sealed ex-ante one
    for mn in ("wm_sv", "wm_sv_direct"):
        fresh = np.stack([mods[mn].batch([v[max(0, oi[j] - 1024):oi[j]]], 20, QS13)[0] for j in js])
        rerun_check("oil", mn, z[f"q_{mn}"][js], fresh, f"{len(js)} origins, ")


def berlin_hourly_from_raw():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/power/raw/price_de_lu/202[56].json"))):
        j = json.load(open(f))
        rows.append(pd.DataFrame({"ts": j["unix_seconds"], "p": j["price"]}))
    df = pd.concat(rows).drop_duplicates("ts")
    t = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/Berlin")
    df["date"], df["hour"] = t.dt.strftime("%Y-%m-%d"), t.dt.hour
    return df.groupby(["date", "hour"])["p"].mean()


def audit_power():
    r = load("power", "results.json")
    pf = os.path.join(ROOT, r["preds_file"])
    check("G", "power preds sha256 matches results", sha(pf) == r["preds_sha256"])
    zq = np.load(pf)
    zl = np.load(os.path.join(RUN, "power", "preds", "power_LABELS_local.npz"))     # local only (market data)
    assert np.array_equal(zq["delivery_days"], zl["delivery_days"])
    z = {**{k: zq[k] for k in zq.files}, "actual": zl["actual"]}
    hr = berlin_hourly_from_raw()
    days = z["delivery_days"]
    act = np.full((len(days), 24), np.nan)
    for k, d in enumerate(days):
        for h in range(24):
            if (d, h) in hr.index:
                act[k, h] = hr.loc[(d, h)]
    m = np.isfinite(act) & np.isfinite(z["actual"])
    check("C", "power: realised hourly prices re-derived from raw Energy-Charts JSON",
          np.allclose(act[m], z["actual"][m]) and m.sum() >= 0.99 * np.isfinite(z["actual"]).sum(),
          f"{int(m.sum())} hours, max diff {np.nanmax(np.abs(act[m] - z['actual'][m])):.2e}")
    nb, nt = 0, 0
    for wn, w in r["windows"].items():
        if not w.get("n"):
            continue
        sel = (days >= w["first"]) & (days <= w["last"])
        check("C", f"power {wn}: n delivery days", int(sel.sum()) == w["n"], f"n={w['n']}")
        y = z["actual"][sel]
        for c, blk in w["by_candidate"].items():
            for mn, s in blk["qs"].items():
                q = z[f"q_{mn}"][sel]                                   # [n, 9, 24]
                qs = np.nanmean(pinball(y, np.moveaxis(q, 1, 2)), 1)          # [n, 24, 9] -> per day
                nt += 1
                nb += not close(qs.mean(), s["qs"])
    check("D", "power: every QS recomputed from saved forecasts", nb == 0, f"{nt - nb}/{nt} match")
    audit_sizes("power", r)
    # leak test: scramble every price/fundamental of day D and later (same context builder as the harness)
    sys.path.insert(0, os.path.join(ROOT, "tools/power"))
    import data as D
    import models as M
    dates, P, C, flags = D.panel()
    rng = np.random.default_rng(2)
    idx = [int(np.searchsorted(dates, np.datetime64(d))) for d in days[::6]]
    changed, n = 0, 0
    for mn in ("wm_power", "wm_power_direct"):
        p = M.get(mn)
        for i in idx:
            c0 = D.build_context(i, 736, dates, P, C, flags)
            P2, C2 = P.copy(), C.copy()
            P2[i:] = rng.uniform(-500, 3000, P2[i:].shape)
            C2[i - 1:] = rng.uniform(0, 1e5, C2[i - 1:].shape)               # fundamentals of D-1 are unknown too
            c1 = D.build_context(i, 736, dates, P2, C2, flags)
            dk = [{"ctx_dates": c0["ctx_dates"], "target_day": c0["target_day"]}]
            a = p.batch([c0["prices"]], [c0["covs"]], 24, list(LV), dates=dk)
            b = p.batch([c1["prices"]], [c1["covs"]], 24, list(LV), dates=dk)
            changed += not np.allclose(a, b, equal_nan=True)
            n += 1
    check("E", "power: future-scramble leak test (prices >= D, fundamentals >= D-1)", changed == 0,
          f"{changed}/{n} forecasts changed")
    ks = list(range(0, len(days), 4))
    for mn in ("wm_power", "wm_power_direct"):
        p = M.get(mn)
        cs = [D.build_context(int(np.searchsorted(dates, np.datetime64(days[k]))), 736, dates, P, C, flags) for k in ks]
        fresh = p.batch([c["prices"] for c in cs], [c["covs"] for c in cs], 24, list(LV),
                        dates=[{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in cs])
        rerun_check("power", mn, z[f"q_{mn}"][ks], fresh, f"{len(ks)} delivery days, ")


def audit_gas():
    r = load("gas", "results.json")
    pf = os.path.join(ROOT, r["preds_file"])
    check("G", "gas preds sha256 matches results", sha(pf) == r["preds_sha256"])
    z = np.load(pf)
    rep = r["reproduction_vs_audit"]
    check("C", "gas: TEST forecasts reproduce the audit cache (inputs identical)",
          all(v["max_abs_diff_bcf"] < 1e-3 for v in rep.values()) and len(rep) >= 10,
          f"max diff {max(v['max_abs_diff_bcf'] for v in rep.values()):.1e} Bcf over {len(rep)} series x 52 weeks")
    x = pd.read_excel(os.path.join(ROOT, "data/gas/raw/ngshistory.xls"), sheet_name=0, header=None, engine="xlrd")
    hdr = int(np.where(x[0].astype(str).str.strip() == "Week ending")[0][0])
    col = int(np.where(x.iloc[hdr].astype(str).str.strip() == "Total Lower 48")[0][0])
    t = x.iloc[hdr + 1:, [0, col]].dropna()
    t.columns = ["week", "l48"]
    t["week"] = pd.to_datetime(t.week).dt.strftime("%Y-%m-%d")
    s = t.set_index("week")["l48"].astype(float)
    ds = s.diff()
    nb, nt = 0, 0
    for wn, w in r["windows"].items():
        if not w.get("n_weeks"):
            check("G", f"gas {wn}: no EIA release yet -> scored n = 0, ex-ante forecast sealed", "ex_ante" in r,
                  r["ex_ante"]["scheduled_release"])
            continue
        got = [float(ds.loc[k]) for k in w["weeks"]]
        check("C", f"gas {wn}: storage changes re-derived from raw EIA ngshistory.xls",
              np.allclose(got, w["actual_bcf"]), f"{got} vs {w['actual_bcf']}")
        y = np.array(w["actual_bcf"])
        for kind, kk in w["kinds"].items():
            for c, blk in kk["by_candidate"].items():
                for mn, sc in blk["qs"].items():
                    q = z[f"{wn}__{kind}__{mn}"]
                    nt += 2
                    nb += not close(pinball(y, q[:, 2:11]).mean(), sc["qs"])
                    nb += not close(np.abs(q[:, 6] - y).mean(), blk["abs_err"][mn]["abs_err"])
    check("D", "gas: every MAE and QS recomputed from saved forecasts", nb == 0, f"{nt - nb}/{nt} match")
    audit_sizes("gas", r)
    # leak test: scramble storage of week k and later, observed weather from the origin day on, and every CPC
    # forecast issued after the origin, in the global store; forecasts must not change
    sys.path.insert(0, os.path.join(ROOT, "tools/gas"))
    import storage as S
    import models as M
    W0, O0, F0 = S.load()
    rng = np.random.default_rng(3)
    ks = np.where((W0.avail >= "2026-10-01") & W0.ds_l48.notna())[0]
    changed, n = 0, 0
    mods = {mn: M.get(mn) for mn in ("wm_gas", "wm_gas_direct", "wm_gas_ens")}
    for k in ks:
        for kind in S.ORIGINS:
            odt = S.origin_dt(W0, k, kind)
            od = pd.Timestamp(odt.date())
            c0 = S.build_context(int(k), kind, W0, O0, F0)
            W1, O1, F1 = W0.copy(), O0.copy(), F0.copy()
            for col in [c for c in W1.columns if c.startswith(("ds_", "lvl_", "dev_"))]:
                W1.loc[W1.index >= k, col] = rng.normal(0, 300, (W1.index >= k).sum())
            O1.loc[O1.index >= od, ["hdd", "cdd"]] = rng.uniform(0, 60, ((O1.index >= od).sum(), 2))
            late = (F1.issue > od) | ((F1.issue == od) & (F1.last_modified > odt.tz_convert("UTC")))
            F1.loc[late, ["hdd", "cdd"]] = rng.uniform(0, 60, (late.sum(), 2))
            c1 = S.build_context(int(k), kind, W1, O1, F1)
            for mn, p in mods.items():
                a = np.asarray(gas_nocache(p).storage([c0], [0.1, 0.5, 0.9], "l48"), float)
                b = np.asarray(gas_nocache(p).storage([c1], [0.1, 0.5, 0.9], "l48"), float)
                changed += not np.allclose(a, b, equal_nan=True)
                n += 1
    for wn, w in r["windows"].items():
        if not w.get("n_weeks"):
            continue
        kk = [int(np.where(W0.week.dt.strftime("%Y-%m-%d") == wk)[0][0]) for wk in w["weeks"]]
        for kind in S.ORIGINS:
            ctxs = [S.build_context(k, kind, W0, O0, F0) for k in kk]
            for mn in ("wm_gas", "wm_gas_ens"):
                fresh = np.asarray(gas_nocache(mods[mn]).storage(ctxs, QS13, "l48"), float)
                rerun_check("gas", f"{mn} ({wn}, {kind})", z[f"{wn}__{kind}__{mn}"], fresh, f"{len(kk)} weeks, ")
    check("E", "gas: future-scramble leak test (storage >= target week, weather after origin)", changed == 0,
          f"{changed}/{n} forecasts changed")


def logrv_from_raw(sym, day):
    """log RV of day `day` from the raw Binance 5-minute zips (own parser): bars opening 00:05..23:55, return of a bar
    = log close minus log close of the previous present bar, >= 274 returns, else NaN."""
    fs = [os.path.join(ROOT, f"data/fin/raw/um/klines_5m/{sym}/{sym}-5m-{day}.zip")]
    if not os.path.exists(fs[0]):
        fs = [os.path.join(ROOT, f"data/fin/raw/um/klines_5m/{sym}/{sym}-5m-{day[:7]}.zip")]
        if not os.path.exists(fs[0]):
            return np.nan
    with zipfile.ZipFile(fs[0]) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if raw[:9] == b"open_time" else None).iloc[:, :5]
    df.columns = ["ot", "o", "h", "l", "c"]
    ot = df.ot.astype("int64")
    t = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms")
    df = df.assign(t=t).drop_duplicates("t").sort_values("t")
    df = df[df.t.dt.strftime("%Y-%m-%d") == day]
    r = np.diff(np.log(df.c.to_numpy(float)))
    lab = df.t.iloc[1:].to_numpy()
    keep = pd.to_datetime(lab).time >= pd.Timestamp("00:05").time()
    r = r[keep]
    return float(np.log(np.sum(r ** 2))) if len(r) >= 274 else np.nan


def audit_fin():
    for variant in FIN_VARIANTS:
        r = load("fin", f"results_{variant}.json")
        if r is None:
            check("C", f"fin {variant} results exist", False)
            continue
        pf = os.path.join(ROOT, r["preds_file"])
        check("G", f"fin {variant} preds sha256 matches results", sha(pf) == r["preds_sha256"])
        zq = np.load(pf)
        zl = np.load(os.path.join(RUN, "fin", "preds", f"fin_LABELS_local_{variant}.npz"))   # local only
        assert np.array_equal(zq["days"], zl["days"])
        z = {**{k: zq[k] for k in zq.files}, "log_rv": zl["log_rv"], "ref": zl["ref"]}
        days, syms, L = z["days"], list(z["symbols"]), z["log_rv"]
        rng = np.random.default_rng(4)
        pick = [(int(a), int(b)) for a, b in zip(rng.integers(0, len(days), 60), rng.integers(0, len(syms), 60))]
        diffs = []
        for a, b in pick:
            x = logrv_from_raw(syms[b], str(days[a]))
            if np.isfinite(x) or np.isfinite(L[a, b]):
                diffs.append(abs(x - L[a, b]) if np.isfinite(x) and np.isfinite(L[a, b]) else np.inf)
        check("C", f"fin {variant}: log-RV labels re-derived from raw Binance zips (60 random coin-days)",
              len(diffs) > 0 and max(diffs) < 1e-9, f"{len(diffs)} labelled, max |diff| {max(diffs):.1e}")
        nb, nt = 0, 0
        for wn, w in r["windows"].items():
            if not w.get("n_days"):
                continue
            sel = (days >= w["first"]) & (days <= w["last"])
            Ls, Rs = L[sel], z["ref"][sel]
            for c, blk in w["by_candidate"].items():
                for mn, s in blk["qs"].items():
                    q = z[f"q_{mn}"][sel]
                    okq = np.isfinite(Ls) & np.isfinite(q[..., 2:11]).all(-1)
                    okh = np.isfinite(Ls) & np.isfinite(Rs) & (Ls != Rs) & np.isfinite(q[..., 6])
                    with np.errstate(all="ignore"):
                        qs = np.nanmean(np.where(okq, pinball(Ls, q[..., 2:11]), np.nan), 1)
                        hit = np.nanmean(np.where(okh, (q[..., 6] > Rs) == (Ls > Rs), np.nan), 1)
                    if not np.isfinite(qs).any():
                        nt += 1
                        nb += s["qs"] is not None
                        continue
                    nt += 2
                    nb += not close(np.nanmean(qs), s["qs"])
                    rh = blk["hit"][mn]["hit"]
                    nb += (np.isfinite(hit).any() or rh is not None) if (rh is None or not np.isfinite(hit).any()) \
                        else not close(np.nanmean(hit), rh)
        check("D", f"fin {variant}: every QS and direction accuracy recomputed", nb == 0, f"{nt - nb}/{nt} match")
        audit_sizes(f"fin {variant}", r)
    # pinned-variant October forecasts must equal the audit's cached TEST forecasts (same inputs as the TEST year)
    r = load("fin", "results_pinned.json")
    if r is not None:
        z = np.load(os.path.join(ROOT, r["preds_file"]))
        aud = np.load(os.path.join(ROOT, "audit/fin_test/crypto/wm_fin.npz"), allow_pickle=True)
        ad = pd.to_datetime(aud["dates"]).strftime("%Y-%m-%d").to_numpy()
        common = [d for d in z["days"] if d in set(ad)]
        ia = [int(np.where(ad == d)[0][0]) for d in common]
        iz = [int(np.where(z["days"] == d)[0][0]) for d in common]
        assert list(aud["symbols"]) == list(z["symbols"])
        a, b = aud["q"][ia], z["q_wm_fin"][iz]
        m = np.isfinite(a) & np.isfinite(b)
        check("C", "fin pinned: October wm_fin forecasts reproduce the audit's TEST cache",
              m.sum() > 0 and np.nanmax(np.abs(a[m] - b[m])) < 1e-4 and (np.isfinite(a) == np.isfinite(b)).all(),
              f"{len(common)} days x 24 coins, max diff {np.nanmax(np.abs(a[m] - b[m])):.1e}")
    # leak test on the harness: scramble every row dated >= D (daily) / hour >= D 00:00 (hourly)
    os.environ["ALMA_FIN_TEST_GO"] = "1"
    sys.path.insert(0, os.path.join(ROOT, "tools/fin"))
    import data as DT
    import harness as H
    import models as M
    d0, h0 = DT.load("crypto_daily", True), DT.load("crypto_hourly", True)
    P0 = H.Panel("crypto", frames={"daily": d0, "hourly": h0})
    rng = np.random.default_rng(5)
    changed, n = 0, 0
    mods = {mn: M.get(mn) for mn in ("wm_fin", "wm_direct")}
    for day in ("2026-10-07", "2026-10-21", "2026-11-02"):
        i = int(np.searchsorted(P0.days, np.datetime64(day, "ns")))
        d1, h1 = d0.copy(), h0.copy()
        md, mh = pd.to_datetime(d1.t) >= pd.Timestamp(day), pd.to_datetime(h1.t) >= pd.Timestamp(day)
        for col in ("open", "high", "low", "close", "rv", "funding_sum", "quote_volume", "fill"):
            if col in d1:
                d1.loc[md, col] = d1.loc[md, col].to_numpy(float) * rng.uniform(0.2, 5, md.sum())
        for col in ("open", "high", "low", "close", "rv5", "quote_volume", "funding"):
            if col in h1:
                h1.loc[mh, col] = h1.loc[mh, col].to_numpy(float) * rng.uniform(0.2, 5, mh.sum())
        P1 = H.Panel("crypto", frames={"daily": d1, "hourly": h1})
        for mn, p in mods.items():
            a = H.run(P0, p, [i])["q"]
            b = H.run(P1, p, [i])["q"]
            changed += not np.allclose(a, b, equal_nan=True)
            n += 1
    rc = load("fin", "results_corrected.json")
    if rc is not None:
        zc = np.load(os.path.join(ROOT, rc["preds_file"]))
        sel = list(range(0, len(zc["days"]), 6))
        di = [int(np.searchsorted(P0.days, np.datetime64(str(zc["days"][k]), "ns"))) for k in sel]
        for mn in ("wm_fin", "wm_direct"):
            fresh = H.run(P0, mods[mn], di)["q"]
            rerun_check("fin", mn, zc[f"q_{mn}"][sel], fresh, f"{len(sel)} days x 24 coins, ")
    check("E", "fin: future-scramble leak test (daily rows >= D, hourly rows >= D 00:00)", changed == 0,
          f"{changed}/{n} forecasts changed")


ALL = ["pins", "nofit", "oil", "power", "gas", "fin"]


def main():
    fns = {"pins": audit_pins, "nofit": audit_nofit, "oil": audit_oil, "power": audit_power, "gas": audit_gas,
           "fin": audit_fin}
    if len(sys.argv) == 3 and sys.argv[1] == "--section":          # child: one section, checks to a JSON file
        k = sys.argv[2]
        try:
            fns[k]()
        except Exception as e:  # noqa: BLE001  (a crash is a failed check, never a silent pass)
            check(k, "audit section ran to completion", False, f"{type(e).__name__}: {e}")
        json.dump(CHECKS, open(os.path.join(RUN, f".audit_{k}.json"), "w"))
        return
    # every project ships its own data.py / models.py, so each section runs in a fresh process
    for k in ALL:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--section", k], cwd=ROOT)
        p = os.path.join(RUN, f".audit_{k}.json")
        if os.path.exists(p):
            CHECKS.extend(json.load(open(p)))
            os.remove(p)
        else:
            check(k, "audit section produced output", False, "child process died")
    n_fail = sum(c["status"] == "FAIL" for c in CHECKS)
    out = {"sections": ALL, "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS,
           "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    tag = ""
    json.dump(out, open(os.path.join(RUN, f"AUDIT{tag}.json"), "w"), indent=1)
    with open(os.path.join(RUN, f"AUDIT{tag}.md"), "w") as fh:
        fh.write(f"# Deployment audit: {len(CHECKS) - n_fail}/{len(CHECKS)} checks passed\n\n")
        fh.write("| Section | Check | Status | Detail |\n|---|---|---|---|\n")
        for c in CHECKS:
            fh.write(f"| {c['section']} | {c['check']} | **{c['status']}** | {str(c['detail']).replace('|', '/')} |\n")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
