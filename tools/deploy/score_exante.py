"""Grade every sealed ex-ante forecast (snapshot runs/deploy_2026_10_11 and each runs/deploy_daily/<date>) whose
target has been published. Never re-runs a model: it only reads the sealed quantiles, and refuses to grade a file
whose sha256 differs from the one recorded when it was sealed.

    .venv/bin/python tools/deploy/score_exante.py   # -> runs/deploy_daily/EXANTE_LEDGER.json + .md

Truth comes straight from the raw downloads: EIA ngshistory.xls (gas), FRED DCOILWTICO.csv (oil), Energy-Charts
price JSON (power, Berlin local hours). One graded week or day is n = 1: the ledger accumulates them.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(ROOT, "runs", "deploy_2026_10_11")
DAILY = os.path.join(ROOT, "runs", "deploy_daily")
LV = np.round(np.arange(0.1, 0.91, 0.1), 2)
CAND = {"gas": "wm_gas_ens", "oil": "wm_sv", "power": "wm_power"}


def run_dirs():
    return [SNAP] + sorted(d for d in glob.glob(os.path.join(DAILY, "20*")) if os.path.isdir(d))


def sealed(rfile):
    r = json.load(open(rfile))
    p = os.path.join(ROOT, r["preds_file"])
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != r["preds_sha256"]:
        raise SystemExit(f"refused: {p} changed since it was sealed ({got} != {r['preds_sha256']})")
    return r, np.load(p, allow_pickle=False)


def pinball(y, q9):
    """scalar y, q9 [9] -> mean pinball over the 9 levels."""
    d = float(y) - np.asarray(q9, float)
    return float(np.mean(np.where(d >= 0, LV * d, (LV - 1) * d)))


def eia_l48():
    x = pd.read_excel(os.path.join(ROOT, "data/gas/raw/ngshistory.xls"), sheet_name=0, header=None, engine="xlrd")
    hdr = int(np.where(x[0].astype(str).str.strip() == "Week ending")[0][0])
    col = int(np.where(x.iloc[hdr].astype(str).str.strip() == "Total Lower 48")[0][0])
    t = x.iloc[hdr + 1:, [0, col]].dropna()
    s = pd.Series(t.iloc[:, 1].astype(float).to_numpy(), index=pd.to_datetime(t.iloc[:, 0]).dt.strftime("%Y-%m-%d"))
    return s.diff()


def wti():
    raw = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    raw.columns = ["date", "v"]
    raw["v"] = pd.to_numeric(raw["v"], errors="coerce")
    return raw.dropna().reset_index(drop=True)


def power_hourly(day):
    f = os.path.join(ROOT, f"data/power/raw/price_de_lu/{day[:4]}.json")
    if not os.path.exists(f):
        return None
    raw = json.load(open(f))
    df = pd.DataFrame({"ts": raw["unix_seconds"], "p": raw["price"]}).drop_duplicates("ts")
    t = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/Berlin")
    hr = df[t.dt.strftime("%Y-%m-%d") == day].groupby(t.dt.hour)["p"].mean()
    return hr.reindex(range(24)).to_numpy(float) if len(hr) >= 23 else None


def power_qs(y, q):
    """y [24] realised prices, q [9, 24] quantiles -> mean pinball over levels and hours."""
    e = y[None, :] - q
    return float(np.nanmean(np.where(e >= 0, LV[:, None] * e, (LV[:, None] - 1) * e)))


def grade():
    rows = []
    ds = eia_l48()
    oilp = wti()
    for d in run_dirs():
        tag = os.path.relpath(d, ROOT)
        # ---- gas
        rf = os.path.join(d, "gas", "results.json")
        if os.path.exists(rf):
            r, z = sealed(rf)
            ex = r["ex_ante"]
            wk = ex["target_week_ending"]
            y = float(ds.loc[wk]) if wk in ds.index else None
            for kind in ex["forecasts"]:
                for k in z.files:
                    if not k.startswith(f"exante__{kind}__"):
                        continue
                    q = z[k][0]
                    mn = k.split("__")[-1]
                    rows.append({"run": tag, "project": "gas", "target": f"week {wk} ({kind} origin)", "model": mn,
                                 "published": y is not None, "actual": y, "median": float(q[6]),
                                 "abs_err": None if y is None else abs(q[6] - y),
                                 "qs": None if y is None else pinball(y, q[2:11]),
                                 "in_80": None if y is None else bool(q[2] <= y <= q[10])})
        # ---- oil
        rf = os.path.join(d, "oil", "results.json")
        if os.path.exists(rf):
            r, z = sealed(rf)
            after = oilp[oilp.date > r["ex_ante"]["origin_after"]].reset_index(drop=True)
            for h in (1, 5, 20):
                ok = len(after) >= h
                y = np.log(after.v.iloc[h - 1]) if ok else None
                for k in z.files:
                    if k.startswith("q_"):
                        q = np.log(z[k][-1, :, h - 1])
                        rows.append({"run": tag, "project": "oil",
                                     "target": f"WTI print {h} after {r['ex_ante']['origin_after']}"
                                               + (f" ({after.date.iloc[h - 1]})" if ok else ""),
                                     "model": k[2:], "published": ok, "actual": float(after.v.iloc[h - 1]) if ok else None,
                                     "median": float(np.exp(q[6])), "abs_err": None if not ok else float(abs(np.exp(q[6]) - np.exp(y))),
                                     "qs": None if not ok else pinball(y, q[2:11]),
                                     "in_80": None if not ok else bool(q[2] <= y <= q[10])})
        # ---- power
        for rf in sorted(glob.glob(os.path.join(d, "power", "exante_*.json"))):
            r, z = sealed(rf)
            day = r["delivery_day"]
            y = power_hourly(day)
            for k in z.files:
                if k.startswith("q_"):
                    q = z[k]                                        # [9, 24]
                    rows.append({"run": tag, "project": "power", "target": f"delivery {day} (24 h)", "model": k[2:],
                                 "published": y is not None, "actual": None if y is None else float(np.nanmean(y)),
                                 "median": float(np.mean(q[4])),
                                 "abs_err": None if y is None else float(np.nanmean(np.abs(q[4] - y))),
                                 "qs": None if y is None else power_qs(y, q),
                                 "in_80": None if y is None else float(np.nanmean((y >= q[0]) & (y <= q[-1])))})
    return rows


def main():
    rows = grade()
    os.makedirs(DAILY, exist_ok=True)
    json.dump(rows, open(os.path.join(DAILY, "EXANTE_LEDGER.json"), "w"), indent=1, default=float)
    L = ["# Sealed ex-ante forecasts: grades (generated by tools/deploy/score_exante.py)\n",
         "Each forecast was written and sha256-sealed before its target was published; truth is read from the raw "
         "downloads. One graded week/day is n = 1, not evidence — read the ledger as it accumulates.\n"]
    for proj in ("gas", "power", "oil"):
        pr = [r for r in rows if r["project"] == proj]
        if not pr:
            continue
        L.append(f"## {proj}\n")
        for tgt in dict.fromkeys(r["target"] for r in pr):
            tr = [r for r in pr if r["target"] == tgt]
            runs = sorted({r["run"] for r in tr})
            tr = [r for r in tr if r["run"] == runs[0]]          # the earliest sealing of each target
            if not tr[0]["published"]:
                L.append(f"- **{tgt}**: not published yet (sealed in `{runs[0]}`)\n")
                continue
            L.append(f"**{tgt}** — actual {tr[0]['actual']:.2f} (sealed in `{runs[0]}`)\n")
            L.append("| model | median | abs. error | QS | in 80% interval |\n|---|---|---|---|---|")
            for r in sorted(tr, key=lambda r: r["qs"]):
                star = " ★" if r["model"] == CAND[proj] else ""
                cov = ("yes" if r["in_80"] else "no") if isinstance(r["in_80"], bool) else f"{r['in_80']:.2f} of hours"
                L.append(f"| {r['model']}{star} | {r['median']:.2f} | {r['abs_err']:.2f} | {r['qs']:.4f} | {cov} |")
            L.append("")
    open(os.path.join(DAILY, "EXANTE_LEDGER.md"), "w").write("\n".join(L))
    n_pub = sum(r["published"] for r in rows)
    print(f"{len(rows)} sealed model-forecasts, {n_pub} graded -> runs/deploy_daily/EXANTE_LEDGER.md")


if __name__ == "__main__":
    main()
