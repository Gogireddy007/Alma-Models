"""Score the sealed ex-ante forecasts of the Oct/Nov 2026 deployment once the targets are published.

    .venv/bin/python tools/gas/fetch.py --only storage --until <today>     # after the EIA release (gas)
    curl -o data/commodity/raw/DCOILWTICO.csv 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO'  (oil)
    .venv/bin/python tools/deploy/score_exante.py        # -> runs/deploy_2026_10_11/EXANTE_SCORES.json

Refuses to score if a sealed forecast file no longer matches the sha256 recorded when it was written (i.e. the
forecast was edited after the fact). Never re-runs a model: it only reads the sealed quantiles.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "runs", "deploy_2026_10_11")
LV = np.round(np.arange(0.1, 0.91, 0.1), 2)


def sealed(project):
    r = json.load(open(os.path.join(RUN, project, "results.json")))
    p = os.path.join(ROOT, r["preds_file"])
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != r["preds_sha256"]:
        raise SystemExit(f"refused: {p} changed since it was sealed ({got} != {r['preds_sha256']})")
    return r, np.load(p)


def pinball(y, q9):
    d = y - np.asarray(q9)
    return float(np.mean(np.where(d >= 0, LV * d, (LV - 1) * d)))


def gas():
    r, z = sealed("gas")
    ex = r["ex_ante"]
    x = pd.read_excel(os.path.join(ROOT, "data/gas/raw/ngshistory.xls"), sheet_name=0, header=None, engine="xlrd")
    hdr = int(np.where(x[0].astype(str).str.strip() == "Week ending")[0][0])
    col = int(np.where(x.iloc[hdr].astype(str).str.strip() == "Total Lower 48")[0][0])
    t = x.iloc[hdr + 1:, [0, col]].dropna()
    s = pd.Series(t.iloc[:, 1].astype(float).to_numpy(), index=pd.to_datetime(t.iloc[:, 0]).dt.strftime("%Y-%m-%d"))
    wk = ex["target_week_ending"]
    if wk not in s.index:
        return {"status": f"not published yet (week ending {wk}, release {ex['scheduled_release']})"}
    y = float(s.diff().loc[wk])
    out = {"target_week_ending": wk, "actual_bcf": y, "scores": {}}
    for kind in ex["forecasts"]:
        for k in z.files:
            if k.startswith(f"exante__{kind}__"):
                q = z[k][0]
                mn = k.split("__")[-1]
                out["scores"].setdefault(kind, {})[mn] = {"abs_err": abs(q[6] - y), "qs": pinball(y, q[2:11]),
                                                          "in_80": bool(q[2] <= y <= q[10])}
    return out


def oil():
    r, z = sealed("oil")
    raw = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    raw.columns = ["date", "v"]
    raw["v"] = pd.to_numeric(raw["v"], errors="coerce")
    raw = raw.dropna()
    after = raw[raw.date > r["ex_ante"]["origin_after"]].reset_index(drop=True)
    out = {"origin_after": r["ex_ante"]["origin_after"], "n_new_prints": int(len(after)), "scores": {}}
    for h in (1, 5, 20):
        if len(after) < h:
            out["scores"][f"h{h}"] = "not published yet"
            continue
        y = np.log(after.v.iloc[h - 1])
        out["scores"][f"h{h}"] = {"date": after.date.iloc[h - 1], "actual": float(after.v.iloc[h - 1]),
                                  "qs_log": {k[2:]: pinball(y, np.log(z[k][-1, 2:11, h - 1]))
                                             for k in z.files if k.startswith("q_")}}
    return out


def power():
    """Every sealed exante_<D>.json: realised hourly prices from the raw Energy-Charts JSON (Berlin local hours)."""
    import glob
    out = {}
    for jf in sorted(glob.glob(os.path.join(RUN, "power", "exante_*.json"))):
        r = json.load(open(jf))
        p = os.path.join(ROOT, r["preds_file"])
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != r["preds_sha256"]:
            raise SystemExit(f"refused: {p} changed since it was sealed")
        z = np.load(p)
        day = r["delivery_day"]
        raw = json.load(open(os.path.join(ROOT, f"data/power/raw/price_de_lu/{day[:4]}.json")))
        df = pd.DataFrame({"ts": raw["unix_seconds"], "p": raw["price"]}).drop_duplicates("ts")
        t = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/Berlin")
        hr = df[t.dt.strftime("%Y-%m-%d") == day].groupby(t.dt.hour)["p"].mean()
        if len(hr) < 23:
            out[day] = "not published yet"
            continue
        y = hr.reindex(range(24)).to_numpy(float)
        prev = r["prev_day_base_price"]
        res = {"actual_base": float(np.nanmean(y)), "models": {}}
        for k in z.files:
            if k.startswith("q_"):
                q = z[k]                                                   # [9, 24]
                d = y[None, :] - q
                res["models"][k[2:]] = {
                    "qs": float(np.nanmean(np.where(d >= 0, LV[:, None] * d, (LV[:, None] - 1) * d))),
                    "mae": float(np.nanmean(np.abs(q[4] - y))),
                    "base_dir_hit": bool((np.mean(q[4]) > prev) == (np.nanmean(y) > prev)),
                    "cov80": float(np.nanmean((y >= q[0]) & (y <= q[-1])))}
        out[day] = res
    return out


if __name__ == "__main__":
    res = {"gas": gas(), "oil": oil(), "power": power()}
    json.dump(res, open(os.path.join(RUN, "EXANTE_SCORES.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:3000])
