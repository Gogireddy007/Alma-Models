"""
Independent re-scorer for the prediction store (tools/predstore.py). Recomputes wql_rel, mase_rel and
cov80_error per task and in aggregate from saved forecasts only; no model is run.

The metric code below is written from the definitions (float64 throughout), not imported from
tools/bench.py or prepare.py. bench is used only to rebuild targets and contexts (load_tasks); every
target is checked against the sha256 recorded in the store before scoring.

Definitions (bench protocol, one task = all windows of one dataset config)
  sanitise   non-finite forecast values -> 0 (counted), then sort along the quantile axis
  mask       finite target points
  WQL        mean over levels a of 2 * sum_mask pinball_a(y, q_a) / sum_mask |y|
  MASE       per window: mean_mask |y - q_0.5| / s ;  s = mean |c_t - c_{t-m}| over the context with
             NaNs removed (m -> 1 if len <= m; s undefined if still len <= m or s == 0);
             task MASE = mean over windows with defined s
  cov80      share of masked points with q_0.1 <= y <= q_0.9
  SN         seasonal naive on the ffill/bfill/0-filled context (m -> 1 if len < m), all quantiles = SN
  *_rel      model / max(SN, 1e-8)
  aggregate  wql_rel, mase_rel: geometric mean over tasks (values clipped at 1e-6);
             wql_rel_family: gmean over families of per-family gmeans; cov80_error: mean |cov80 - 0.8|

    uv run tools/rescore.py b_curr [syn5_10k ...]      # writes runs/preds/<id>/<split>/rescore.{csv,json}
    uv run tools/rescore.py b_curr --check-bench        # also replays the stored forecasts through
                                                        # bench.evaluate; fails unless |diff| <= 1e-6
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
import predstore as PS  # noqa: E402

TOL = 1e-6
HEAD = ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family")


def ffill_bfill_zero(x):
    x = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.where(ok, np.arange(len(x)), -1)
    idx = np.maximum.accumulate(idx)
    first = int(np.argmax(ok))
    idx[idx < 0] = first                       # leading gap: back-fill from the first finite value
    return x[idx]


def seasonal_naive(ctx, h, m):
    c = ffill_bfill_zero(ctx)
    if len(c) < m:
        m = 1
    return np.resize(c[-m:], h) if m > 0 else np.zeros(h)


def mase_scale(ctx, m):
    c = np.asarray(ctx, dtype=np.float64)
    c = c[~np.isnan(c)]
    if len(c) <= m:
        m = 1
    if len(c) <= m:
        return np.nan
    s = np.mean(np.abs(c[m:] - c[:-m]))
    return s if s > 0 else np.nan


def task_scores(q, y, ctxs, m, levels):
    """q [n, Q, h] (sanitised, sorted), y [n, h] -> (wql, mase, cov80)."""
    q = q.astype(np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y64)
    yz = np.where(mask, y64, 0.0)
    a = np.asarray(levels, dtype=np.float64)[None, :, None]
    e = yz[:, None, :] - q
    pin = np.where(e >= 0, a * e, (a - 1) * e) * mask[:, None, :]
    denom = max(np.abs(yz).sum(), 1e-8)
    wql = float(np.mean(2.0 * pin.sum(axis=(0, 2)) / denom))
    med = q[:, int(np.argmin(np.abs(a.ravel() - 0.5))), :]
    ae = (np.abs(yz - med) * mask).sum(1) / np.maximum(mask.sum(1), 1)
    sc = np.array([mase_scale(c, m) for c in ctxs])
    r = ae / sc
    r = r[np.isfinite(r)]
    mase = float(r.mean()) if len(r) else np.nan
    inside = (yz >= q[:, 0, :]) & (yz <= q[:, -1, :]) & mask
    cov = float(inside.sum() / max(mask.sum(), 1))
    return wql, mase, cov


def gmean(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 1e-6, None)
    return float(np.exp(np.log(v).mean()))


def rescore(model_id, split, tasks_by_name):
    meta, preds = PS.load_split(model_id, split)
    levels = meta["quantile_levels"]
    rows = []
    for name, rec in meta["splits"][split]["tasks"].items():
        t = tasks_by_name[name]
        if PS.target_sha256(t["targets"]) != rec["target_sha256"]:
            raise SystemExit(f"{model_id}/{name}: target sha256 mismatch (store vs bench) - protocol changed")
        q = preds[name]["q"].astype(np.float64)
        if not (np.array_equal(preds[name]["window"], t["window"])
                and list(preds[name]["series_id"]) == list(t["series_id"])):
            raise SystemExit(f"{model_id}/{name}: window alignment differs from bench")
        n_bad = int((~np.isfinite(q)).any(axis=(1, 2)).sum())
        q = np.sort(np.where(np.isfinite(q), q, 0.0), axis=1)
        h, m, ctxs = t["horizon"], t["season"], t["contexts"]
        wql, mase, cov = task_scores(q, t["targets"], ctxs, m, levels)
        sn = np.stack([seasonal_naive(c, h, m) for c in ctxs])
        sn_wql, sn_mase, _ = task_scores(np.repeat(sn[:, None, :], len(levels), 1), t["targets"], ctxs, m, levels)
        rows.append({"dataset": name, "family": t["family"], "n_forecasts": len(ctxs), "horizon": h,
                     "wql": wql, "mase": mase, "wql_rel": wql / max(sn_wql, 1e-8),
                     "mase_rel": mase / max(sn_mase, 1e-8), "cov80": cov, "n_nonfinite": n_bad,
                     "sn_wql": sn_wql, "sn_mase": sn_mase})
    df = pd.DataFrame(rows)
    fam = df.groupby("family")["wql_rel"].apply(gmean)
    agg = {"wql_rel": gmean(df["wql_rel"]), "mase_rel": gmean(df["mase_rel"]),
           "cov80_error": float(np.mean(np.abs(df["cov80"] - 0.8))), "wql_rel_family": gmean(fam),
           "win_rate_vs_sn": float((df["wql_rel"] < 1).mean()), "n_tasks": len(df), "n_families": len(fam)}
    return meta, df, agg


def bench_replay(model_id, split, tasks):
    """bench.evaluate on the stored forecasts (cross-check only; never the source of rescore numbers)."""
    BN = PS._bench()
    _, preds = PS.load_split(model_id, split)
    it = {"i": 0}
    order = [t["name"] for t in tasks if t["contexts"] and t["name"] in preds]

    def replay(ctxs, h, m, freq):
        name = order[it["i"]]; it["i"] += 1
        assert preds[name]["q"].shape[0] == len(ctxs) and preds[name]["q"].shape[2] == h
        return preds[name]["q"]
    replay.__name__ = f"predstore_replay_{model_id}"
    return BN.evaluate(replay, split=split, verbose=False, tasks=[t for t in tasks if t["name"] in preds])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="val", choices=PS.ALLOWED_SPLITS)
    ap.add_argument("--check-bench", action="store_true")
    a = ap.parse_args()
    BN = PS._bench()
    tasks = BN.load_tasks(a.split)
    by_name = {t["name"]: t for t in tasks}
    fail = False
    for mid in a.models:
        meta, df, agg = rescore(mid, a.split, by_name)
        out = os.path.join(PS.STORE, mid, a.split)
        df.to_csv(os.path.join(out, "rescore.csv"), index=False)
        report = {"model_id": mid, "split": a.split, "rescore": agg,
                  "harness_headline": (meta.get("harness_headline") or {}).get(a.split)}
        line = f"{mid:24s} rescore wql_rel={agg['wql_rel']:.6f} mase_rel={agg['mase_rel']:.6f} " \
               f"cov80_error={agg['cov80_error']:.6f} fam={agg['wql_rel_family']:.6f} ({agg['n_tasks']} tasks)"
        hh = report["harness_headline"]
        if hh:
            d = max(abs(agg[k] - hh[k]) for k in HEAD if k in hh)
            report["max_abs_diff_vs_harness_headline"] = d
            line += f" | vs logged {d:.2e}"
        if a.check_bench:
            res = bench_replay(mid, a.split, tasks)
            b = res["per_dataset"].set_index("dataset")
            r = df.set_index("dataset")
            dt = max(float(np.nanmax(np.abs(r[k] - b.loc[r.index, k]))) for k in ("wql_rel", "mase_rel", "cov80"))
            da = max(abs(agg[k] - res[k]) for k in HEAD)
            report.update({"bench_replay": {k: float(res[k]) for k in HEAD},
                           "max_abs_diff_per_task": dt, "max_abs_diff_aggregate": da})
            ok = dt <= TOL and da <= TOL
            fail |= not ok
            line += f" | bench replay task {dt:.1e} agg {da:.1e} {'OK' if ok else 'MISMATCH'}"
        json.dump(report, open(os.path.join(out, "rescore.json"), "w"), indent=1, default=float)
        print(line, flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
