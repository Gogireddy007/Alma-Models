"""
Alma-Live Gas historical harness (DEV / VAL). The ONLY clean claim test is the forward paper test from launch
(docs/live_prereg.md); history is for development and sanity. 2025-11-01 onward is EXPOSED (a previous project's
TEST showed HH prices) and is refused unless --allow-exposed, and then labelled EXPOSED everywhere.

Splits (decision days): DEV <= 2024-06-30 (scored 2017-01-03 .. 2024-06-28, walk-forward annual refits for fitted
models, each fit only on days whose label was available before the refit date), VAL 2024-07-01 .. 2025-10-31
(fitted models trained once on all DEV labels available by 2024-06-30 16:00 ET).

Clock: ctx = Store().asof(common.decision_time(d)) (15:30 ET); position filled at the 16:00 ET close of d, held to the
close of d+1: outcome = UNG close(d)->close(d+1). Costs: 10 bp per side x {1, 2, 3}.
Metrics: direction accuracy of sign(position) on non-flat days (Wilson 95% CI), Sharpe (252) net of costs with
block-bootstrap CI, PSR(0), DSR (k = number of strategies scored in the split, variance of their Sharpe), max
drawdown, turnover (mean |delta w|), CRPS-free: p_up Brier where models give it.

    nice -n 10 .venv/bin/python tools/live/run_eval.py --split dev  [--models a,b] [--rebuild-features]
    nice -n 10 .venv/bin/python tools/live/run_eval.py --split val
"""
import os
import sys
import json
import time
import argparse
import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from store import Store, FEATURES, storage_labels  # noqa: E402
import models as M  # noqa: E402

_p = os.path.join(C.ROOT, "tools", "fin", "metrics.py")
_spec = importlib.util.spec_from_file_location("fin_metrics", _p)
FM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FM)

EVAL = os.path.join(C.RUNS, "eval")
FEAT_START = pd.Timestamp("2014-01-02")
DEV_SCORE_START = pd.Timestamp("2017-01-03")
COST_BP = 10.0
COST_MULTS = (1, 2, 3)


def md(df, fmt="{:.4g}"):
    """Markdown table without optional dependencies."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)) for v in r]
        out.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(out)


def data_pins():
    man = C.load_manifest()
    return {k: v["sha256"] for k, v in sorted(man["pit"].items())}


def pins_tag(pins):
    return C.sha256_bytes(json.dumps(pins, sort_keys=True).encode())[:12]


def build_features(S, end, rebuild=False):
    pins = data_pins()
    tag = pins_tag(pins)
    os.makedirs(EVAL, exist_ok=True)
    path = os.path.join(EVAL, f"features_{tag}_{end:%Y%m%d}.parquet")
    if os.path.exists(path) and not rebuild:
        return pd.read_parquet(path), tag
    days = C.trading_days(FEAT_START, end)
    rows = []
    t0 = time.time()
    for i, d in enumerate(days):
        ctx = S.asof(C.decision_time(d))
        rows.append({"day": d, **ctx["features"], "input_sha256": ctx["meta"]["input_sha256"]})
        if i % 250 == 0:
            C.log(f"features {d.date()} ({i}/{len(days)}, {time.time() - t0:.0f}s)")
    X = pd.DataFrame(rows).set_index("day")
    X.to_parquet(path)
    return X, tag


class _PseudoCtx(dict):
    pass


def rule_positions(name, X):
    m = M.load(name)
    out = []
    for d, r in X.iterrows():
        ctx = _PseudoCtx(features=r[FEATURES].to_dict(), day=d)
        with C.no_data_access():
            out.append(m.act(ctx).get("UNG", 0.0))
    return pd.Series(out, index=X.index)


def fitted_positions(name, X, y, avail, score_days, split):
    """Walk-forward (DEV: annual refits) or single DEV fit (VAL). Returns (positions, p_up)."""
    pos = pd.Series(np.nan, index=score_days)
    pup = pd.Series(np.nan, index=score_days)
    if split == "dev":
        blocks = [(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31")) for yr in range(score_days[0].year, score_days[-1].year + 1)]
    else:
        blocks = [(score_days[0], score_days[-1])]
    for b0, b1 in blocks:
        cutoff = min(b0, C.et(C.DEV_END, 16, 0) + pd.Timedelta(days=1)) if split == "val" else b0
        cut = pd.Timestamp(cutoff)
        cut = cut if cut.tzinfo else cut.tz_localize(C.ET).tz_convert("UTC")
        tr = avail.index[(avail < cut) & (avail.index <= C.DEV_END)]
        assert tr.max() <= C.DEV_END, "fit rows beyond DEV"
        m = M.load(name)
        m.fit(X.loc[tr, FEATURES].to_numpy(float), y.loc[tr].to_numpy(float))
        days = [d for d in score_days if b0 <= d <= b1]
        if hasattr(m, "prev"):
            m.prev = 0.0
        for d in days:
            ctx = _PseudoCtx(features=X.loc[d, FEATURES].to_dict(), day=d)
            with C.no_data_access():
                pos[d] = m.act(ctx).get("UNG", 0.0)
                pr = m.predict(ctx)
            pup[d] = pr.get("p_up", np.nan)
    return pos, pup


def zeroshot_positions(name, S, score_days):
    m = M.load(name)
    pos, pup = pd.Series(np.nan, index=score_days), pd.Series(np.nan, index=score_days)
    for d in score_days:
        ctx = S.asof(C.decision_time(d), with_features=False)
        with C.no_data_access():
            pr = m.predict(ctx)
        pup[d] = pr["p_up"]
        pos[d] = float(np.sign(pr["mu"]))
    return pos, pup


def score(pos, ret, k_trials=None, var_sr=None, pup=None):
    pos = pos.reindex(ret.index).fillna(0.0)
    simple = np.expm1(ret)
    dw = pos.diff().abs().fillna(pos.abs())
    out = {}
    nz = pos != 0
    k = int(((np.sign(pos) == np.sign(simple)) & nz & (simple != 0)).sum())
    n = int((nz & (simple != 0)).sum())
    lo, hi = FM.wilson(k, n) if n else (np.nan, np.nan)
    out.update({"n_days": int(len(ret)), "n_active": n, "dir_acc": k / n if n else np.nan, "dir_lo": lo, "dir_hi": hi,
                "turnover": float(dw.mean())})
    for cm in COST_MULTS:
        r = pos * simple - cm * COST_BP * 1e-4 * dw
        t = FM.trading_summary(r.to_numpy(), dw.to_numpy(), 252, k_trials, var_sr)
        out.update({f"sharpe_{cm}x": t["sharpe"], f"sharpe_lo_{cm}x": t["sharpe_boot_lo"], f"sharpe_hi_{cm}x": t["sharpe_boot_hi"],
                    f"psr_{cm}x": t["psr0"], f"dsr_{cm}x": t["dsr"], f"mdd_{cm}x": t["max_drawdown"],
                    f"ann_ret_{cm}x": t["ann_return"]})
    if pup is not None and pup.notna().any():
        p = pup.reindex(ret.index)
        ok = p.notna() & (simple != 0)
        out["brier"] = float(((p[ok] - (simple[ok] > 0)) ** 2).mean())
    return out


def pinball(y, qd):
    if not qd or not np.isfinite(y):
        return np.nan
    return float(np.mean([max(q * (y - v), (q - 1) * (y - v)) for q, v in qd.items()]))


def _boot_mean(x, ref=None):
    x = np.asarray(x, float)
    ok = np.isfinite(x) if ref is None else (np.isfinite(x) & np.isfinite(ref))
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    d = x[ok] if ref is None else x[ok] - np.asarray(ref, float)[ok]
    idx = FM.boot_idx(len(d))
    bs = d[idx].mean(1)
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975)), p


def eval_forecasts(S, names, score_days, split, L):
    """Primary targets. Daily: UNG next-day return quantiles, log RV5 quantiles. Weekly: storage deviation quantiles
    at the last decision time before each report in the split."""
    models = {}
    for n in names:
        try:
            models[n] = M.load(n)
        except Exception as e:  # noqa: BLE001
            C.log(f"{n}: load error {e!r}")
    SL = storage_labels(S)
    SL = SL[np.array([C.split_of(d) == split for d in SL["origin_day"]]) & SL["origin_day"].isin(set(score_days)).to_numpy()]
    rows = []
    days = list(score_days)
    origins = set(SL["origin_day"])
    t0 = time.time()
    for i, d in enumerate(days):
        ctx = S.asof(C.decision_time(d))
        y_r = L.at[d, "ung_ret"]
        rv = L.at[d, "ung_rv5"]
        y_v = np.log(rv) if np.isfinite(rv) and rv > 0 else np.nan
        for n, m in models.items():
            with C.no_data_access():
                pr = m.predict(ctx)
            row = {"day": d, "model": n, "qs_ret": pinball(y_r, pr.get("ung_q")), "qs_rv5": pinball(y_v, pr.get("rv5_q")),
                   "cov90_ret": float(pr["ung_q"][0.05] <= y_r <= pr["ung_q"][0.95]) if pr.get("ung_q") and np.isfinite(y_r) else np.nan,
                   "cov90_rv5": float(pr["rv5_q"][0.05] <= y_v <= pr["rv5_q"][0.95]) if pr.get("rv5_q") and np.isfinite(y_v) else np.nan}
            if d in origins:
                w = SL.index[SL["origin_day"] == d][0]
                if ctx["next_storage"]["week"] == w and pr.get("stor_dev_q"):
                    yd = SL.at[w, "dev"]
                    q = pr["stor_dev_q"]
                    row.update({"week": w, "qs_stor": pinball(yd, q), "cov90_stor": float(q[0.05] <= yd <= q[0.95]),
                                "stor_dir_hit": float(np.sign(q[0.5]) == np.sign(yd)) if q[0.5] != 0 and yd != 0 else np.nan})
            rows.append(row)
        if i % 200 == 0:
            C.log(f"forecasts {d.date()} ({i}/{len(days)}, {time.time() - t0:.0f}s)")
    R = pd.DataFrame(rows)
    out = []
    ref = "fc_hist" if "fc_hist" in models else names[0]
    for n in models:
        g = R[R.model == n].set_index("day")
        gr = R[R.model == ref].set_index("day")
        rec = {"model": n}
        for tgt in ("ret", "rv5", "stor"):
            col = f"qs_{tgt}"
            x = g[col].dropna() if col in g else pd.Series(dtype=float)
            if not len(x):
                continue
            xr = gr[col].reindex(x.index)
            lo, hi, _ = _boot_mean(x.to_numpy())
            dlo, dhi, p = _boot_mean(x.to_numpy(), xr.to_numpy()) if n != ref else (np.nan, np.nan, np.nan)
            rec.update({f"{tgt}_n": int(len(x)), f"{tgt}_qs": float(x.mean()), f"{tgt}_qs_lo": lo, f"{tgt}_qs_hi": hi,
                        f"{tgt}_skill_vs_{ref}": float(1 - x.mean() / xr.mean()) if n != ref else 0.0,
                        f"{tgt}_dqs_p": p, f"{tgt}_cov90": float(g[f"cov90_{tgt}"].mean())})
        if "stor_dir_hit" in g:
            h = g["stor_dir_hit"].dropna()
            if len(h):
                k = int(h.sum())
                lo, hi = FM.wilson(k, len(h))
                rec.update({"stor_dir_acc": k / len(h), "stor_dir_lo": lo, "stor_dir_hi": hi})
        out.append(rec)
    return pd.DataFrame(out).set_index("model"), R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "val", "exposed"], required=True)
    ap.add_argument("--models", default="", help="forecasters (default fc_hist,fc_ewma + architect wm_* via --models)")
    ap.add_argument("--trading", default="", help="trading models (default: the rule baselines); 'none' to skip")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--allow-exposed", action="store_true")
    a = ap.parse_args(argv)
    if a.split == "exposed" and not a.allow_exposed:
        raise SystemExit("refused: 2025-11-01+ is EXPOSED (prior TEST showed HH prices); pass --allow-exposed to run "
                         "it as a labelled, non-claim diagnostic")
    S = Store()
    end = {"dev": C.DEV_END, "val": C.VAL_END, "exposed": pd.Timestamp.now().normalize()}[a.split]
    X, tag = build_features(S, end, a.rebuild_features)
    L = S.labels(X.index)
    y, avail = L["ung_ret"], L["available_at"]
    lim = {"dev": C.et(C.DEV_END, 23, 59), "val": C.et(C.VAL_END, 23, 59)}.get(a.split)
    if a.split == "exposed":
        score_days = [d for d in X.index if d >= C.EXPOSED_START and np.isfinite(y.get(d, np.nan))]
    else:
        lo = DEV_SCORE_START if a.split == "dev" else C.VAL_START
        hi = C.DEV_END if a.split == "dev" else C.VAL_END
        # every label (incl. the 5-day RV) must be complete inside the split
        score_days = [d for d in X.index if lo <= d <= hi and np.isfinite(y.get(d, np.nan))
                      and C.et(C.add_bdays(d, 5), 16, 0) <= lim]
    score_days = pd.DatetimeIndex(score_days)
    out = os.path.join(EVAL, a.split if a.split != "exposed" else "EXPOSED")
    os.makedirs(out, exist_ok=True)
    label = "EXPOSED - not a claim" if a.split == "exposed" else a.split

    # ---- primary: forecasting targets
    fnames = [s for s in a.models.split(",") if s] or M.FORECASTERS
    FR, R = eval_forecasts(S, fnames, score_days, a.split, L)
    FR.to_csv(os.path.join(out, "forecast_metrics.csv"), float_format="%.5g")
    R.to_parquet(os.path.join(out, "forecast_rows.parquet"))

    # ---- secondary: trading (UNG close->close, 10 bp/side x 1/2/3)
    tnames = [] if a.trading == "none" else ([s for s in a.trading.split(",") if s] or M.RULES)
    ret = y.loc[score_days]
    pos, pups, errors = {}, {}, {}
    for n in tnames:
        try:
            if n in M.FITTED:
                pos[n], pups[n] = fitted_positions(n, X, y, avail, score_days, a.split)
            elif n in M.ZEROSHOT:
                pos[n], pups[n] = zeroshot_positions(n, S, score_days)
            elif n in M.RULES or n == "flat":
                pos[n] = rule_positions(n, X.loc[score_days])
            else:
                m = M.load(n)
                ps = []
                for d in score_days:
                    ctx = S.asof(C.decision_time(d))
                    with C.no_data_access():
                        ps.append(m.act(ctx).get("UNG", 0.0))
                        if hasattr(m, "observe"):
                            m.observe({"day": d, "labels": L.loc[d].drop("available_at").to_dict(),
                                       "available_at": L.loc[d, "available_at"]})
                pos[n] = pd.Series(ps, index=score_days)
        except Exception as e:  # noqa: BLE001
            errors[n] = repr(e)[:300]
            C.log(f"{n}: ERROR {e!r}")
    res = pd.DataFrame()
    if pos:
        srs = [FM.sharpe((p.reindex(ret.index).fillna(0) * np.expm1(ret)).to_numpy(), 1) for p in pos.values()]
        srs = [v for v in srs if np.isfinite(v)]
        var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else np.nan
        res = pd.DataFrame([{"model": n, **score(p, ret, len(pos), var_sr, pups.get(n))} for n, p in pos.items()]).set_index("model")
        res.to_csv(os.path.join(out, "trading_metrics.csv"), float_format="%.5g")
        pd.DataFrame(pos).to_parquet(os.path.join(out, "positions.parquet"))
    meta = {"split": a.split, "label": label, "score_days": [str(score_days[0].date()), str(score_days[-1].date()), int(len(score_days))],
            "clock": {"decision": "15:30 ET", "fill": "16:00 ET close"}, "cost_bp_per_side": COST_BP,
            "data_pins": data_pins(), "features_tag": tag, "errors": errors, "k_trials_trading": len(pos),
            "model_load_errors": M.LOAD_ERRORS, "run_at": C.iso(C.utcnow()), "git_head": _git()}
    json.dump(meta, open(os.path.join(out, "manifest.json"), "w"), indent=1, default=str)
    tcols = ["n_active", "dir_acc", "dir_lo", "dir_hi", "sharpe_1x", "sharpe_lo_1x", "sharpe_hi_1x", "psr_1x", "dsr_1x",
             "sharpe_2x", "sharpe_3x", "mdd_1x", "turnover"]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(f"# Alma-Live Gas harness: {label}\n\nScore days {meta['score_days']}; decision 15:30 ET (ctx = asof), "
                f"fill 16:00 ET close. Data pins tag {tag}. History is NOT the claim test (the forward paper test is).\n\n"
                "## Primary targets (quantile score = mean pinball over levels 0.05..0.95; lower is better)\n\n"
                "ret = UNG log return close(d)->close(d+1); rv5 = log sum of squared UNG returns d+1..d+5; stor = "
                "storage deviation vs naive 5-yr expectation (Bcf), origin = last decision before the first print.\n\n")
        f.write(md(FR))
        if len(res):
            f.write("\n\n## Secondary: trading (UNG, 10 bp/side; Sharpe at 1x/2x/3x costs; DSR k = %d)\n\n" % len(pos))
            f.write(md(res.reindex(columns=[c for c in tcols if c in res]), "{:.3f}"))
        if errors:
            f.write(f"\n\nErrors: {json.dumps(errors)}\n")
    print(FR.round(4).to_string())
    if len(res):
        print(res.reindex(columns=[c for c in tcols if c in res]).round(3).to_string())
    return 0


def _git():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
