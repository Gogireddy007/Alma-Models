"""
Alma-Fin walk-forward evaluation (docs/fin_prereg.md is binding).

    nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val [--market crypto|etf|all] [--models a,b]
    nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val --metrics-only
    nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val --rehearse        # TEST procedure on VAL
    nice -n 10 .venv/bin/python tools/fin/run_eval.py --split test --i-have-auditor-go   # ONCE; needs FROZEN.json

A model name "x:frozen" runs model x with observe() disabled (zero-shot / frozen twin of a continual model).
Outputs: runs/fin/<split>/<market>/<model>.npz, runs/fin/<split>/metrics.json, report.md.
"""
import os
import gc
import sys
import json
import time
import hashlib
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import harness as H     # noqa: E402
import metrics as MT    # noqa: E402
import data as DT       # noqa: E402

QS = H.QS
I9 = [QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
IMED = QS.index(0.5)
OUT = os.path.join(ROOT, "runs", "fin")
FROZEN = os.path.join(OUT, "FROZEN.json")
PREREG = os.path.join(ROOT, "docs", "fin_prereg.md")
FORECASTERS = ["majority", "rv_persist", "har", "harx", "har_roll", "garch", "lgbm", "lgbm_asset", "har_dow", "lgbm_dow", "har_intra_dow", "chronos2", "chronos_bolt_base",
               "tirex", "timestral_5m", "kronos_small"]
AGENTS = ["buy_hold", "tsmom_1", "tsmom_7", "tsmom_30", "meanrev_1", "voltgt_tsmom", "lgbm_policy"]
ETF_SKIP = {"harx", "har_intra_dow"}            # need funding / taker flow / hourly 5-min RV (absent for ETFs)
ETF_ALIAS = {"lgbm_asset": "lgbm_asset_etf"}    # per-asset twin fitted on each ETF's own DEV history
KIND = {"majority": "baseline", "rv_persist": "baseline", "har": "baseline", "harx": "baseline",
        "har_roll": "baseline", "garch": "baseline", "lgbm": "competitor", "lgbm_asset": "competitor", "har_dow": "baseline", "lgbm_dow": "competitor", "har_intra_dow": "baseline", "chronos2": "fm",
        "chronos_bolt_base": "fm", "tirex": "fm", "timestral_5m": "fm", "kronos_small": "fm",
        "buy_hold": "baseline", "tsmom_1": "baseline", "tsmom_7": "baseline", "tsmom_30": "baseline",
        "meanrev_1": "baseline", "voltgt_tsmom": "baseline", "lgbm_policy": "competitor"}
CONTAMINATED = {"timestral_5m": "released 2026-10-15; training data may include TEST"}
DATA_PINS = {
    "data/fin/crypto_daily.parquet": "1b5c9ee8cfe60bcc618d0a742027ff4318c6947b82e59895b9fb5af8af574904",
    "data/fin/crypto_daily_SEALED.parquet": "6429949adf5aadc18c230845355b69f375cafdef3f5f39976a62a581a2258cc4",
    "data/fin/crypto_hourly.parquet": "5bdccc569c4638e995469423fc233285afc95f94cabe9345049354c8499ac084",
    "data/fin/crypto_hourly_SEALED.parquet": "b72f5d60d1abddff7c1c9819e973e464c346e6762160422000b0b3a5f154ac96",
    "data/fin/etf_daily.parquet": "964a1f50f5f74aeb368b6e48368d51bdbcd0f1ede0f1b3e487a688f10775f1b0",
    "data/fin/etf_daily_SEALED.parquet": "9a720627b71dc0b9b2e68580f49bbab946ef652399b79f37dc20d0b000c5dc06",
    "data/fin/universe.json": "361e0373e86822099dfcbe5e87f6849e240dc431d8ba9042178294fb0a9e92b0",
}   # prereg section 12 (before any score)
RHO = 0.512               # cross-coin P1 label correlation, DEV 2022-07..2025-06, 24 coins (runs/fin/label_corr.json); N_eff only
RUN_DIR = {}
# P2 (trading) window end per split: the 2026-10 funding archive is not published before TEST and the Binance REST
# endpoint is geo-restricted here, so TEST trading P&L is scored on decision days <= 2026-09-29 (the 2026-09-30 position pays the 2026-10-01 00:00 settlement).
P2_LAST = {"dev": "2025-06-30", "val": "2025-10-31", "test": "2026-09-29"}


def split_dir(split):
    return RUN_DIR.get(split, os.path.join(OUT, split))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def frozen():
    return json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}


def claim_models():
    return list(frozen().get("claim_models", frozen().get("models", [])))


def kind_of(mn):
    base = mn.split(":")[0]
    if base in KIND:
        return KIND[base]
    return "candidate" if mn in claim_models() else "model"


# --------------------------------------------------------------------------- forecasting
def forecast(split, market, names, strict=False, force=False):
    import models as M
    P = H.Panel(market, allow_test=(split == "test"))
    days = P.split_days(split)
    sdir = os.path.join(split_dir(split), market)
    os.makedirs(sdir, exist_ok=True)
    for mn in names:
        if market == "etf" and mn in ETF_SKIP:
            continue
        dst = os.path.join(sdir, f"{mn.replace(':', '__')}.npz")
        if strict and os.path.exists(dst):
            raise SystemExit(f"strict mode: {dst} exists")
        if os.path.exists(dst) and not force:
            z = np.load(dst)
            if np.array_equal(z["days"], days):
                print(f"[{split}/{market}] {mn}: cached", flush=True)
                continue
        base, frozen_flag = mn.split(":")[0], mn.endswith(":frozen")
        try:
            model = M.get(ETF_ALIAS.get(base, base) if market == "etf" else base)
        except Exception as e:
            if strict:
                raise
            print(f"[{split}/{market}] {mn}: SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        t0 = time.time()
        res = H.run(P, model, days, online=False if frozen_flag else None,
                    log=lambda s: print(f"[{split}/{market}] {mn}{s}", flush=True))
        meta = dict(getattr(model, "meta", {}), model=mn, split=split, market=market, seconds=round(time.time() - t0, 1),
                    n_days=len(days), online=res["online"], kind=res["kind"],
                    created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        arrs = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
        np.savez(dst + ".tmp.npz", **arrs, symbols=np.array(res["symbols"]),
                 dates=P.days[days].astype("datetime64[D]").astype(str),
                 meta=np.array(json.dumps(meta, default=str)))
        os.replace(dst + ".tmp.npz", dst)
        print(f"[{split}/{market}] {mn}: done {time.time() - t0:.0f}s, updates {int(res['updates'].sum())}", flush=True)
        del model, res
        gc.collect()


def load_preds(split, market):
    sdir = os.path.join(split_dir(split), market)
    out = {}
    if os.path.isdir(sdir):
        for f in sorted(os.listdir(sdir)):
            if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                z = np.load(os.path.join(sdir, f), allow_pickle=False)
                out[f[:-4].replace("__", ":")] = {k: z[k] for k in z.files}
    return out


# --------------------------------------------------------------------------- scoring
def dev_majority(P):
    import models as M
    return float(M.majority().meta["dev_up_fraction"])


def direction(z, labs_ref, maj):
    """Harness-derived P1 direction: p_up > 0.5 if present, else median > reference; ties -> DEV majority."""
    if np.isfinite(z["p_up"]).any():
        p = z["p_up"]
        d = np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, float(maj > 0.5)))
        return np.where(np.isfinite(p), d, np.nan), "p_up"
    med = z["q"][:, :, IMED]
    d = np.where(med > labs_ref, 1.0, np.where(med < labs_ref, 0.0, float(maj > 0.5)))
    return np.where(np.isfinite(med) & np.isfinite(labs_ref), d, np.nan), "median"


def score(split):
    res = {"split": split, "markets": {}}
    for market in ("crypto", "etf"):
        preds = load_preds(split, market)
        if not preds:
            continue
        P = H.Panel(market, allow_test=(split == "test"))
        maj = dev_majority(P) if market == "crypto" else None
        groups = {"all": P.symbols} if market == "etf" else \
            {"train": [s for s in P.symbols if P.group[s] == "train"],
             "heldout": [s for s in P.symbols if P.group[s] == "heldout"], "all": P.symbols}
        R = {"forecasters": {}, "agents": {}, "pairwise": {}, "n_symbols": {g: len(v) for g, v in groups.items()}}
        any_days = next(iter(preds.values()))["days"]
        L = [P.labels(int(i)) for i in any_days]
        Y = np.stack([l["y_vol"] for l in L])
        LR = np.stack([l["logrv"] for l in L])
        REF = np.stack([l["ref"] for l in L])
        YR = np.stack([l["y_ret"] for l in L])
        R["base_rate"] = {g: float(np.nanmean(Y[:, [P.symbols.index(s) for s in v]])) for g, v in groups.items()}
        if maj is not None:
            R["dev_majority"] = maj
        series = {}                  # (target, group) -> model -> per-day value
        for mn, z in preds.items():
            if not np.array_equal(z["days"], any_days):
                print(f"WARNING {market}/{mn}: different day set; skipped in scoring", flush=True)
                continue
            js_all = [P.symbols.index(s) for s in z["symbols"]]
            if str(z.get("kind", "")) == "agent" or "pos" in z:
                keep = P.days[z["days"]] <= np.datetime64(P2_LAST[split], "ns")
                for g, syms in groups.items():
                    js = [z["symbols"].tolist().index(s) for s in syms]
                    out = {"p2_last_day": P2_LAST[split]}
                    for mult in (1.0, 2.0):
                        pl = H.pnl(P, z["days"][keep], z["pos"][keep][:, js], symbols=syms, cost_mult=mult)
                        fz = frozen()
                        summ = MT.trading_summary(pl["net"], pl["turnover"], P.periods, fz.get("K"),
                                                  fz.get("var_sharpe_trials"))
                        summ.update(gross_mean=float(pl["gross"].mean()), funding_mean=float(pl["funding"].mean()),
                                    cost_mean=float(pl["costs"].mean()))
                        out["base" if mult == 1 else "cost_x2"] = summ
                        if mult == 1:
                            series.setdefault(("P2", g), {})[mn] = pl["net"]
                    out["updates"] = int(z["updates"].sum())
                    R["agents"].setdefault(mn, {})[g] = out
                continue
            for g, syms in groups.items():
                jz = [z["symbols"].tolist().index(s) for s in syms]
                jp = [P.symbols.index(s) for s in syms]
                y, lr, ref, yr = Y[:, jp], LR[:, jp], REF[:, jp], YR[:, jp]
                zz = {k: z[k][:, jz] for k in ("p_up", "q", "p_ret_up")}
                out = {"updates": int(z["updates"].sum())}
                d, how = direction(zz, ref, maj if maj is not None else 0.5)
                # RT-13 (prereg A4): the scored set is every coin-day with a finite label; a missing prediction
                # counts as the DEV-majority direction, so abstention cannot raise accuracy; coverage is reported
                has_pred = bool(np.isfinite(d).any())
                lab_ok = np.isfinite(y)
                cov_p1 = float((np.isfinite(d) & lab_ok).sum() / max(1, lab_ok.sum()))
                d = np.where(lab_ok & ~np.isfinite(d), float((maj if maj is not None else 0.5) > 0.5), d)
                ok = np.isfinite(d) & lab_ok
                if has_pred and ok.any():
                    hit = (d == y).astype(float)
                    num = np.where(ok, hit, 0).sum(1)
                    den = ok.sum(1)
                    dd = den > 0
                    dayhit = num[dd] / den[dd]
                    k, n = int(num.sum()), int(den.sum())
                    m = len(syms)
                    neff = n / (1 + (m - 1) * RHO) if RHO is not None else None
                    _, st, p_half = MT.dm_hln(0.5 - dayhit, 1)
                    pt = MT.pesaran_timmermann(d[ok], y[ok])
                    out["P1"] = {"acc": k / n, "n": n, "coverage": cov_p1, "n_days": int(dd.sum()), "source": how,
                                 "wilson_coinday": MT.wilson(k, n), "binom_p_coinday": MT.binom_p(k, n),
                                 "n_eff": neff, "wilson_eff": MT.wilson_eff(k, n, neff) if neff else None,
                                 "boot7_ci": MT.boot_ratio_ci(num[dd], den[dd]), "p_vs_half_dm_daily": p_half,
                                 "pt_stat": pt[0], "pt_p_onesided": pt[1], "pred_up_rate": float(np.mean(d[ok]))}
                    series.setdefault(("P1", g), {})[mn] = (np.where(dd, num / np.maximum(den, 1), np.nan),
                                                            np.where(ok, 1 - hit, np.nan))
                    if np.isfinite(zz["p_up"]).any():
                        pu = np.clip(zz["p_up"], 1e-6, 1 - 1e-6)
                        okp = np.isfinite(pu) & np.isfinite(y)
                        out["P1"]["brier"] = float(np.mean((pu[okp] - y[okp]) ** 2))
                        out["P1"]["logloss"] = float(-np.mean(y[okp] * np.log(pu[okp]) + (1 - y[okp]) * np.log(1 - pu[okp])))
                q = zz["q"]
                okq = np.isfinite(lr) & np.isfinite(q[:, :, I9]).all(-1)
                cov_p3 = float(okq.sum() / max(1, np.isfinite(lr).sum()))
                if okq.any():
                    lv = [QS[i] for i in I9]
                    ql = np.full(lr.shape, np.nan)
                    ql[okq] = MT.pinball(lr[okq], q[:, :, I9][okq], lv)
                    ae = np.abs(lr - q[:, :, IMED])
                    cov = ((lr >= q[:, :, QS.index(0.1)]) & (lr <= q[:, :, QS.index(0.9)]))[okq]
                    out["P3"] = {"qs": float(np.nanmean(ql)), "crps_approx": float(2 * np.nanmean(ql)),
                                 "mae_median": float(np.nanmean(np.where(okq, ae, np.nan))), "cov80": float(cov.mean()),
                                 "n": int(okq.sum()), "coverage": cov_p3}
                    series.setdefault(("P3", g), {})[mn] = ql          # [days, assets]; pairwise on the intersection
                if np.isfinite(zz["p_ret_up"]).any():
                    pr = zz["p_ret_up"]
                    dr = np.where(pr > 0.5, 1.0, 0.0)
                    okr = np.isfinite(pr) & np.isfinite(yr)
                    hr = (dr == yr)[okr]
                    out["P4"] = {"acc": float(hr.mean()), "n": int(okr.sum()), "wilson_coinday": MT.wilson(int(hr.sum()), int(okr.sum()))}
                R["forecasters"].setdefault(mn, {})[g] = out
        R["pairwise"] = pairwise(series, P.periods)
        add_holm(R)
        res["markets"][market] = R
    return res


def pairwise(series, periods):
    out = {}
    for (tgt, g), ser in series.items():
        names = list(ser)
        pw = {}
        for a in names:
            for b in names:
                if a == b:
                    continue
                if tgt == "P1":
                    da, la = ser[a]
                    db, lb = ser[b]
                    ok = np.isfinite(da) & np.isfinite(db)
                    if ok.sum() < 10:
                        continue
                    diff = da[ok] - db[ok]                        # daily accuracy difference (a - b > 0: a better)
                    md, st, p = MT.dm_hln(-diff, 1)
                    lo, hi = MT.boot_mean_ci(diff)
                    both = np.isfinite(la) & np.isfinite(lb)
                    n01, n10, pm = MT.mcnemar_exact(la[both] == 1, lb[both] == 1)
                    pw[f"{a}|{b}"] = {"n_days": int(ok.sum()), "mean_diff": float(diff.mean()), "p": p, "boot_lo": lo,
                                      "boot_hi": hi, "a_better": bool(diff.mean() > 0),
                                      "mcnemar_coinday_p_supplementary": pm, "n_a_wrong_b_right": n01,
                                      "n_a_right_b_wrong": n10}
                elif tgt == "P3":
                    both = np.isfinite(ser[a]) & np.isfinite(ser[b])    # common coin-days only (RT-13)
                    with np.errstate(all="ignore"):
                        xa = np.nanmean(np.where(both, ser[a], np.nan), 1)
                        xb = np.nanmean(np.where(both, ser[b], np.nan), 1)
                    ok = np.isfinite(xa) & np.isfinite(xb)
                    if ok.sum() < 10:
                        continue
                    diff = xa[ok] - xb[ok]                        # loss: a - b < 0 means a better
                    md, st, p = MT.dm_hln(diff, 1)
                    lo, hi = MT.boot_mean_ci(diff)
                    pw[f"{a}|{b}"] = {"n_days": int(ok.sum()), "mean_diff": float(diff.mean()), "p": p, "boot_lo": lo,
                                      "boot_hi": hi, "a_better": bool(diff.mean() < 0)}
                else:                                             # P2: Sharpe difference, paired block bootstrap
                    d0, lo, hi, p = MT.sharpe_diff_test(ser[a], ser[b], periods)
                    pw[f"{a}|{b}"] = {"sharpe_diff": d0, "boot_lo": lo, "boot_hi": hi, "p": p,
                                      "a_better": bool(d0 > 0) if np.isfinite(d0) else False}
        out[f"{tgt}/{g}"] = pw
    return out


def add_holm(R):
    claim = claim_models()
    for key, pw in R["pairwise"].items():
        names = sorted({k.split("|")[0] for k in pw})
        for a in names:
            if a not in claim:
                continue
            keys = [k for k in pw if k.startswith(a + "|") and not CONTAMINATED.get(k.split("|")[1].split(":")[0])]
            for k, pa in zip(keys, MT.holm([pw[k]["p"] for k in keys])):
                pw[k]["p_holm_own"] = float(pa)
                pw[k]["beats"] = bool(pw[k]["a_better"] and pa < 0.05 and (pw[k]["boot_lo"] > 0 or pw[k]["boot_hi"] < 0))
    # RT-13: a claim model whose coverage of the scored set is < 99% on a target/group gets no "beats" there
    for key, pw in R["pairwise"].items():
        tgt, g = key.split("/")
        for k, v in pw.items():
            a = k.split("|")[0]
            if a in claim and tgt in ("P1", "P3"):
                cv = R["forecasters"].get(a, {}).get(g, {}).get(tgt, {}).get("coverage", 0.0)
                if cv < 0.99:
                    v["beats"] = False
                    v["coverage_void"] = cv


def fmt_dsr(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.3f}"


def report(res, path):
    L = [f"# Alma-Fin {res['split'].upper()} report", "", f"commit {res.get('commit')}, scored {res.get('scored')}", ""]
    for market, R in res["markets"].items():
        L += [f"## {market}", "", f"base rate (y_vol = 1): {R['base_rate']}" + (f"; DEV majority {R['dev_majority']:.3f}" if 'dev_majority' in R else ""), ""]
        if R["forecasters"]:
            for g in R["n_symbols"]:
                L += [f"### Forecasters, group {g} ({R['n_symbols'][g]} assets)", "",
                      "| model | kind | P1 acc | 95% CI (7d block) | p vs 0.5 (DM daily) | PT p | brier | P3 QS | MAE med | cov80 | P4 acc | updates |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                rows = sorted(R["forecasters"].items(), key=lambda kv: -kv[1].get(g, {}).get("P1", {}).get("acc", 0))
                for mn, v in rows:
                    o = v.get(g, {})
                    p1, p3, p4 = o.get("P1", {}), o.get("P3", {}), o.get("P4", {})
                    f = lambda x, d=3: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
                    ci = p1.get("boot7_ci")
                    L.append(f"| {mn}{' (CONTAMINATED)' if mn in CONTAMINATED else ''} | {kind_of(mn)} | {f(p1.get('acc'))} | "
                             f"{'-' if not ci else f'{ci[0]:.3f}-{ci[1]:.3f}'} | {f(p1.get('p_vs_half_dm_daily'), 4)} | "
                             f"{f(p1.get('pt_p_onesided'), 4)} | {f(p1.get('brier'))} | {f(p3.get('qs'), 4)} | "
                             f"{f(p3.get('mae_median'))} | {f(p3.get('cov80'))} | {f(p4.get('acc'))} | {o.get('updates', 0)} |")
                L.append("")
        if R["agents"]:
            for g in R["n_symbols"]:
                L += [f"### Agents (net of costs), group {g}", "",
                      "| model | Sharpe | 95% CI | PSR(0) | DSR | ann ret | ann vol | maxDD | turnover | Sharpe @2x cost |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
                for mn, v in sorted(R["agents"].items(), key=lambda kv: -np.nan_to_num(kv[1][g]["base"]["sharpe"], nan=-9)):
                    b, c = v[g]["base"], v[g]["cost_x2"]
                    L.append(f"| {mn} | {b['sharpe']:.2f} | {b['sharpe_boot_lo']:.2f}..{b['sharpe_boot_hi']:.2f} | "
                             f"{b['psr0']:.3f} | {fmt_dsr(b['dsr'])} | "
                             f"{b['ann_return']:.3f} | {b['ann_vol']:.3f} | {b['max_drawdown']:.3f} | {b['turnover']:.3f} | {c['sharpe']:.2f} |")
                L.append("")
    L += ["P1 inference: DM-HLN (h=1) on the daily cross-asset mean accuracy and a 7-day block bootstrap; coin-day",
          "Wilson/binomial ignore cross-sectional correlation and are descriptive. VAL is for debugging, not claims.", ""]
    open(path, "w").write("\n".join(L))


# --------------------------------------------------------------------------- strict mode
def preflight(models):
    import models as M
    if M.LOAD_ERRORS:
        raise SystemExit(f"preflight: architect module import errors: {M.LOAD_ERRORS}")
    fz = frozen()
    metas = {}
    for mn in models:
        p = M.get(mn.split(":")[0])
        meta = dict(getattr(p, "meta", {}))
        ck = meta.get("ckpt")
        for path, sh in (fz.get("sha256") or {}).items():
            if ck == path and meta.get("ckpt_sha256") != sh:
                raise SystemExit(f"preflight: {mn} checkpoint {ck} sha256 != FROZEN.json")
        if mn in fz.get("models", []):          # RT-6: every frozen model must carry a pinned, matching checkpoint
            for key, ck_key in (("ckpt", "ckpt_sha256"), ("wm_ckpt", "wm_ckpt_sha256")):
                if key == "wm_ckpt" and key not in meta:
                    continue
                pth, sh = meta.get(key), meta.get(ck_key)
                want = (fz.get("sha256") or {}).get(pth)
                if not pth or not sh or not want or sh != want:
                    raise SystemExit(f"preflight: frozen model {mn}: {key}={pth} {ck_key}={sh} not matching FROZEN.json sha256 {want}")
                if sha_file(os.path.join(ROOT, pth)) != want:
                    raise SystemExit(f"preflight: {pth} on disk != FROZEN.json sha256")
        metas[mn] = meta
        del p
    return metas


def check_pins():
    if not DATA_PINS:
        raise SystemExit("strict mode refused: run_eval.DATA_PINS not set")
    for rel, sh in DATA_PINS.items():
        if sha_file(os.path.join(ROOT, rel)) != sh:
            raise SystemExit(f"strict mode refused: {rel} sha256 != pin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "val", "test"])
    ap.add_argument("--market", default="all", choices=["crypto", "etf", "all"])
    ap.add_argument("--models", default="")
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--i-have-auditor-go", action="store_true")
    a = ap.parse_args()
    strict = (a.split == "test" or a.rehearse) and not a.metrics_only
    markets = ["crypto", "etf"] if a.market == "all" else [a.market]
    if a.rehearse:
        if a.split != "val":
            raise SystemExit("--rehearse only with --split val")
        RUN_DIR["val"] = os.path.join(OUT, "rehearsal_val")
    run_rec, rp = None, None
    if a.split == "test" and not a.metrics_only:
        if not a.i_have_auditor_go:
            raise SystemExit("TEST refused: pass --i-have-auditor-go (only after the auditor's GO is relayed)")
        os.environ["ALMA_FIN_TEST_GO"] = "1"
    if strict:
        rdir = split_dir(a.split)
        if a.split == "test" and not os.path.exists(FROZEN):
            raise SystemExit("strict mode refused: runs/fin/FROZEN.json missing")
        if a.split == "test":
            tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", "runs/fin/FROZEN.json"],
                                     capture_output=True).returncode == 0
            if not tracked:
                raise SystemExit("strict mode refused: runs/fin/FROZEN.json is not committed")
        rp = os.path.join(rdir, "RUN.json")
        if os.path.exists(rp):
            raise SystemExit(f"strict mode refused: already run ({rp})")
        if os.path.isdir(rdir) and any(f.endswith(".npz") for _, _, fs in os.walk(rdir) for f in fs):
            raise SystemExit(f"strict mode refused: {rdir} holds forecast files")
        if a.models or a.force or a.market != "all":
            raise SystemExit("strict mode refused: pre-registered model list and both markets only")
        check_pins()
        fz = frozen() or {"models": [], "note": "rehearsal before freeze"}
        models = FORECASTERS + AGENTS + [m for m in fz.get("models", []) if m not in FORECASTERS + AGENTS]
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=all", "tools/fin",
                                "docs/fin_prereg.md", "runs/fin/FROZEN.json"], capture_output=True, text=True).stdout.strip()
        if dirty and a.split == "test":
            raise SystemExit(f"strict mode refused: uncommitted changes:\n{dirty}")
        metas = preflight(models)
        run_rec = {"status": "started", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_head(),
                   "dirty": dirty, "frozen": fz, "frozen_sha256": sha_file(FROZEN) if os.path.exists(FROZEN) else None,
                   "data_pins": DATA_PINS, "models": models, "model_meta": metas, "split": a.split,
                   "rehearsal": a.rehearse, "argv": sys.argv,
                   "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise SystemExit("strict mode refused: set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
        os.makedirs(rdir, exist_ok=True)
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    else:
        models = [m for m in a.models.split(",") if m] or (FORECASTERS + AGENTS)
    try:
        if not a.metrics_only:
            for mk in markets:
                forecast(a.split, mk, models, strict=strict, force=a.force)
        res = score(a.split)
        res.update(commit=git_head(), scored=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rehearsal=a.rehearse, rho=RHO,
                   data_sha256={os.path.relpath(f, ROOT): sha_file(f) for k, f in DT.FILES.items()
                                if os.path.exists(f) and not k.endswith("SEALED")})
        path = os.path.join(split_dir(a.split), "metrics.json")
        json.dump(res, open(path, "w"), indent=1, default=float)
        report(res, os.path.join(split_dir(a.split), "report.md"))
    except BaseException as e:
        if run_rec is not None:
            import traceback
            run_rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
            json.dump(run_rec, open(rp, "w"), indent=1, default=str)
        raise
    if run_rec is not None:
        rdir = split_dir(a.split)
        hashes = {os.path.relpath(os.path.join(dp, f), ROOT): sha_file(os.path.join(dp, f))
                  for dp, _, fs in os.walk(rdir) for f in sorted(fs) if f.endswith(".npz") or f == "metrics.json"}
        json.dump(hashes, open(os.path.join(rdir, "hashes.json"), "w"), indent=1, sort_keys=True)
        run_rec.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       hashes_sha256=sha_file(os.path.join(rdir, "hashes.json")))
        json.dump(run_rec, open(rp, "w"), indent=1, default=str)
    print(f"metrics -> {path}")


if __name__ == "__main__":
    main()
