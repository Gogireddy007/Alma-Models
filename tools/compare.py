"""
Paired statistical comparison of forecasting runs, on the old prepare.evaluate suite or on the
leak-audited bench (tools/bench.py, TSAR_BENCH=1). Read docs/eval_rigor.md for the decision rules
these numbers feed; the bench claim gate (one gate, "Gate G") is defined there.

A RUN SPEC is any of
    runs/q_base.log           train.py log (the per-dataset / per-task table printed by evaluate)
    path/to/file.csv          a per-dataset CSV (prepare.evaluate) or per-task CSV (bench.evaluate)
    runs/dumps/base_s0        a TSAR_DUMP directory from tools/run.py
    chronos2                  a baseline name. Old suite: baselines/dumps/<name>/ if present, else
                              baselines/per_dataset/<name>_<split>.csv. Bench: baselines/bench/dumps/<name>/
                              if present, else baselines/bench/per_task/<name>_<split>.csv
    sn                        Seasonal Naive itself (wql_rel = 1 everywhere)
    0.6788                    a bare number (headline only; usable by `seeds`, not by `pair`)
Several seeds of one config: join specs with commas, or use a glob in quotes:
    "runs/base_s*.log"   or   runs/base_s0.log,runs/base_s1.log,runs/base_s2.log

Bench or old suite? Decided from the files given (a bench table has a `family` column). Mixing the
two is refused. With only baseline names, pass --bench (or set TSAR_BENCH=1).

Commands
    pair  A B          paired comparison of A against B (ratio < 1 means A is better)
    claim A B [B2 ..]  same, then a PASS/FAIL verdict (exit code 0 = PASS).
                       old suite: one B, eval_rigor R4.  bench: Gate G against every B, both metrics,
                       intersection-union verdict + Holm-adjusted individual statements.
    seeds A            mean +- std over seeds of the headline metric, per-dataset spread
    board              every baseline and the given runs, ranked, with ratio CIs vs a reference

Common options: --split val|test|holdout  --metric wql_rel|mase_rel  --boot 10000
                --datasets a,b,c | --zs4 | --exclude-zs4    --seed 0   --bench
claim (bench):  --set final (the pre-registered baseline set)  --candidates K (Bonferroni across
                claim-bearing candidates)  --min-seeds N (default 3; the final claim uses 5)

Examples
    uv run tools/compare.py pair runs/q_latent.log runs/q_base.log
    uv run tools/compare.py claim "runs/base_s*.log" timestral_28m
    uv run tools/compare.py claim "runs/dumps/final_s*" --set final --min-seeds 5
    uv run tools/compare.py seeds "runs/base_s*.log"
    uv run tools/compare.py board --bench --ref chronos2
"""
import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BL_DIR = os.path.join(ROOT, "baselines")
BENCH_BL_DIR = os.path.join(BL_DIR, "bench")
ZS4 = ["monash_tourism_monthly", "monash_traffic", "nn5", "m4_quarterly"]   # train.py ZS_SETS
# Pre-registered baseline set B for the final bench claim (docs/audit_v2.md section 5.1).
# Gate G baseline set = the user's full list (audit_v3 H-5). Names as in baselines/bench/per_task/.
# chronos_bolt_base has no scores yet: a --set final claim refuses to run until it does (--allow-incomplete
# runs the rest and can never PASS). AutoETS diverges on some VAL tasks (SZ_TAXI_1H wql_rel ~2e10); it is
# kept as-is (no fixing of baselines) and its divergent tasks are printed with every comparison.
FINAL_SET = ["chronos2", "chronos_bolt_base", "timestral_209m", "timestral_28m", "timestral_5m", "tirex",
             "auto_ets", "auto_theta", "seasonal_naive_pi"]
DIVERGE = 10.0                 # a baseline task with metric > this (x Seasonal Naive) is reported as divergent
TIE = 1e-3                     # |log ratio| below this is a tie (tasks and families alike)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_TABLE_HDR = re.compile(r"^\s*dataset\s+wql\s+mase\s+wql_rel")             # prepare.evaluate
_BENCH_HDR = re.compile(r"^\s*dataset\s+family\s+n_forecasts\b")            # bench.evaluate
_STR_COLS = {"dataset", "family", "freq"}


def parse_log(path):
    """Return {split: per-dataset DataFrame} from a train.py log, old suite or bench. A test run
    prints the test table, then the summary, then (old suite) the holdout table; a bench eval-only
    test session prints the val table, then the test table."""
    lines = open(path, errors="replace").read().splitlines()
    tables, i = [], 0
    while i < len(lines):
        if _TABLE_HDR.match(lines[i]) or _BENCH_HDR.match(lines[i]):
            cols = lines[i].split()
            rows, i = [], i + 1
            while i < len(lines) and len(lines[i].split()) == len(cols):
                parts = lines[i].split()
                try:
                    rows.append([p if c in _STR_COLS else float(p) for c, p in zip(cols, parts)])
                except ValueError:
                    break
                i += 1
            tables.append((i, pd.DataFrame(rows, columns=cols)))
        else:
            i += 1
    out = {}
    for end, df in tables:
        split = None
        for ln in lines[end:end + 20]:          # the summary line right after names the split
            m = re.match(r"^(val|test|holdout)_wql_rel:", ln) or re.match(r"^(val|test): wql_rel=", ln)
            if m:
                split = m.group(1)
                break
        if split is None:
            split = "holdout" if "test" in out else "val"
        out[split] = df
    return out


def spec_kind(path, split):
    """True = bench table, False = old-suite table, None = cannot tell (missing split / not a run)."""
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, f"per_task_{split}.csv")):
            return True
        if os.path.exists(os.path.join(path, f"per_dataset_{split}.csv")):
            return False
        return None
    if path.endswith(".csv"):
        return "family" in pd.read_csv(path, nrows=0).columns
    try:
        txt = open(path, errors="replace").read().splitlines()
    except OSError:
        return None
    if any(_BENCH_HDR.match(ln) for ln in txt):
        return True
    if any(_TABLE_HDR.match(ln) for ln in txt):
        return False
    return None


def _expand(text):
    out = []
    for part in text.split(","):
        hits = sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else [part]
        if not hits:
            sys.exit(f"glob {part!r} matched nothing")
        out += hits
    return out


def detect_bench(texts, split, flag=False):
    """Bench mode if any given file is a bench table; refuse to mix bench and old-suite runs."""
    kinds = {spec_kind(p, split) for t in texts for p in _expand(t) if os.path.exists(p)} - {None}
    if kinds == {True, False}:
        sys.exit("refusing to compare bench (per-task, has `family`) runs with old-suite (per-dataset) runs")
    if kinds:
        bench = kinds.pop()
        if flag and not bench:
            sys.exit("--bench given, but the runs are old-suite tables")
        return bench
    return bool(flag or int(os.environ.get("TSAR_BENCH", 0) or 0))


class Run:
    """One scored run: per-dataset (per-task) table + (optionally) per-series pinball numerators."""

    def __init__(self, spec, split, bench=False):
        self.spec, self.split, self.bench = spec, split, bench
        self.df, self.preds, self.headline = None, None, None
        if re.fullmatch(r"[0-9.]+", spec):
            self.headline = float(spec)
            return
        if spec.lower() in ("sn", "seasonal_naive"):
            self.preds = "sn"
            self.df = None                       # filled lazily once we know the datasets
            return
        path = spec if os.path.exists(spec) else None
        if path is None:                         # baseline name
            if bench:
                d = os.path.join(BENCH_BL_DIR, "dumps", spec)
                c = os.path.join(BENCH_BL_DIR, "per_task", f"{spec}_{split}.csv")
                path = d if os.path.exists(os.path.join(d, f"per_task_{split}.csv")) else c
                avail = os.path.join(BENCH_BL_DIR, "per_task")
            else:
                d = os.path.join(BL_DIR, "dumps", spec)
                c = os.path.join(BL_DIR, "per_dataset", f"{spec}_{split}.csv")
                path = d if os.path.isdir(d) and os.path.exists(os.path.join(d, f"per_dataset_{split}.csv")) else c
                avail = os.path.join(BL_DIR, "per_dataset")
            if not os.path.exists(path):
                have = sorted(f[:-len(f"_{split}.csv")] for f in os.listdir(avail) if f.endswith(f"_{split}.csv")) \
                    if os.path.isdir(avail) else []
                sys.exit(f"cannot resolve run spec {spec!r} (split={split}, {'bench' if bench else 'old suite'}). "
                         f"Baselines available: {have}")
        if os.path.isdir(path):
            tab = os.path.join(path, f"per_task_{split}.csv")
            if not os.path.exists(tab):
                tab = os.path.join(path, f"per_dataset_{split}.csv")
            if not os.path.exists(tab):
                sys.exit(f"{path}: no per_task_{split}.csv / per_dataset_{split}.csv")
            self.df = pd.read_csv(tab)
            npz = os.path.join(path, f"preds_{split}.npz")
            if os.path.exists(npz) and not bench:     # series-level stats exist for the old suite only
                self.preds = dict(np.load(npz))
        elif path.endswith(".csv"):
            self.df = pd.read_csv(path)
        else:
            tabs = parse_log(path)
            if split not in tabs:
                sys.exit(f"{path}: no per-dataset table for split={split} (found {list(tabs)})")
            self.df = tabs[split]
        is_bench = "family" in self.df.columns
        if is_bench != bench:
            sys.exit(f"{spec}: is a {'bench' if is_bench else 'old-suite'} table, but this comparison is "
                     f"{'bench' if bench else 'old-suite'}; refusing to mix")
        self.df = self.df.set_index("dataset")

    @property
    def has_series(self):
        return self.preds is not None and not self.bench

    def per_dataset(self, metric, datasets):
        if self.preds == "sn" and self.df is None:
            return pd.Series(1.0, index=datasets)
        return self.df.loc[datasets, metric]

    def gmean(self, metric, datasets=None):
        if self.headline is not None:
            return self.headline
        ds = datasets if datasets is not None else list(self.df.index)
        return float(np.exp(np.log(np.clip(self.per_dataset(metric, ds), 1e-6, None)).mean()))


def load_specs(text, split, bench=False):
    return [Run(s, split, bench) for s in _expand(text)]

# ---------------------------------------------------------------------------
# Per-series scores (old suite only: needs data/suite.pkl; mirrors prepare._score exactly)
# ---------------------------------------------------------------------------

_SUITE = {}


def _windows(name, split):
    import prepare as P
    if not _SUITE:
        _SUITE.update({d["name"]: d for d in P.load_suite()})
    ds = _SUITE[name]
    ctxs, tgts = P._windows(ds, split)
    return ds, ctxs, tgts


def series_pinball(q, name, split):
    """Per-series pinball numerator (sum over quantiles and horizon). Dataset WQL is
    2 * sum_i num_i / (Q * sum|y|), so ratios of WQL between two models on the same series
    set are ratios of sums of these numerators."""
    import prepare as P
    ds, ctxs, tgts = _windows(name, split)
    if isinstance(q, str):                   # seasonal naive
        h, m = ds["horizon"], ds["season"]
        sn = np.stack([P._seasonal_naive(c, h, m) for c in ctxs])
        q = np.repeat(sn[:, None, :], len(P.QUANTILES), axis=1)
    q = np.sort(np.nan_to_num(np.asarray(q, np.float32), nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    mask = ~np.isnan(tgts)
    y = np.nan_to_num(tgts)
    err = y[:, None, :] - q
    Qs = P.QUANTILES[None, :, None]
    pin = np.maximum(Qs * err, (Qs - 1) * err) * mask[:, None, :]
    return pin.sum(axis=(1, 2)).astype(np.float64)


_SN_CACHE = {}


def series_scores(run, name, split):
    if run.preds == "sn":
        key = (name, split)
        if key not in _SN_CACHE:
            _SN_CACHE[key] = series_pinball("sn", name, split)
        return _SN_CACHE[key]
    return series_pinball(run.preds[name], name, split)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def gm(x):
    return float(np.exp(np.mean(x)))


def seed_mean_log(runs, metric, datasets):
    """[D] per-dataset log metric averaged over seeds (= log of the geometric mean)."""
    return np.mean([np.log(np.clip(r.per_dataset(metric, datasets).to_numpy(float), 1e-6, None))
                    for r in runs], axis=0)


def _boot_p(boots):
    """Two-sided percentile-bootstrap p-value for H0: ratio = 1 (consistent with the CI: the
    (1 - a) interval excludes 1 exactly when p < a)."""
    B = len(boots)
    lo, hi = (boots >= 1).sum(), (boots <= 1).sum()
    return float(min(1.0, 2 * (min(lo, hi) + 1) / (B + 1)))


def _ci(boots, level):
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _check_parity(tables, order):
    """Bench: every run must have been scored on identical windows (same counts and horizons)."""
    for col in ("n_forecasts", "horizon"):
        have = [r for r in tables if col in r.df.columns]
        if len(have) < 2:
            continue
        ref = have[0].df.loc[order, col].to_numpy(float)
        for r in have[1:]:
            v = r.df.loc[order, col].to_numpy(float)
            if not np.array_equal(ref, v):
                bad = [d for d, x, y in zip(order, ref, v) if x != y]
                sys.exit(f"{r.spec} vs {have[0].spec}: different {col} on {bad[:5]}: not the same windows")


def compare(A, B, split, metric="wql_rel", n_boot=10000, datasets=None, seed=0, quiet=False, level=0.95):
    """A, B: lists of Run (seeds). Returns a dict of statistics for ratio A/B."""
    rng = np.random.default_rng(seed)
    for r in A + B:
        if r.headline is not None:
            sys.exit("pair/claim need per-dataset scores; a bare number only works with `seeds`")
    tables = [r for r in A + B if r.df is not None]
    common = set.intersection(*[set(r.df.index) for r in tables]) if tables else None
    if common is None:
        sys.exit("need at least one side with a per-dataset table")
    order = [d for d in tables[0].df.index if d in common]
    if datasets:
        missing = [d for d in datasets if d not in common]
        if missing:
            sys.exit(f"datasets not in both runs: {missing}")
        order = [d for d in order if d in datasets]
    dropped = sorted(set.union(*[set(r.df.index) for r in tables]) - set(order)) if tables else []
    bench = any(r.bench for r in A + B)
    if bench:
        _check_parity(tables, order)
    D = len(order)
    lr = seed_mean_log(A, metric, order) - seed_mean_log(B, metric, order)   # [D] log ratio
    lb_all = seed_mean_log(B, metric, order)
    out = {"datasets": order, "dropped": dropped, "log_ratio": lr, "ratio": gm(lr),
           "ratio_median": float(np.exp(np.median(lr))),
           "B_divergent": [(d, float(np.exp(v))) for d, v in zip(order, lb_all) if v > np.log(DIVERGE)],
           "nA": len(A), "nB": len(B), "level": level, "bench": bench, "metric": metric}

    # dataset (task) bootstrap (percentile) + t-interval on the mean log ratio
    idx = rng.integers(0, D, (n_boot, D))
    boots = np.exp(lr[idx].mean(1))
    out["ci_ds"] = _ci(boots, level)
    se = lr.std(ddof=1) / np.sqrt(D)
    tq = stats.t.ppf(1 - (1 - level) / 2, D - 1)
    out["ci_t"] = (gm(lr.mean() - tq * se), gm(lr.mean() + tq * se))

    # family-cluster bootstrap (bench). Same rng stream as bench.compare, so CI_fam is identical.
    fam_tab = next((r.df for r in tables if "family" in r.df.columns), None)
    if fam_tab is not None:
        fams = fam_tab.loc[order, "family"].to_numpy()
        ufam = np.unique(fams)
        G = len(ufam)
        gid = np.searchsorted(ufam, fams)
        fsum = np.bincount(gid, weights=lr, minlength=G)
        fcnt = np.bincount(gid, minlength=G).astype(float)
        fmean = fsum / fcnt
        J = np.stack([rng.integers(0, G, G) for _ in range(n_boot)])
        boot_f = np.exp(fsum[J].sum(1) / fcnt[J].sum(1))           # task-weighted ratio
        boot_fw = np.exp(fmean[J].mean(1))                           # family-weighted ratio
        out.update({"families": list(ufam), "fam_log_ratio": fmean, "ratio_famw": gm(fmean),
                    "ci_fam": _ci(boot_f, level), "ci_famw": _ci(boot_fw, level),
                    "p_fam": _boot_p(boot_f), "p_famw": _boot_p(boot_fw),
                    "fam_wins": int((fmean < -TIE).sum()), "fam_losses": int((fmean > TIE).sum())})
        fse = fmean.std(ddof=1) / np.sqrt(G)
        ftq = stats.t.ppf(1 - (1 - level) / 2, G - 1)
        out["ci_fam_t"] = (gm(fmean.mean() - ftq * fse), gm(fmean.mean() + ftq * fse))
        nf = out["fam_wins"] + out["fam_losses"]
        out["p_sign_fam"] = float(stats.binomtest(out["fam_wins"], nf, 0.5).pvalue) if nf else 1.0
        # interval calibration (gate C1): e = |cov80 - 0.8| per task, Alma averaged over seeds
        if all(r.df is not None and "cov80" in r.df.columns for r in A + B):
            eA = np.mean([np.abs(r.df.loc[order, "cov80"].to_numpy(float) - 0.8) for r in A], 0)
            eB = np.mean([np.abs(r.df.loc[order, "cov80"].to_numpy(float) - 0.8) for r in B], 0)
            dc = eA - eB
            csum = np.bincount(gid, weights=dc, minlength=G)
            boot_c = csum[J].sum(1) / fcnt[J].sum(1)
            out["cov80_err"] = (float(eA.mean()), float(eB.mean()))
            out["ci_cov_diff"] = _ci(boot_c, level)
        # A4: non-finite forecasts on A's side
        nn = [r.df.loc[order, "n_nonfinite"].sum() if "n_nonfinite" in r.df.columns else None for r in A]
        out["nonfinite_A"] = None if any(v is None for v in nn) else int(sum(nn))

    # wins / losses (ties: |log ratio| < 0.1%)
    out["wins"] = int((lr < -TIE).sum())
    out["losses"] = int((lr > TIE).sum())
    out["ties"] = D - out["wins"] - out["losses"]
    n = out["wins"] + out["losses"]
    out["p_sign"] = float(stats.binomtest(out["wins"], n, 0.5).pvalue) if n else 1.0
    out["p_wilcoxon"] = float(stats.wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.0

    # seed level: paired by index when both sides have the same number of seeds >= 2
    if len(A) >= 2 and len(A) == len(B):
        per_seed = np.array([np.mean(np.log(a.per_dataset(metric, order).to_numpy(float))
                                     - np.log(b.per_dataset(metric, order).to_numpy(float)))
                             for a, b in zip(A, B)])
    elif bench and len(B) == 1:      # bench: every seed of A against the deterministic baseline
        lb = seed_mean_log(B, metric, order)
        per_seed = np.array([np.mean(np.log(np.clip(a.per_dataset(metric, order).to_numpy(float), 1e-6, None)) - lb)
                             for a in A])
    else:
        per_seed = None
    if per_seed is not None:
        out["seed_ratios"] = np.exp(per_seed)
        S = len(per_seed)
        if S >= 2:
            tq = stats.t.ppf(1 - (1 - level) / 2, S - 1)
            se = per_seed.std(ddof=1) / np.sqrt(S)
            out["ci_seed"] = (gm(per_seed.mean() - tq * se), gm(per_seed.mean() + tq * se))

    # series level (old suite, WQL only, every run on both sides needs a dump or be `sn`)
    if not bench and metric == "wql_rel" and all(r.has_series for r in A + B):
        n_series_boot = min(n_boot, 2000)
        logs_series = np.zeros((n_series_boot, D))
        for j, name in enumerate(order):
            sa = np.stack([series_scores(r, name, split) for r in A])   # [SA, n]
            sb = np.stack([series_scores(r, name, split) for r in B])
            ns = sa.shape[1]
            ii = rng.integers(0, ns, (n_series_boot, ns))
            la = np.log(np.maximum(sa[:, ii].sum(-1), 1e-12)).mean(0)   # [boot]
            lb = np.log(np.maximum(sb[:, ii].sum(-1), 1e-12)).mean(0)
            logs_series[:, j] = la - lb
        s_only = np.exp(logs_series.mean(1))
        out["ci_series"] = (float(np.quantile(s_only, 0.025)), float(np.quantile(s_only, 0.975)))
        jj = rng.integers(0, D, (n_series_boot, D))
        hier = np.exp(np.take_along_axis(logs_series, jj, 1).mean(1))
        out["ci_hier"] = (float(np.quantile(hier, 0.025)), float(np.quantile(hier, 0.975)))
    return out


def report(out, A, B, metric, split, show_table=True):
    nameA = A[0].spec + (f" (+{len(A) - 1} seeds)" if len(A) > 1 else "")
    nameB = B[0].spec + (f" (+{len(B) - 1} seeds)" if len(B) > 1 else "")
    D = len(out["datasets"])
    lv = f"{100 * out['level']:.4g}%"
    unit = "tasks" if out["bench"] else "datasets"
    print(f"A = {nameA}\nB = {nameB}\nsplit={split} metric={metric} {unit}={D}"
          + (f" (dropped, not in both: {', '.join(out['dropped'])})" if out["dropped"] else ""))
    gA = np.exp(seed_mean_log(A, metric, out["datasets"]).mean())
    gB = np.exp(seed_mean_log(B, metric, out["datasets"]).mean())
    print(f"  {metric}: A={gA:.4f}  B={gB:.4f}   (geometric mean over {unit}; seeds pooled)")
    print(f"  ratio A/B          {out['ratio']:.4f}   ({(out['ratio'] - 1) * 100:+.2f}%)   "
          f"median per-{unit[:-1]} ratio {out['ratio_median']:.4f}")
    if out.get("B_divergent"):
        print(f"  B divergent {unit} ({metric} > {DIVERGE:g}, reported as-is, not fixed): "
              + ", ".join(f"{d}={v:.3g}" for d, v in out["B_divergent"]))
        print(f"    the ratio is a geometric mean: each divergent {unit[:-1]} enters as its log (2e10 -> 23.7), so it "
              f"cannot swamp the mean as in an arithmetic mean, but it does pull the ratio in A's favour; the "
              f"median and CI_fam (family bootstrap) are the robust read")
    if "ci_fam" in out:
        G = len(out["families"])
        print(f"  {lv} CI_fam       [{out['ci_fam'][0]:.4f}, {out['ci_fam'][1]:.4f}]   PRIMARY: family-cluster "
              f"bootstrap ({G} families), task-weighted")
        print(f"  {lv} CI_famw      [{out['ci_famw'][0]:.4f}, {out['ci_famw'][1]:.4f}]   family-weighted ratio "
              f"{out['ratio_famw']:.4f}, bootstrap over families")
        print(f"  {lv} CI fam t     [{out['ci_fam_t'][0]:.4f}, {out['ci_fam_t'][1]:.4f}]   t-interval over "
              f"family mean log ratios ({G - 1} df), cross-check")
    print(f"  {lv} CI {unit:9s} [{out['ci_ds'][0]:.4f}, {out['ci_ds'][1]:.4f}]   bootstrap over {unit}"
          + (" (too narrow: tasks of a family are correlated)" if out["bench"] else ""))
    print(f"  {lv} CI t         [{out['ci_t'][0]:.4f}, {out['ci_t'][1]:.4f}]   t-interval on mean log ratio")
    if "ci_series" in out:
        print(f"  95% CI series      [{out['ci_series'][0]:.4f}, {out['ci_series'][1]:.4f}]   "
              "bootstrap over eval series within datasets (test-sample noise)")
        print(f"  95% CI hier        [{out['ci_hier'][0]:.4f}, {out['ci_hier'][1]:.4f}]   datasets then series")
    elif not out["bench"]:
        print("  (series-level CI unavailable: needs TSAR_DUMP dirs / baselines/dumps on both sides)")
    if "seed_ratios" in out:
        rs = " ".join(f"{r:.4f}" for r in out["seed_ratios"])
        ci = f"[{out['ci_seed'][0]:.4f}, {out['ci_seed'][1]:.4f}]" if "ci_seed" in out else "(n/a, 1 seed)"
        print(f"  {lv} CI seeds     {ci}   per-seed ratios: {rs}")
    elif len(A) == 1 or len(B) == 1:
        print("  (no seed-level CI: single seed on at least one side; training noise NOT captured)")
    print(f"  wins/losses/ties   {out['wins']}/{out['losses']}/{out['ties']} {unit}   "
          f"sign test p={out['p_sign']:.3f}   Wilcoxon p={out['p_wilcoxon']:.3f}")
    if "ci_fam" in out:
        print(f"  families W/L       {out['fam_wins']}/{out['fam_losses']} of {len(out['families'])}   "
              f"family sign test p={out['p_sign_fam']:.3f}")
        if "cov80_err" in out:
            print(f"  cov80_error        A={out['cov80_err'][0]:.4f}  B={out['cov80_err'][1]:.4f}   "
                  f"{lv} CI of mean(e_A - e_B) [{out['ci_cov_diff'][0]:+.4f}, {out['ci_cov_diff'][1]:+.4f}]")
    if show_table:
        print(f"  {'dataset':32s} {'A':>8s} {'B':>8s} {'A/B':>7s}")
        a = np.exp(seed_mean_log(A, metric, out["datasets"]))
        b = np.exp(seed_mean_log(B, metric, out["datasets"]))
        for d, x, y, l in sorted(zip(out["datasets"], a, b, out["log_ratio"]), key=lambda t: t[3]):
            flag = "  win" if l < -TIE else ("  LOSS" if l > TIE else "")
            print(f"  {d:32s} {x:8.4f} {y:8.4f} {np.exp(l):7.3f}{flag}")
        if "ci_fam" in out:
            print(f"  {'family':32s} {'A/B':>7s}")
            for f, l in sorted(zip(out["families"], out["fam_log_ratio"]), key=lambda t: t[1]):
                flag = "  win" if l < -TIE else ("  LOSS" if l > TIE else "")
                print(f"  {f:32s} {np.exp(l):7.3f}{flag}")


def verdict(out, alpha_upper=1.0):
    """Old suite (eval_rigor R4): 'A beats B' only if every available CI upper bound is < 1."""
    checks = [("dataset bootstrap", out["ci_ds"][1])]
    for k, label in (("ci_hier", "hierarchical bootstrap"), ("ci_seed", "seed t-interval")):
        if k in out:
            checks.append((label, out[k][1]))
    ok = all(u < alpha_upper for _, u in checks)
    return ok, checks


def bench_gate(out, min_seeds=3):
    """Gate G (docs/eval_rigor.md, bench claim gate) for one baseline and one metric.
    Returns [(label, passed, detail)]. Every item must pass."""
    g = []
    g.append(("G1 CI_fam upper < 1", out["ci_fam"][1] < 1, f"upper={out['ci_fam'][1]:.4f}"))
    g.append(("G2 CI_famw upper < 1", out["ci_famw"][1] < 1, f"upper={out['ci_famw'][1]:.4f}"))
    G = len(out["families"])
    g.append(("G3 family majority", out["fam_wins"] > G / 2, f"{out['fam_wins']}/{G} families won (need > {G / 2:g})"))
    sr = out.get("seed_ratios")
    if sr is None:
        g.append(("G4 seed condition", False, "no per-seed ratios (B must be a single deterministic run)"))
    else:
        ok = bool(np.all(sr < 1)) and len(sr) >= min_seeds and ("ci_seed" not in out or out["ci_seed"][1] < 1)
        det = f"{len(sr)} seed(s), max ratio {sr.max():.4f}" + (
            f", t upper {out['ci_seed'][1]:.4f}" if "ci_seed" in out else "") + (
            f"; need >= {min_seeds} seeds" if len(sr) < min_seeds else "")
        g.append(("G4 seed condition", ok, det))
    nf = out.get("nonfinite_A")
    g.append(("G5 no non-finite forecasts", nf == 0, "n_nonfinite column missing" if nf is None else f"{nf} bad forecasts"))
    if out["dropped"]:
        g.append(("G0 identical task set", False, f"tasks not in both runs: {out['dropped'][:5]}"))
    return g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _datasets(args):
    if args.datasets:
        return args.datasets.split(",")
    if args.zs4:
        return ZS4
    return None


def _available(spec, split):
    """A run spec resolves without exiting: an existing path/glob, or a bench baseline with a table."""
    if re.fullmatch(r"[0-9.]+", spec) or spec.lower() in ("sn", "seasonal_naive") or os.path.exists(spec) \
            or any(ch in spec for ch in "*?[,"):
        return True
    return os.path.exists(os.path.join(BENCH_BL_DIR, "dumps", spec, f"per_task_{split}.csv")) or \
        os.path.exists(os.path.join(BENCH_BL_DIR, "per_task", f"{spec}_{split}.csv"))


def _baseline_specs(args):
    bs = list(args.b)
    if getattr(args, "set", None) == "final":
        bs += [b for b in FINAL_SET if b not in bs]
    return bs


def cmd_pair(args, claim=False):
    bs = _baseline_specs(args)
    if not bs:
        sys.exit("give at least one baseline B (or --set final)")
    bench = detect_bench([args.a] + bs, args.split, args.bench)
    if claim and bench:
        return cmd_claim_bench(args, bs)
    if len(bs) != 1:
        sys.exit("pair (and the old-suite claim) take exactly one B")
    A, B = load_specs(args.a, args.split, bench), load_specs(bs[0], args.split, bench)
    ds = _datasets(args)
    if args.exclude_zs4:
        base = [r for r in A + B if r.df is not None][0]
        ds = [d for d in base.df.index if d not in ZS4]
    out = compare(A, B, args.split, args.metric, args.boot, ds, args.seed)
    report(out, A, B, args.metric, args.split, show_table=not args.brief)
    if claim:
        ok, checks = verdict(out)
        print("\nCLAIM 'A beats B':", "PASS" if ok else "FAIL")
        for label, u in checks:
            print(f"  {label:24s} upper={u:.4f}  {'< 1 ok' if u < 1 else '>= 1  -> not established'}")
        if len(A) < 3:
            print(f"  note: A has {len(A)} seed(s); docs/eval_rigor.md requires >= 3 seeds for a reported claim")
        sys.exit(0 if ok else 1)


def cmd_claim_bench(args, bs):
    """Gate G against every baseline, on wql_rel and mase_rel. Exit 0 only if the conjunctive
    claim 'A beats every B' (intersection-union test) passes."""
    if args.datasets or args.zs4 or args.exclude_zs4:
        sys.exit("a bench claim is made on the full task set; --datasets/--zs4 are not allowed")
    alpha = args.alpha / args.candidates                      # Bonferroni across claim-bearing candidates
    level = 1 - alpha
    missing = [b for b in bs if not _available(b, args.split)]
    if missing:
        print(f"INCOMPLETE baseline set: no {args.split} scores for {', '.join(missing)}"
              + (" (pre-registered FINAL_SET members)" if getattr(args, "set", None) == "final" else ""))
        if not args.allow_incomplete:
            sys.exit(2)
        bs = [b for b in bs if b not in missing]
        print("  --allow-incomplete: comparing the rest; the claim cannot PASS\n")
    A = load_specs(args.a, args.split, True)
    metrics = ["wql_rel", "mase_rel"]
    rows = []                                                 # (B, metric, out, gate)
    for b in bs:
        B = load_specs(b, args.split, True)
        if len(B) != 1:
            sys.exit(f"baseline {b!r} must be one deterministic run (got {len(B)})")
        for metric in metrics:
            out = compare(A, B, args.split, metric, args.boot, None, args.seed, level=level)
            print("=" * 100)
            report(out, A, B, metric, args.split, show_table=not args.brief)
            gate = bench_gate(out, args.min_seeds)
            for label, ok, det in gate:
                print(f"  {'PASS' if ok else 'FAIL'}  {label:28s} {det}")
            rows.append((b, metric, out, gate))
    # Holm across the individual 'A beats B on metric' statements (each p = the larger of the two
    # family-bootstrap p-values, so it is consistent with G1 and G2).
    p_raw = [max(o["p_fam"], o["p_famw"]) for _, _, o, _ in rows]
    p_holm = holm(p_raw)
    p_sign_holm = holm([o["p_sign_fam"] for _, _, o, _ in rows])
    print("=" * 100)
    print(f"Gate G summary   split={args.split}  A={args.a}  seeds={len(A)}  alpha={args.alpha}"
          + (f"/{args.candidates} candidates = {alpha:.4g}" if args.candidates > 1 else "")
          + f"  CI level {100 * level:.4g}%  bootstrap B={args.boot} seed={args.seed}")
    print(f"  {'baseline':20s} {'metric':9s} {'ratio':>7s} {'median':>7s} {'CI_fam':>17s} {'CI_famw':>17s} {'famW/L':>7s} "
          f"{'p_boot':>7s} {'p_Holm':>7s} {'sgnHolm':>7s}  gate")
    all_ok, per_b = True, {}
    for (b, metric, o, gate), pr, ph, ps in zip(rows, p_raw, p_holm, p_sign_holm):
        ok = all(x for _, x, _ in gate)
        all_ok &= ok
        per_b.setdefault(b, []).append(ok and ph < alpha)
        print(f"  {b[-20:]:20s} {metric:9s} {o['ratio']:7.4f} {o['ratio_median']:7.4f} [{o['ci_fam'][0]:.4f},{o['ci_fam'][1]:.4f}] "
              f"[{o['ci_famw'][0]:.4f},{o['ci_famw'][1]:.4f}] {o['fam_wins']:>3d}/{o['fam_losses']:<3d} "
              f"{pr:7.4f} {ph:7.4f} {ps:7.4f}  {'PASS' if ok else 'FAIL'}")
    div = sorted({(b, d) for b, _, o, _ in rows for d, _ in o.get("B_divergent", [])})
    print("  ratio = geometric mean of per-task ratios (a divergent baseline task enters as its log, so it cannot "
          "dominate as in an arithmetic mean); median = median per-task ratio, insensitive to such tasks.")
    if div:
        print(f"  divergent baseline tasks (> {DIVERGE:g} x SN, reported as-is): "
              + ", ".join(f"{b}:{d}" for b, d in div))
    print("\nIndividual statements 'A beats B' (gate G on both metrics AND Holm-adjusted p < alpha):")
    for b, oks in per_b.items():
        print(f"  {b:24s} {'PASS' if all(oks) else 'FAIL -> write comparable to B'}")
    wq = [o for _, m, o, _ in rows if m == "wql_rel"]
    best = min(wq, key=lambda o: o["ratio"]) if wq else None
    worst_r = max(o["ratio"] for o in wq) if wq else None
    if best is not None:
        lm = all(o["ratio"] <= 0.95 and o["ci_fam"][1] <= 0.98 for o in wq)
        print(f"Large margin (wql_rel ratio <= 0.95 and CI_fam upper <= 0.98 vs every B): "
              f"{'yes' if lm and all_ok else 'no'}  (worst ratio {worst_r:.4f})")
    cal = [o for _, m, o, _ in rows if m == "wql_rel" and "ci_cov_diff" in o]
    if cal:
        c_ok = all(o["ci_cov_diff"][1] < 0 and o["cov80_err"][0] <= o["cov80_err"][1] for o in cal)
        print(f"Calibration C1 (cov80 error, claimable only if accuracy passes): {'PASS' if c_ok and all_ok else 'FAIL'}")
    if missing:
        all_ok = False
        print(f"\nINCOMPLETE: {', '.join(missing)} not scored on {args.split}; the claim is FAIL by construction")
    print(f"\nCLAIM 'A beats every baseline in {{{', '.join(bs)}}}' (intersection-union, gate G on wql_rel "
          f"and mase_rel): {'PASS' if all_ok else 'FAIL'}")
    if len(A) < args.min_seeds:
        print(f"  note: {len(A)} seed(s) < --min-seeds {args.min_seeds}: not a reportable claim")
    sys.exit(0 if all_ok else 1)


def cmd_seeds(args):
    bench = detect_bench([args.a], args.split, args.bench)
    runs = load_specs(args.a, args.split, bench)
    ds = _datasets(args)
    vals = np.array([r.gmean(args.metric, ds) for r in runs])
    for r, v in zip(runs, vals):
        print(f"  {r.spec:50s} {args.metric}={v:.4f}")
    n = len(vals)
    sd = vals.std(ddof=1) if n > 1 else float("nan")
    print(f"{args.metric}: mean={vals.mean():.4f}  std={sd:.4f}  (n={n} seeds)  "
          f"min={vals.min():.4f} max={vals.max():.4f}")
    if n > 1:
        tq = stats.t.ppf(0.975, n - 1)
        print(f"  95% CI of the mean: [{vals.mean() - tq * sd / np.sqrt(n):.4f}, "
              f"{vals.mean() + tq * sd / np.sqrt(n):.4f}]   report as {vals.mean():.3f} +- {sd:.3f}")
        print(f"  decision threshold (2 x seed std): {2 * sd:.4f}")
    tabled = [r for r in runs if r.df is not None]
    if len(tabled) > 1:
        common = [d for d in tabled[0].df.index if all(d in r.df.index for r in tabled)]
        if ds:
            common = [d for d in common if d in ds]
        M = np.stack([r.per_dataset(args.metric, common).to_numpy(float) for r in tabled])
        print(f"  {'dataset':32s} {'mean':>8s} {'std':>8s} {'cv%':>6s}")
        for j in np.argsort(-M.std(0, ddof=1) / M.mean(0)):
            print(f"  {common[j]:32s} {M[:, j].mean():8.4f} {M[:, j].std(ddof=1):8.4f} "
                  f"{100 * M[:, j].std(ddof=1) / M[:, j].mean():6.2f}")


def cmd_board(args):
    split = args.split
    bench = detect_bench(list(args.runs) + [args.ref], split, args.bench)
    pdir = os.path.join(BENCH_BL_DIR, "per_task") if bench else os.path.join(BL_DIR, "per_dataset")
    names = sorted(f[:-len(f"_{split}.csv")] for f in os.listdir(pdir) if f.endswith(f"_{split}.csv"))
    entries = [(n, [Run(n, split, bench)]) for n in names]
    for spec in args.runs:
        try:
            entries.append((spec, load_specs(spec, split, bench)))
        except SystemExit as e:
            print(f"skipping {spec}: {e}")
    ref = load_specs(args.ref, split, bench)
    rows = []
    for name, runs in entries:
        out = compare(runs, ref, split, args.metric, args.boot, _datasets(args), args.seed)
        g = np.exp(seed_mean_log(runs, args.metric, out["datasets"]).mean())
        ci = out.get("ci_fam", out.get("ci_hier", out["ci_ds"]))
        rows.append((g, name, out["ratio"], ci, out["wins"], out["losses"], len(out["datasets"])))
    print(f"split={split} metric={args.metric} reference={args.ref}"
          + ("  (bench; CI = family-cluster bootstrap)" if bench else ""))
    print(f"  {'model':34s} {args.metric:>8s} {'vs ref':>7s} {'95% CI':>17s} {'W/L':>6s}  D")
    for g, name, r, ci, w, l, D in sorted(rows):
        print(f"  {name[-34:]:34s} {g:8.4f} {r:7.4f} [{ci[0]:.3f},{ci[1]:.3f}] {w:>2d}/{l:<2d}  {D}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--split", default="val", choices=["val", "test", "holdout"])
        p.add_argument("--metric", default="wql_rel", choices=["wql_rel", "mase_rel"])
        p.add_argument("--boot", type=int, default=10000)
        p.add_argument("--datasets", help="comma-separated subset")
        p.add_argument("--zs4", action="store_true", help="only the 4 ZS_SETS of train.py")
        p.add_argument("--exclude-zs4", action="store_true")
        p.add_argument("--seed", type=int, default=0, help="bootstrap rng seed")
        p.add_argument("--brief", action="store_true", help="omit the per-dataset table")
        p.add_argument("--bench", action="store_true", help="resolve baseline names on the bench "
                       "(auto-detected from the run files; also implied by TSAR_BENCH=1)")

    p = sub.add_parser("pair")
    p.add_argument("a")
    p.add_argument("b", nargs=1)
    common(p)
    p = sub.add_parser("claim")
    p.add_argument("a")
    p.add_argument("b", nargs="*")
    p.add_argument("--set", choices=["final"], help="bench: add the pre-registered baseline set")
    p.add_argument("--alpha", type=float, default=0.05, help="two-sided level per comparison")
    p.add_argument("--candidates", type=int, default=1, help="claim-bearing candidates (Bonferroni)")
    p.add_argument("--min-seeds", type=int, default=3, help="bench: seeds of A required for PASS")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="bench: run without baselines that have no scores (the claim then always FAILs)")
    common(p)
    p = sub.add_parser("seeds")
    p.add_argument("a")
    common(p)
    p = sub.add_parser("board")
    p.add_argument("runs", nargs="*")
    p.add_argument("--ref", default="sn")
    common(p)
    args = ap.parse_args()
    if args.cmd == "pair":
        cmd_pair(args)
    elif args.cmd == "claim":
        cmd_pair(args, claim=True)
    elif args.cmd == "seeds":
        cmd_seeds(args)
    else:
        cmd_board(args)


if __name__ == "__main__":
    main()
