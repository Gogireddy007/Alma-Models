"""Bench val leaderboard over every finished run in runs/*.log, seeds grouped.

    uv run tools/leaderboard.py [--ref b_base256] [--ref-seeds c_base_s1,c_base_s2] [--boot 5000] [--no-cache]

- A bench run is a log with a `val_wql_rel:` line and a bench per-task table (has `family`).
  Only the VAL table is read; test / holdout tables are never parsed into the board.
- Seeds: `<name>_s<k>` joins `<name>`; seed 0 is `<name>` itself.
- The reference is --ref plus --ref-seeds as its seeds 1..k (default: b_base256, with c_base_s1/s2,
  which are the same code path; c_base == b_base256 bit-for-bit at seed 0).
- Comparisons are seed-matched (audit_v3 H-3): seed k vs reference seed k, reference seed 0 standing in
  for a seed the reference lacks (`0*` in the ref seeds column).
- Each row: mean +- std over seeds of the summary lines (val wql_rel, mase_rel, cov80_error),
  training_seconds, peak_mem_mb, and compare.compare() vs the reference (paired family-cluster
  bootstrap, task-weighted CI_fam, both metrics). For multi-seed rows the seed-0-only (rung-1) ratio
  and CI are also kept, so promote.py decides rung 2 on the rung-1 evidence alone.
Also regenerates results_bench.tsv (program.md-style log, one row per config, seeds grouped; status
keep = CI_fam upper < 1 with cov80_error no worse than +0.01, or wql ratio <= 1 with cov80_error better by > 0.01;
promising = ratio < 1 but CI crosses 1;
ablation = name matches ABLATION_RE; crash = runs/<name>.failed; else discard). Rows are rebuilt from
the logs each run, so it is idempotent; descriptions you edit (not starting `auto:`) are preserved.
The prov column flags UNVERIFIED runs: no runs/<name>.prov.json, or script sha256 != the committed file.
Failed metrics (lessons L6): a run whose val wql_rel is non-finite or > WQL_FAIL (2.0) is never ranked or
compared; it is listed under "Failed" in the md and logged as crash in the tsv, to be investigated.
Writes runs/leaderboard.md (for people) and runs/leaderboard.json (for tools/promote.py). The json also
caches each compare() result keyed by log names + mtimes, so a re-run only bootstraps new or changed rows
(cold ~25 s with --boot 5000 on a busy box, warm ~10 s, mostly imports).
"""
import argparse, contextlib, glob, hashlib, io, json, os, re, subprocess, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
SEED_RE = re.compile(r"^(.*)_s(\d+)$")
SUMMARY = {"wql": "val_wql_rel", "mase": "val_mase_rel", "cov80": "val_cov80_error",
           "train_s": "training_seconds", "peak_mb": "peak_mem_mb", "params": "num_params_M"}


def split_seed(name):
    m = SEED_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def summary(text):
    out = {}
    for k, key in SUMMARY.items():
        m = re.search(rf"^{key}:\s+([-\d.naninf]+)", text, re.M)
        out[k] = float(m.group(1)) if m else float("nan")
    return out


WQL_FAIL = 2.0
FAILED = {}                                            # name -> reason, filled by scan() (lessons L6)


def bad_metric(wql):
    """Reason a wql_rel cannot be ranked (non-finite or > WQL_FAIL), or None."""
    return None if np.isfinite(wql) and wql <= WQL_FAIL else f"failed metric: val wql_rel {wql} (> {WQL_FAIL} or non-finite)"


def scan():
    """{name: {"path", "summary", "run"}} for every finished bench val log with a rankable wql_rel."""
    runs = {}
    FAILED.clear()
    for path in sorted(glob.glob(os.path.join(RUNS, "*.log"))):
        text = open(path, errors="replace").read()
        if not re.search(r"^val_wql_rel:", text, re.M) or not any(C._BENCH_HDR.match(l) for l in text.splitlines()):
            continue
        name = os.path.basename(path)[:-4]
        why = bad_metric(summary(text)["wql"])
        if why:
            FAILED[name] = why
            continue
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                run = C.Run(path, "val", bench=True)
        except SystemExit as e:
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        runs[name] = {"path": path, "summary": summary(text), "run": run}
    return runs


def group(runs, ref, ref_seeds):
    groups = {}
    for name in runs:
        base, seed = split_seed(name)
        groups.setdefault(base, {})[seed] = name
    if ref in runs:                                     # the reference borrows its other seeds
        for i, s in enumerate(ref_seeds, 1):
            if s in runs:
                groups[ref][i] = s
    return {g: dict(sorted(v.items())) for g, v in groups.items()}


_CACHE, _USED = {}, set()


def _key(A, B, metric, n_boot):
    st = lambda r: f"{os.path.basename(r.spec)}@{os.path.getmtime(r.spec):.0f}"
    return "|".join([metric, str(n_boot), ",".join(map(st, A)), ",".join(map(st, B))])


def _cmp(A, B, metric, n_boot):
    k = _key(A, B, metric, n_boot)
    _USED.add(k)
    if k in _CACHE:
        return _CACHE[k]
    _CACHE[k] = _cmp_raw(A, B, metric, n_boot)
    return _CACHE[k]


def _cmp_raw(A, B, metric, n_boot):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            o = C.compare(A, B, "val", metric=metric, n_boot=n_boot, seed=0, quiet=True)
    except SystemExit as e:
        return {"error": str(e)}
    return {"ratio": o["ratio"], "ci_fam": list(o.get("ci_fam", (np.nan, np.nan))),
            "fam_wins": o.get("fam_wins"), "fam_losses": o.get("fam_losses"), "n_tasks": len(o["datasets"])}


def ms(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if not len(xs):
        return [float("nan"), float("nan")]
    return [float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else float("nan")]


# ---------------------------------------------------------------------------
# Provenance (runs/<name>.prov.json, written by tools/jobq.py)
# ---------------------------------------------------------------------------

_BLOB = {}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).stdout


def prov(name):
    f = os.path.join(RUNS, f"{name}.prov.json")
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return None


def verify(name):
    """'' if the log has a prov.json whose script sha256 equals the script committed at its git_sha,
    else the reason it is UNVERIFIED."""
    p = prov(name)
    if p is None:
        return "no prov"
    key = (p["git_sha"], p["script"])
    if key not in _BLOB:
        blob = git("show", f"{p['git_sha']}:{p['script']}")
        _BLOB[key] = hashlib.sha256(blob).hexdigest() if blob else None
    if _BLOB[key] is None:
        return "script not committed"
    return "" if _BLOB[key] == p["sha256"].get(p["script"]) else "script != commit"


def build(ref="b_base256", ref_seeds=("c_base_s1", "c_base_s2"), n_boot=5000):
    runs = scan()
    if ref not in runs:
        sys.exit(f"reference {ref} has no finished bench val log")
    groups = group(runs, ref, list(ref_seeds))
    rmap = groups[ref]                                             # reference seed -> log
    rows = []
    for g, seeds in groups.items():
        names = list(seeds.values())
        A = [runs[n]["run"] for n in names]
        row = {"name": g, "seeds": list(seeds), "logs": names, "n_seeds": len(names)}
        for k in SUMMARY:
            row[k] = ms([runs[n]["summary"][k] for n in names])
        refs = groups[ref]                                         # seed -> log; compare calibration seed-matched
        shared = [k for k in seeds if k in refs] or [None]
        cov = lambda n: runs[n]["summary"]["cov80"]
        row["dcov"] = float(np.mean([cov(seeds[k]) - cov(refs[k]) if k is not None
                                     else cov(names[0]) - cov(refs[min(refs)]) for k in shared]))
        bad = {n: v for n in names if (v := verify(n))}
        row["unverified"] = sorted(set(bad.values()))
        # H-3: seed-matched reference. Candidate seed k is paired with reference seed k (same data and
        # synthetic RNG streams); a seed the reference lacks is paired with reference seed 0 (marked *).
        R = [runs[rmap.get(k, rmap[0])]["run"] for k in seeds]
        row["ref_match"] = ",".join(f"{k}" if k in rmap else "0*" for k in seeds)
        for metric, key in (("wql_rel", "vs_wql"), ("mase_rel", "vs_mase")):
            row[key] = _cmp(A, R, metric, n_boot)
            if 0 in seeds and len(names) > 1:
                row[key + "_s0"] = _cmp([runs[seeds[0]]["run"]], [runs[rmap[0]]["run"]], metric, n_boot)
            elif 0 in seeds:
                row[key + "_s0"] = row[key]
        rows.append(row)
    rows.sort(key=lambda r: (np.nan_to_num(r["wql"][0], nan=9.0), r["name"]))
    return {"ref": ref, "ref_seeds": list(ref_seeds), "ref_logs": list(groups[ref].values()), "n_boot": n_boot,
            "generated": time.strftime("%Y-%m-%d %H:%M"), "rows": rows, "failed": dict(sorted(FAILED.items())),
            "cache": {k: _CACHE[k] for k in sorted(_USED)}}                 # drop stale entries


def _f(x, d=4):
    return "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pm(v, d=4):
    return _f(v[0], d) + (f" ± {_f(v[1], d)}" if np.isfinite(v[1]) else "")


def _ci(c):
    if "error" in c:
        return "n/a"
    lo, hi = c["ci_fam"]
    mark = " **" if hi < 1 else ""
    return f"{c['ratio']:.3f} [{lo:.3f}, {hi:.3f}]{mark}"


def to_md(b):
    L = [f"# Bench val leaderboard ({b['generated']})", "",
         f"Reference: `{b['ref']}`, seeds = {', '.join(b['ref_logs'])}. Seed-matched (audit_v3 H-3): candidate seed k "
         f"is paired with reference seed k; a seed the reference lacks is paired with reference seed 0, shown as "
         f"`0*` in `ref seeds`. The rung-1 column is seed 0 vs reference seed 0. Ratio < 1 = better than the "
         f"reference; [CI] = paired family-cluster bootstrap, task-weighted 95% CI_fam ({b['n_boot']} draws), "
         f"compare.py. ** = CI_fam upper < 1. VAL only. Sorted by mean val wql_rel. prov: UNVERIFIED = some seed "
         f"has no runs/<name>.prov.json (pre-jobq-provenance run) or its script differs from the committed one.", "",
         "| # | name | n | seeds | wql_rel | mase_rel | cov80_err | wql ratio [CI_fam] | mase ratio [CI_fam] "
         "| ref seeds | rung-1 (s0) wql ratio [CI_fam] | train_s | peak_mb | prov |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(b["rows"], 1):
        s0 = _ci(r["vs_wql_s0"]) if r["n_seeds"] > 1 and "vs_wql_s0" in r else ""
        L.append(f"| {i} | {r['name']} | {r['n_seeds']} | {','.join(map(str, r['seeds']))} | {_pm(r['wql'])} "
                 f"| {_pm(r['mase'])} | {_pm(r['cov80'])} | {_ci(r['vs_wql'])} | {_ci(r['vs_mase'])} | {r['ref_match']} | {s0} "
                 f"| {_f(r['train_s'][0], 0)} | {_f(r['peak_mb'][0], 0)} "
                 f"| {'UNVERIFIED (' + ', '.join(r['unverified']) + ')' if r['unverified'] else 'ok'} |")
    if b.get("failed"):
        L += ["", f"## Failed (not ranked; val wql_rel > {WQL_FAIL} or non-finite, lessons L6): investigate", ""]
        L += [f"- {n}: {why}" for n, why in b["failed"].items()]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# results_bench.tsv (program.md-style experiment log for the bench; regenerated, idempotent)
# ---------------------------------------------------------------------------

TSV = os.path.join(ROOT, "results_bench.tsv")
TSV_COLS = ["commit", "script", "name", "seeds", "val_wql_rel", "val_mase_rel", "cov80_err", "params_M",
            "train_s", "peak_mb", "ratio", "ci_fam_lo", "ci_fam_hi", "status", "description"]
ABLATION_RE = re.compile(r"^b_a\d_|_A\d+$|_nocov$|(^|_)loo_")   # audit ablations, eval-time withholds, rung-3 LOO
DESC_DROP = {"TSAR_STEPS", "TSAR_BENCH", "TSAR_SEED", "TSAR_RUN"}


def job_defs():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jobq import parse
    defs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "jobs", "*.txt"))):
        try:
            for j in parse(f):
                defs.setdefault(j["name"], j)
        except (ValueError, IndexError):
            pass
    return defs


def commit_of(name, defs):
    p = prov(name)
    if p:
        return p["git_sha"][:7] + ("+dirty" if p["git_dirty"] else "")
    log = os.path.join(RUNS, f"{name}.log")         # pre-provenance: HEAD when the log was last written
    t = int(os.path.getmtime(log)) if os.path.exists(log) else int(time.time())
    sha = git("rev-list", "-1", f"--before={t}", "HEAD").decode().strip()
    return "~" + sha[:7] if sha else "?"


def status(r, ref):
    if ABLATION_RE.search(r["name"]):
        return "ablation"
    c = r.get("vs_wql", {})
    if not c or "error" in c:
        return "discard"
    if r["name"] == ref["name"]:
        return "keep"
    dcov = r["dcov"]                                               # seed-matched (the reference drifts across seeds)
    if c["ci_fam"][1] < 1 and dcov <= 0.01:
        return "keep"                                              # accuracy win, calibration not worse
    if c["ratio"] <= 1.0 and dcov < -0.01:
        return "keep"                                              # calibration win, accuracy not worse
    return "promising" if c["ratio"] < 1 else "discard"


def write_tsv(b, path=TSV):
    """Regenerate the file from the board; hand-written descriptions (not starting `auto:`) survive."""
    old = {}
    if os.path.exists(path):
        for ln in open(path).read().splitlines()[1:]:
            f = ln.split("\t")
            if len(f) == len(TSV_COLS) and not f[-1].startswith("auto:"):
                old[f[2]] = f[-1]
    defs = job_defs()
    ref = next(r for r in b["rows"] if r["name"] == b["ref"])
    fmt = lambda x, d: "0" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    out = ["\t".join(TSV_COLS)]
    rows = list(b["rows"])
    for fail in sorted(glob.glob(os.path.join(RUNS, "*.failed"))):    # crashes: metrics 0 (program.md)
        n = os.path.basename(fail)[:-7]
        rows.append({"name": n, "logs": [n], "seeds": [split_seed(n)[1]], "crash": open(fail).read().strip()})
    for n, why in b.get("failed", {}).items():                        # lessons L6: a blown-up metric is a crash
        rows.append({"name": n, "logs": [n], "seeds": [split_seed(n)[1]], "crash": why})
    for r in rows:
        first = r["logs"][0]
        j = defs.get(first) or {}
        p = prov(first) or {}
        script = p.get("script") or j.get("script", "?")
        env = p.get("env") or j.get("env", {})
        commits = ",".join(dict.fromkeys(commit_of(n, defs) for n in r["logs"]))
        flags = " ".join(f"{k}={v}" for k, v in env.items() if k not in DESC_DROP)
        if "crash" in r:
            vals, st = ["0"] * 9, "crash"
            desc = old.get(r["name"], f"auto: {r['crash']}; {flags}")
        else:
            c = r["vs_wql"]
            vals = [fmt(r["wql"][0], 6), fmt(r["mase"][0], 6), fmt(r["cov80"][0], 4), fmt(r["params"][0], 3),
                    fmt(r["train_s"][0], 1), fmt(r["peak_mb"][0], 0), fmt(c.get("ratio"), 4),
                    fmt(c.get("ci_fam", [None, None])[0], 4), fmt(c.get("ci_fam", [None, None])[1], 4)]
            st = status(r, ref)
            desc = old.get(r["name"], "auto: " + ("reference" if r["name"] == b["ref"] else flags or "defaults"))
        seeds = ",".join(map(str, r["seeds"]))
        out.append("\t".join([commits, script, r["name"], seeds, *vals, st, desc.replace("\t", " ")]))
    text = "\n".join(out) + "\n"
    if not os.path.exists(path) or open(path).read() != text:
        open(path, "w").write(text)
    return len(out) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="b_base256")
    ap.add_argument("--ref-seeds", default="c_base_s1,c_base_s2", help="logs that act as the reference's seeds 1..k")
    ap.add_argument("--boot", type=int, default=5000, help="bootstrap draws (compare.py uses 10000; 5000 keeps a cold run < 30 s)")
    ap.add_argument("--no-cache", action="store_true", help="recompute every CI (cold run)")
    ap.add_argument("--out", default=os.path.join(RUNS, "leaderboard.md"))
    a = ap.parse_args()
    t0 = time.time()
    side = os.path.splitext(a.out)[0] + ".json"
    if not a.no_cache and os.path.exists(side):       # compare() results keyed by log names + mtimes
        try:
            _CACHE.update(json.load(open(side)).get("cache", {}))
        except (OSError, ValueError):
            pass
    b = build(a.ref, [s for s in a.ref_seeds.split(",") if s], a.boot)
    md = to_md(b)
    open(a.out, "w").write(md)
    json.dump(b, open(side, "w"), indent=1, default=float)
    n = write_tsv(b)
    print(md)
    print(f"results_bench.tsv: {n} rows", file=sys.stderr)
    print(f"wrote {a.out} (+ .json), {len(b['rows'])} rows, {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
