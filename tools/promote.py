"""Climb the experiment ladder (docs/team.md) from the leaderboard: emit the next rung's jobq file.

    uv run tools/leaderboard.py                                   # refresh runs/leaderboard.json first
    uv run tools/promote.py runs/jobs/rung1.txt -o runs/jobs/rung2_auto.txt
    uv run tools/promote.py --rung3 runs/jobs/rung1.txt -o runs/jobs/rung3_auto.txt

Rung 2 (default). For every job in the rung-1 jobs file whose RUNG-1 (seed-0-only) result vs the
reference passes `CI_fam upper < --upper (1.0)` OR `ratio <= --ratio (0.985)` on val wql_rel, emit
seeds --seeds (1,2) as `<name>_s<k>`, same script, env + TSAR_SEED=k, same mem=. A re-score job
(`run=<parent>`) becomes `run=<parent>_s<k> | after=<parent>_s<k>`, and the parent's seed-k training job
is emitted before it even if the parent itself did not pass. The reference's seeds (leaderboard
--ref-seeds, default c_base_s1/s2) are emitted first when their logs are not finished. Order: reference
seeds, then all seed-1 jobs, then all seed-2 jobs (two seeds of everything arrive early). Jobs whose log
is already finished are kept (jobq skips them) and marked `# done`.

Rung 3 (--rung3). Confirmed winners = leaderboard rows with >= --min-seeds (3) seeds, seed-mean val
wql_rel below the reference's and paired CI_fam upper < 1 (all seeds vs all reference seeds). Their
flags (from the given jobs files; seed/steps/bench/size keys dropped) are unioned, the better winner
winning any conflict (listed as a comment). Emits on train_v2.py at TSAR_D_MODEL=160 TSAR_N_LAYERS=3
(~1.5M params), TSAR_STEPS=10000, seeds 0 and 1: <p>base (flags off: the same-scale reference),
<p>combo, and <p>loo_<winner> for each winner (union of the others) when there are >= 2 winners; a
leave-one-out identical to an earlier variant (flags subsumed by another winner) is skipped.

Reads runs/leaderboard.json and the jobs files only. Never launches anything; VAL evidence only.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobq import RUNS, done, parse  # noqa: E402
from leaderboard import split_seed  # noqa: E402

DROP = {"TSAR_SEED", "TSAR_STEPS", "TSAR_BENCH", "TSAR_RUN", "TSAR_D_MODEL", "TSAR_N_LAYERS", "TSAR_SPLIT"}


def line(name, script, env, after=None, run=None, mem=None, note=""):
    parts = [name, script, " ".join(f"{k}={v}" for k, v in env.items())]
    if after:
        parts.append(f"after={after}")
    if run and run != name:
        parts.append(f"run={run}")
    if mem:
        parts.append(f"mem={int(mem)}")
    tags = [t for t in (note, "done" if done(name) else "",
                        "FAILED marker" if (RUNS / f"{name}.failed").exists() else "") if t]
    return " | ".join(parts) + (f"   # {'; '.join(tags)}" if tags else "")


def load_board(path):
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run `uv run tools/leaderboard.py` first")
    b = json.load(open(path))
    t = os.path.getmtime(path)
    newer = [p.stem for p in RUNS.glob("*.log") if p.stat().st_mtime > t and done(p.stem)]
    if newer:
        print(f"warning: {len(newer)} finished logs are newer than {path} (e.g. {newer[:3]}); "
              f"re-run tools/leaderboard.py", file=sys.stderr)
    return b, {r["name"]: r for r in b["rows"]}


def fmt(c):
    return "no CI" if not c or "error" in c else f"wql {c['ratio']:.4f} [{c['ci_fam'][0]:.4f}, {c['ci_fam'][1]:.4f}]"


def rung2(a, board, rows):
    jobs = parse(a.jobs[0])
    by = {j["name"]: j for j in jobs}
    ref = board["ref"]
    skip = {ref, *board.get("ref_seeds", [])}
    seeds = [int(s) for s in a.seeds.split(",")]
    out, emitted, keep, drop = [], set(), [], []
    also = set(filter(None, a.also.split(",")))

    def seed_job(j, k, note=""):
        """Lines for job j at seed k (its parent's seed-k training job first, for a re-score)."""
        name = f"{j['name']}_s{k}"
        if name in emitted:
            return []
        L, after, run = [], None, None
        if j["run"] != j["name"]:
            parent = by.get(j["run"])
            if parent is None:
                return [f"# SKIP {name}: re-scores {j['run']}, which is not in {a.jobs[0]}"]
            L += seed_job(parent, k, f"parent of {name}")
            after = run = f"{parent['name']}_s{k}"
        emitted.add(name)
        env = {kk: v for kk, v in j["env"].items() if kk != "TSAR_SEED"} | {"TSAR_SEED": str(k)}
        return L + [line(name, j["script"], env, after, run, j["mem"], note)]

    # reference seeds first
    ref_lines = []
    for rs in board.get("ref_seeds", []):
        base, k = split_seed(rs)
        if done(rs):
            continue
        j = by.get(base) or by.get(ref)
        if j is None:
            ref_lines.append(f"# MISSING reference seed {rs}: no job for {base} or {ref} in {a.jobs[0]}")
            continue
        ref_lines += seed_job(j, k, "reference seed")

    for j in jobs:
        if j["name"] in skip:
            continue
        r = rows.get(j["name"])
        c = r.get("vs_wql_s0") if r else None
        if not c or "error" in c:
            drop.append(f"#   {j['name']}: no rung-1 score yet" + (f" ({c['error'][:80]})" if c else ""))
            continue
        hi, ratio = c["ci_fam"][1], c["ratio"]
        if hi < a.upper or ratio <= a.ratio or j["name"] in also:
            why = ("CI_fam upper < %.3g" % a.upper if hi < a.upper else "ratio <= %.3g" % a.ratio
                   if ratio <= a.ratio else "--also (manual)")
            keep.append((j, f"{fmt(c)}: {why}"))
        else:
            drop.append(f"#   {j['name']}: {fmt(c)}")

    out += [f"# Rung 2 (confirm), generated by tools/promote.py from {a.jobs[0]} + {a.board}",
            f"# reference: {ref} + {', '.join(board.get('ref_seeds', []))}; rung-1 (seed 0) rule: "
            f"wql CI_fam upper < {a.upper} or ratio <= {a.ratio}; seeds {a.seeds}"]
    out += ["# promoted:"] + [f"#   {j['name']}: {why}" for j, why in keep]
    if ref_lines:
        out += ["", "# reference seeds"] + ref_lines
    for k in seeds:
        out += ["", f"# seed {k}"]
        for j, _ in keep:
            out += seed_job(j, k)
    out += ["", "# not promoted (rung-1 evidence):"] + drop
    return out


def union(winners, flags):
    env, src, conflicts = {}, {}, []
    for w in winners:                                  # best first: first writer wins
        for k, v in flags[w].items():
            if k not in env:
                env[k], src[k] = v, w
            elif env[k] != v:
                conflicts.append(f"{k}: {src[k]}={env[k]} kept over {w}={v}")
    return env, conflicts


def rung3(a, board, rows):
    defs = {}
    for f in a.jobs:
        for j in parse(f):
            defs.setdefault(split_seed(j["name"])[0], j)
    ref = rows[board["ref"]]
    ref_group = {board["ref"], *(split_seed(s)[0] for s in board.get("ref_seeds", []))}
    cand, notes = [], []
    for r in board["rows"]:
        c = r.get("vs_wql", {})
        if r["name"] in ref_group or r["n_seeds"] < a.min_seeds or "error" in c or not c:
            continue
        ok = r["wql"][0] < ref["wql"][0] and c["ratio"] < 1 and c["ci_fam"][1] < 1
        if not ok:
            notes.append(f"#   {r['name']} (n={r['n_seeds']}): {fmt(c)}, mean {r['wql'][0]:.4f}: not confirmed")
        elif r["name"] not in defs:
            notes.append(f"#   {r['name']}: confirmed but no job definition in {a.jobs}")
        else:
            cand.append((c["ratio"], r["name"], c))
    cand.sort()
    winners = [n for _, n, _ in cand]
    flags = {w: {k: v for k, v in defs[w]["env"].items() if k not in DROP} for w in winners}
    out = [f"# Rung 3 (combine), generated by tools/promote.py --rung3 from {', '.join(a.jobs)} + {a.board}",
           f"# confirmed winners (>= {a.min_seeds} seeds, seed-mean better than {board['ref']}, CI_fam upper < 1):"]
    out += [f"#   {n} ({defs[n]['script']}): {fmt(c)}" for _, n, c in cand] or ["#   none"]
    out += ["# not confirmed:"] + (notes or ["#   none"])
    if not winners:
        return out + ["# nothing to combine yet"]
    combo, conflicts = union(winners, flags)
    out += ["# flag conflicts:"] + [f"#   {c}" for c in conflicts or ["none"]]
    scale = {"TSAR_STEPS": str(a.steps), "TSAR_BENCH": "1", "TSAR_D_MODEL": str(a.d_model),
             "TSAR_N_LAYERS": str(a.n_layers)}
    variants = [("base", {}), ("combo", combo)]
    if len(winners) >= 2:
        for w in winners:
            env = union([x for x in winners if x != w], flags)[0]
            same = next((t for t, e in variants if e == env), None)
            if same:                                   # w's flags are subsumed: no compute for a duplicate
                out.append(f"#   loo_{w} skipped: identical to {same} (its flags are covered by the other winners)")
            else:
                variants.append((f"loo_{w}", env))
    for k in a.r3_seeds:
        out += ["", f"# seed {k}"]
        for tag, env in variants:
            name = f"{a.prefix}{tag}" + (f"_s{k}" if k else "")
            out.append(line(name, a.script, scale | {"TSAR_SEED": str(k)} | env, mem=a.mem))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", help="rung-1 jobs file (rung 2); jobs files holding the winners' flags (rung 3)")
    ap.add_argument("--board", default=str(RUNS / "leaderboard.json"))
    ap.add_argument("-o", "--out", help="write here (default stdout)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--upper", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=0.985)
    ap.add_argument("--also", default="", help="rung 2: comma list promoted regardless (e.g. a calibration-only win)")
    ap.add_argument("--rung3", action="store_true")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--script", default="train_v2.py")
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--mem", type=int, default=2500, help="rung-3 mem= estimate (MB), a guess until one finishes")
    ap.add_argument("--prefix", default="r3_")
    a = ap.parse_args()
    a.r3_seeds = [0, 1]
    board, rows = load_board(a.board)
    if not a.rung3 and len(a.jobs) != 1:
        sys.exit("rung 2 takes exactly one rung-1 jobs file")
    text = "\n".join(rung3(a, board, rows) if a.rung3 else rung2(a, board, rows)) + "\n"
    if a.out:
        if os.path.abspath(a.out) in {os.path.abspath(j) for j in a.jobs}:
            sys.exit("refusing to overwrite an input jobs file")
        open(a.out, "w").write(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
