"""Staged architecture search (successive halving) on the DEV split, for the M3 8 GB box. Hana Kim, 2026-10-30.

    uv run tools/search.py init [--n 300] [--seed 0]        sample configs (OFAT around the incumbent + Sobol)
    uv run tools/search.py jobs R  [--partial] [--force]    write runs/jobs/search_r<R>.txt (R = 0..3)
    uv run tools/search.py ablate --around c0123            add one-flag-at-a-time variants of a config to R0
    uv run tools/search.py rank                             parse logs -> runs/search/leaderboard.md
    uv run tools/search.py proxycheck [--score]             runs/jobs/search_proxycheck.txt / rank correlation
    uv run tools/search.py preflight [--dry] [--only ID..]  smoke one config per axis value + the largest-memory
                                                            config per script (lessons L3/L4); `jobs` needs it
    uv run tools/search.py prune                            delete checkpoints of screens that were not promoted
    uv run tools/search.py flags                            TSAR_* keys of the search scripts not in the space

Rungs (all through jobq, which the lead runs; this tool never launches anything):
    R0  every config, --r0-steps (400) steps, TSAR_DEV=1, seed 0    -> ranked by DEV wql_rel (mase_rel tiebreak)
    R1  top --r1-keep (0.1, ~30 of 300) of R0, 2500 steps, TSAR_DEV=1 -> ranked by DEV wql_rel
    R2  top --r2-keep (8) of R1, 2500 steps, bench VAL (no TSAR_DEV)
    R3  top --r3-keep (3) of R2 by VAL, 10k steps x seeds 0,1,2, VAL (then leaderboard.py / compare.py CIs)
Budget (cost model in est_minutes, MPS serial): 300 sampled configs average ~3.4 min at 400 steps (long contexts
and big models dominate; the incumbent is ~1.5 min), so R0 ~17 h, R1 ~10 h, R2 ~2.5 h, R3 ~6-8 h.
Only R2 and R3 touch VAL, so VAL sees <= 8 + 3 search configs instead of hundreds (audit_v3 H-4). R0 also runs
the incumbent at seeds 1 and 2: their spread is the noise floor printed in the leaderboard.

Resumable: configs (runs/search/configs.jsonl) and each rung's membership (runs/search/state.json) are frozen
once written; rerunning `jobs R` rewrites the same file and jobq skips finished jobs (a job is done when its log
has dev_wql_rel:/val_wql_rel:). A rung is promoted only when every job of the previous rung is done or failed
(runs/<name>.failed), unless --partial. Every sampling and promotion is appended to runs/search/search.log.
"""
import argparse, json, math, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_space as SP  # noqa: E402

ROOT = SP.ROOT
RUNS = ROOT / "runs"
SD = RUNS / "search"
CONFIGS, STATE, LOG, BOARD = SD / "configs.jsonl", SD / "state.json", SD / "search.log", SD / "leaderboard.md"
STEPS = {0: 400, 1: 2500, 2: 2500, 3: 10000}
WQL_FAIL = 2.0            # lessons L6: wql_rel above this (or non-finite) is a failure, not a score
SEC_PER_STEP = 0.154      # syn5_only_s1 on MPS: 384.5 s / 2500 steps at ctx 256, d96, 2 layers
OVERHEAD_S = 25           # startup + data load + eval (DEV ~3 s, VAL ~22 s)


def log(msg):
    SD.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_configs():
    return [json.loads(l) for l in CONFIGS.read_text().splitlines() if l.strip()] if CONFIGS.exists() else []


def save_configs(cfgs):
    SD.mkdir(parents=True, exist_ok=True)
    CONFIGS.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cfgs))


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"rungs": {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, sort_keys=True))


def key(cfg):
    return json.dumps(SP.normalize(cfg), sort_keys=True)


def nondefault(cfg):
    c = SP.normalize(cfg)
    return {k: v for k, v in c.items() if v != SP.AXES[k]["values"][0]}


def cost_factor(cfg):
    """Rough per-step cost relative to the incumbent (tokens per batch x width x depth), for budgeting only."""
    c = SP.normalize(cfg)
    batch = SP.AXES["context"]["tie"]["TSAR_BATCH"][c["context"]]
    tok = (c["context"] + max(c["max_h"], c.get("long_h") or 0)) * batch / ((256 + 64) * 128)
    # calibrated on ctx512 (wave3 log: 0.32 s/step vs 0.154 at ctx256 -> 2.1x; tok 1.8 x 1.23); ctx 2048 was 11x on CPU
    ctx = tok * (c["context"] / 256) ** 0.3
    return ctx * (0.4 + 0.6 * (c["d_model"] / 96) * (c["n_layers"] / 2)) * (1 + 0.15 * (c["a2_slots"] - 1))


def est_minutes(cfg, steps):
    return (steps * SEC_PER_STEP * cost_factor(cfg) + OVERHEAD_S) / 60


def add(cfgs, seen, cfg, source, stats, prefer=None):
    cfg = SP.normalize(cfg)
    k = key(cfg)
    if k in seen:
        stats["dup"] += 1
        return False
    if SP.refused(cfg):
        stats["refused"] += 1
        return False
    script = SP.script_for(cfg, prefer)
    if not script:
        stats["no_script"] += 1
        return False
    seen.add(k)
    cfgs.append({"id": f"c{len(cfgs):04d}", "cfg": cfg, "source": source, "script": script})
    return True


def ofat(base):
    """Every single-axis move away from `base` (conditional axes included when their parent is active)."""
    out = []
    for k, a in SP.AXES.items():
        if not SP.active(base, k):
            continue
        out += [({**base, k: v}, f"ofat:{k}={v}") for v in a["values"] if v != base.get(k)]
    return out


def cmd_init(a):
    if CONFIGS.exists() and not a.force:
        sys.exit(f"{CONFIGS} exists (frozen); use `ablate` to add configs, or --force to resample from scratch")
    from scipy.stats import qmc
    cfgs, seen, stats = [], set(), {"dup": 0, "refused": 0, "no_script": 0}
    inc = SP.incumbent()
    add(cfgs, seen, inc, "incumbent", stats)
    for cfg, src in ofat(inc):
        add(cfgs, seen, cfg, src, stats)
    n_ofat = len(cfgs)
    names = list(SP.AXES)
    usable = [s for s in SP.SCRIPTS if SP.script_ok(s)]       # scripts with a TSAR_DEV hook; ~equal share each
    sob = qmc.Sobol(len(names) + 1, scramble=True, seed=a.seed)
    tries = 0
    while len(cfgs) < a.n and tries < 50 * a.n:
        for u in sob.random(64):
            tries += 1
            script = usable[min(int(u[0] * len(usable)), len(usable) - 1)]   # dim 0 picks the script family
            cfg = {k: SP.AXES[k]["values"][min(int(x * len(SP.AXES[k]["values"])), len(SP.AXES[k]["values"]) - 1)]
                   for k, x in zip(names, u[1:])}
            add(cfgs, seen, SP.restrict(cfg, script), f"sobol:{a.seed}", stats, prefer=script)
            if len(cfgs) >= a.n:
                break
    save_configs(cfgs)
    st = load_state()
    st.update({"seed": a.seed, "n": a.n, "r0_steps": a.r0_steps})
    save_state(st)
    log(f"init: scripts {usable}; {len(cfgs)} configs ({n_ofat} incumbent+OFAT, {len(cfgs) - n_ofat} Sobol seed {a.seed}, "
        f"{tries} Sobol draws; skipped {stats}); per script "
        f"{ {s: sum(c['script'] == s for c in cfgs) for s in SP.SCRIPTS} }")


def cmd_ablate(a):
    cfgs = load_configs()
    by = {c["id"]: c for c in cfgs}
    if a.around not in by:
        sys.exit(f"unknown config {a.around}")
    seen, stats, n0 = {key(c["cfg"]) for c in cfgs}, {"dup": 0, "refused": 0, "no_script": 0}, len(cfgs)
    for cfg, src in ofat(by[a.around]["cfg"]):
        add(cfgs, seen, cfg, f"ofat@{a.around}:{src[5:]}", stats)
    save_configs(cfgs)
    log(f"ablate around {a.around}: +{len(cfgs) - n0} configs (skipped {stats}); rerun `jobs 0` to queue them")


# ---------------------------------------------------------------------------------------------- logs and ranking

_SUM = re.compile(r"^(dev|val)_(wql_rel|mase_rel|cov80_error|subday_wql_rel|family_wql_rel):\s+(\S+)", re.M)


def job_name(cid, rung, seed=0):
    return f"search_r{rung}_{cid}" + (f"_s{seed}" if seed else "")


def parse(name):
    """{'status': done|failed|pending, metrics...} from runs/<name>.log."""
    f = RUNS / f"{name}.log"
    txt = f.read_text(errors="ignore") if f.exists() else ""
    m = {f"{s}_{k}": float(v) for s, k, v in _SUM.findall(txt)}
    t = re.search(r"^training_seconds:\s+(\S+)", txt, re.M)
    p = re.search(r"^num_params_M:\s+(\S+)", txt, re.M)
    st = "done" if ("dev_wql_rel" in m or "val_wql_rel" in m) else \
        "failed" if (RUNS / f"{name}.failed").exists() else "pending"
    why = None
    if st == "done":                                   # lessons L6: a blown-up metric is a failure, never ranked
        w = m.get("dev_wql_rel", m.get("val_wql_rel"))
        if not np.isfinite(w) or w > WQL_FAIL:
            st, why = "failed", f"wql_rel {w} > {WQL_FAIL} or non-finite"
    return {"status": st, "why": why, **m, "train_s": float(t.group(1)) if t else float("nan"),
            "params_M": float(p.group(1)) if p else float("nan")}


def rung_jobs(st, rung):
    """[(name, config id, seed)] of a frozen rung."""
    r = st["rungs"].get(str(rung))
    if not r:
        return []
    seeds = r.get("seeds", [0])
    out = [(job_name(cid, rung, s), cid, s) for cid in r["ids"] for s in seeds]
    out += [(job_name(cid, rung, s), cid, s) for cid, s in r.get("noise", [])]
    return out


def rung_scores(st, rung):
    """Per config: mean over seeds of the rung's split metrics; failed/missing -> inf. Sorted best first."""
    split = "dev" if rung <= 1 else "val"
    rows = {}
    for name, cid, s in rung_jobs(st, rung):
        rows.setdefault(cid, []).append((s, parse(name)))
    out = []
    for cid, rs in rows.items():
        main = [r for s, r in rs if rung == 3 or s == 0]
        done = [r for r in main if r["status"] == "done"]
        get = lambda k: float(np.mean([r.get(k, np.nan) for r in done])) if done else math.inf
        status = "done" if len(done) == len(main) else ("failed" if any(r["status"] == "failed" for r in main) else "pending")
        why = next((r["why"] for r in main if r.get("why")), None)
        if why:
            status = f"failed ({why})"
        out.append({"id": cid, "status": status, "n": len(done), "wql": get(f"{split}_wql_rel"),
                    "mase": get(f"{split}_mase_rel"), "cov": get(f"{split}_cov80_error"),
                    "subday": get("dev_subday_wql_rel") if split == "dev" else float("nan"),
                    "train_s": get("train_s"), "params_M": get("params_M")})
    out.sort(key=lambda r: (r["wql"], r["mase"], r["id"]))
    return out


def noise_floor(st):
    vals = [parse(job_name("c0000", 0, s)) for s in (0, 1, 2)]
    v = [r["dev_wql_rel"] for r in vals if r["status"] == "done"]
    return (float(np.std(v, ddof=1)), len(v)) if len(v) >= 2 else (float("nan"), len(v))


def env_line(cfg_rec, rung, seed):
    env = SP.env_of(cfg_rec["cfg"])
    env["TSAR_STEPS"] = str(STEPS[rung])
    if rung <= 1:
        env["TSAR_DEV"] = "1"
    if seed:
        env["TSAR_SEED"] = str(seed)
    env.update(SP.JOB_ENV)
    return " ".join(f"{k}={v}" for k, v in env.items())


def cmd_jobs(a):
    R = a.rung
    cfgs = {c["id"]: c for c in load_configs()}
    if not cfgs:
        sys.exit("no configs; run `init` first")
    st = load_state()
    STEPS[0] = st.get("r0_steps", STEPS[0])
    frozen = st["rungs"].get(str(R))
    if R == 0:
        ids = sorted(cfgs)                                     # R0 grows with `ablate`; ids are append-only
        st["rungs"]["0"] = {"ids": ids, "seeds": [0], "noise": [["c0000", 1], ["c0000", 2]]}
        if not frozen:
            log(f"R0: {len(ids)} configs at {STEPS[0]} steps on DEV (+ incumbent seeds 1, 2 as noise controls)")
        elif len(ids) != len(frozen["ids"]):
            log(f"R0: grew {len(frozen['ids'])} -> {len(ids)} configs")
    elif frozen and not a.force:
        log(f"R{R}: membership frozen ({len(frozen['ids'])} configs); rewriting the jobs file only")
    else:
        prev = rung_scores(st, R - 1)
        if not prev:
            sys.exit(f"rung {R - 1} has no jobs yet")
        open_ = [r for r in prev if r["status"] == "pending"]
        if open_ and not a.partial:
            sys.exit(f"rung {R - 1}: {len(open_)} jobs not finished (e.g. {open_[0]['id']}); wait, or --partial")
        ok = [r for r in prev if r["status"] == "done"]
        keep = {1: max(1, math.ceil(len(ok) * a.r1_keep)), 2: a.r2_keep, 3: a.r3_keep}[R]
        ids = [r["id"] for r in ok[:keep]]
        st["rungs"][str(R)] = {"ids": ids, "seeds": [0, 1, 2] if R == 3 else [0]}
        log(f"R{R}: promoted {len(ids)} of {len(ok)} finished R{R - 1} configs "
            f"({len(prev) - len(ok)} failed/pending): {' '.join(ids)}")
    save_state(st)
    jobs = rung_jobs(st, R)
    out = RUNS / "jobs" / f"search_r{R}.txt"
    split = "DEV (TSAR_DEV=1)" if R <= 1 else "bench VAL"
    mins = sum(est_minutes(cfgs[cid]["cfg"], STEPS[R]) for _, cid, _ in jobs)
    todo = [j for j in jobs if parse(j[0])["status"] != "done"]
    L = [f"# search rung {R}: {len(jobs)} jobs, {STEPS[R]} steps, scored on {split}. Written by tools/search.py "
         f"{time.strftime('%Y-%m-%d %H:%M')}.",
         f"# Estimated {mins / 60:.1f} h serial on MPS (--max 1); {len(todo)} not yet done. Run: "
         f"uv run tools/jobq.py runs/jobs/{out.name} --max 1",
         "# Do not edit: regenerate with `uv run tools/search.py jobs " + str(R) + "`."]
    passed = preflight_passed(a.stale_ok)
    if a.stale_ok:
        log(f"R{R}: --stale-ok: preflight passes on earlier script versions accepted")
    blocked = {}
    for name, cid, s in jobs:
        c = cfgs[cid]
        miss = [f"{k}={v}" for k, v in nondefault(c["cfg"]).items() if (c["script"], k, v) not in passed]
        if (c["script"], None, None) not in passed:
            miss.insert(0, "incumbent")
        if miss and not a.no_preflight:
            blocked[name] = miss
            continue
        L.append(f"{name} | {c['script']} | {env_line(c, R, s)} | mem={SP.mem_mb(c['cfg'])}")
    if blocked:
        vals = sorted({v for m in blocked.values() for v in m})
        L.insert(3, f"# {len(blocked)} jobs left out: axis values with no passing preflight on the current script "
                    f"(lessons L3): {' '.join(vals)}")
        log(f"R{R}: {len(blocked)} jobs left out, unpreflighted values: {' '.join(vals)}")
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(jobs)} jobs, ~{mins / 60:.1f} h serial)")
    cmd_rank(a)


def _fmt(x, n=4):
    return "-" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{n}f}"


def cmd_rank(a):
    cfgs = {c["id"]: c for c in load_configs()}
    st = load_state()
    sd, nsd = noise_floor(st)
    n_cfg = len(cfgs)
    L = [f"# Architecture search leaderboard ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"tools/search.py. {n_cfg} configs sampled (seed {st.get('seed')}). R0/R1 are ranked on the DEV split "
         "(tools/devsplit.py: held-out TRAIN families traffic, aus_electricity, nn5, tourism, car_parts; zero-shot, "
         "bench metrics vs seasonal naive), never on VAL. R2/R3 are bench VAL and are val-selected from "
         f"{n_cfg} DEV-screened configs (multiplicity disclosure, audit_v3 H-4). Sorted by wql_rel, mase_rel tiebreak.",
         "", f"R0 noise floor: sd of DEV wql_rel over incumbent seeds 0,1,2 = {_fmt(sd)} (n={nsd}). "
         "Differences under ~2 sd are not evidence.", ""]
    for R in range(4):
        rows = rung_scores(st, R)
        if not rows:
            continue
        split = "DEV" if R <= 1 else "VAL"
        inc = next((r for r in rows if r["id"] == "c0000"), None)
        n_done = sum(r["status"] == "done" for r in rows)
        L += [f"## R{R}: {STEPS[R] if R else st.get('r0_steps', STEPS[0])} steps, {split}, "
              f"{n_done}/{len(rows)} done", "",
              f"| # | id | {split} wql_rel | vs c0000 | mase_rel | cov80_err | subday wql | params_M | train_s | status | source | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows[:a.top if R <= 1 else None], 1):
            c = cfgs[r["id"]]
            rel = r["wql"] / inc["wql"] if inc and inc["status"] == "done" and r["status"] == "done" else float("nan")
            flags = " ".join(f"{k}={v}" for k, v in nondefault(c["cfg"]).items()) or "(incumbent)"
            L.append(f"| {i} | {r['id']} | {_fmt(r['wql'])} | {_fmt(rel, 3)} | {_fmt(r['mase'])} | {_fmt(r['cov'])} "
                     f"| {_fmt(r['subday'])} | {_fmt(r['params_M'], 3)} | {_fmt(r['train_s'], 0)} | {r['status']} "
                     f"| {c['source']} | {flags} |")
        if R <= 1 and len(rows) > a.top:
            L.append(f"| ... | {len(rows) - a.top} more | | | | | | | | | | |")
        L.append("")
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    BOARD.write_text("\n".join(L) + "\n")
    print(f"wrote {BOARD}")


# ---------------------------------------------------------------------------------------------- preflight (L3, L4)

PF = SD / "preflight.json"


def _sha(script):
    import hashlib
    return hashlib.sha256((ROOT / script).read_bytes()).hexdigest()


def preflight_cases():
    """[(pf id, script, axis, value, cfg)]: per script the incumbent, one config per non-default value of each axis
    the script implements (incumbent + that value; if that is refused, + the first other single move that makes it
    legal), and the largest-memory config of configs.jsonl (else the largest legal OFAT pair)."""
    out = []
    inc = SP.incumbent()
    for script in SP.SCRIPTS:
        if not SP.script_ok(script):
            continue
        sid = Path(script).stem
        out.append((f"pf_{sid}_incumbent", script, None, None, inc))
        for k, ax in SP.AXES.items():
            if script not in ax["scripts"]:
                continue
            for v in ax["values"][1:]:
                base = dict(inc)
                if ax.get("when"):
                    base[ax["when"][0]] = ax["when"][1]
                tries = [{**base, k: v}] + [{**base, k: v, k2: v2} for k2, a2 in SP.AXES.items() if k2 != k
                                             and script in a2["scripts"] for v2 in a2["values"][1:]]
                cfg = next((c for c in tries if not SP.refused(SP.normalize(c))
                            and SP.normalize(c).get(k) == v and SP.script_for(c, script) == script), None)
                if cfg is None:
                    print(f"preflight: no legal config for {script} {k}={v}")
                    continue
                out.append((f"pf_{sid}_{k}_{str(v).replace(',', '-')}", script, k, v, SP.normalize(cfg)))
        pool = [c["cfg"] for c in load_configs() if c["script"] == script and not SP.refused(SP.normalize(c["cfg"]))]
        if pool:
            big = max(pool, key=lambda c: (SP.mem_mb(c), cost_factor(c)))
            out.append((f"pf_{sid}_largest", script, "largest", None, big))
    return out


def preflight_passed(stale_ok=False):
    """{(script, axis, value)} that passed on the current sha256 of the script; (script, None, None) = incumbent.
    A case passes every non-default value it sets, so `largest` also covers its own values."""
    res = json.loads(PF.read_text()) if PF.exists() else {}
    sha = {s: _sha(s) for s in SP.SCRIPTS if (ROOT / s).exists()}
    ok = set()
    for r in res.values():
        if r["status"] == "pass" and (stale_ok or r["sha256"] == sha.get(r["script"])):
            ok |= {(r["script"], k, v) for k, v in nondefault(r["cfg"]).items()}
            if r["axis"] is None:
                ok.add((r["script"], None, None))
    return ok


def _peak_mb(path):
    txt = path.read_text(errors="ignore") if path.exists() else ""
    m = re.search(r"(\d+)\s+peak memory footprint", txt)
    return int(m.group(1)) / 2**20 if m else float("nan")


def cmd_preflight(a):
    """One smoke per case through tools/smoke.py (team lock, memory wait), on the job's device with JOB_ENV:
    a few real-batch training steps, then the DEV eval at the config's context length (eval path, L4)."""
    import subprocess
    cases = preflight_cases()
    res = json.loads(PF.read_text()) if PF.exists() else {}
    out_dir = RUNS / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for pid, script, k, v, cfg in cases:
        old = res.get(pid)
        fresh = old and old["status"] == "pass" and old["sha256"] == _sha(script) and old["cfg"] == cfg
        if (a.only and pid not in a.only) or (fresh and not a.redo and not a.only):
            continue
        todo.append((pid, script, k, v, cfg))
    print(f"preflight: {len(cases)} cases, {len(todo)} to run ({a.steps} steps + DEV eval each)")
    for pid, script, k, v, cfg in todo:
        env = SP.env_of(cfg)
        env.update({"TSAR_STEPS": str(a.steps), "TSAR_DEV": "1", **SP.JOB_ENV})
        envs = " ".join(f"{x}={y}" for x, y in env.items())
        if a.dry:
            print(f"  {pid:44s} {script} {envs}")
            continue
        log_p, time_p = out_dir / f"{pid}.log", out_dir / f"{pid}.time"
        cmd = ["uv", "run", "tools/smoke.py", "--timeout", str(a.timeout), "--", "env", *[f"{x}={y}" for x, y in env.items()],
               "/usr/bin/time", "-l", "-o", str(time_p), "python", script]
        t0 = time.time()
        with open(log_p, "w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        txt = log_p.read_text(errors="ignore")
        m = re.search(r"^dev_wql_rel:\s+(\S+)", txt, re.M)
        mps = re.search(r"^peak_mem_mb:\s+(\S+)", txt, re.M)
        wql = float(m.group(1)) if m else float("nan")
        ok = rc == 0 and m is not None and np.isfinite(wql)        # 3 steps: wql may exceed 2; it must be finite
        err = "" if ok else next((l for l in reversed(txt.splitlines()) if "Error" in l or "error" in l
                                  or "refus" in l or "timeout" in l), f"exit {rc}")[:200]
        res[pid] = {"script": script, "sha256": _sha(script), "axis": k, "value": v, "cfg": cfg,
                    "status": "pass" if ok else "fail", "error": err, "wall_s": round(time.time() - t0, 1),
                    "peak_footprint_mb": round(_peak_mb(time_p)), "peak_mps_mb": float(mps.group(1)) if mps else None,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
        PF.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        r = res[pid]
        print(f"  {pid:44s} {r['status'].upper():4s} {r['wall_s']:5.0f}s footprint {r['peak_footprint_mb']} MB "
              f"mps {r['peak_mps_mb']} MB {r['error']}", flush=True)
    if not a.dry:
        bad = [p for p, r in res.items() if r["status"] != "pass"]
        log(f"preflight: {sum(r['status'] == 'pass' for r in res.values())} pass, {len(bad)} fail: {' '.join(bad)}")


# ---------------------------------------------------------------------------------------------- proxy validity

# Finished 2500-step bench VAL runs (seed 0) whose configs train_search.py reproduces with TSAR_DEV=1 at R0
# steps. VAL wql_rel spans 0.685-0.805. Env is the run's own recipe (not BASE), so the pairing is exact.
PROXY = [
    ("syn5_only", "TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1"),
    ("b_curr", "TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5"),
    ("syn5_curr_50", "TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=0.5"),
    ("arch_koopman_j2", "TSAR_BENCH=1 TSAR_TRANS=koopman TSAR_JUMP=2"),
    ("ms_hdec", "TSAR_BENCH=1 TSAR_HDEC=1"),
    ("b_dyn_j2", "TSAR_BENCH=1 TSAR_JUMP=2"),
    ("v2_base", "TSAR_BENCH=1"),
    ("b_base512", "TSAR_BENCH=1 TSAR_CONTEXT=512 TSAR_BATCH=96"),
    ("arch_rssm", "TSAR_BENCH=1 TSAR_TRANS=rssm"),
    ("ms_ext", "TSAR_BENCH=1 TSAR_LONG_H=256"),
]


def cmd_proxycheck(a):
    steps = a.steps or load_state().get("r0_steps", STEPS[0])
    if not a.score:
        out = RUNS / "jobs" / "search_proxycheck.txt"
        L = ["# Proxy validity check for tools/search.py (Hana Kim). Each finished 2500-step VAL run (seed 0) is re-run",
             f"# as an R0 screen: train_search.py, {steps} steps, TSAR_DEV=1 (DEV families held out, DEV scored).",
             "# Then `uv run tools/search.py proxycheck --score` gives Spearman/Kendall of R0-DEV vs 2500-step VAL.",
             "# Trust R0 ranking only if Spearman >= 0.7 (n=10: p~0.02); else raise R0 steps or reweight DEV.",
             f"# ~{sum(est_minutes({}, steps) * (2.1 if 'CONTEXT=512' in e else 1) for _, e in PROXY):.0f} min serial on MPS. "
             "If Spearman < 0.7, rerun with `proxycheck --steps 600` (new job names)."]
        for ref, env in PROXY:
            L.append(f"search_px{steps}_{ref} | train_search.py | {env} TSAR_STEPS={steps} TSAR_DEV=1 "
                     f"{' '.join(f'{k}={v}' for k, v in SP.JOB_ENV.items())} | mem=2500")
        out.write_text("\n".join(L) + "\n")
        print(f"wrote {out}")
        return
    from scipy.stats import kendalltau, spearmanr
    xs, ys, rows = [], [], []
    for ref, _ in PROXY:
        d, v = parse(f"search_px{steps}_{ref}"), parse(ref)
        rows.append((ref, d.get("dev_wql_rel"), v.get("val_wql_rel"), d.get("dev_subday_wql_rel")))
        if d["status"] == "done" and "val_wql_rel" in v:
            xs.append(d["dev_wql_rel"]); ys.append(v["val_wql_rel"])
    for r in rows:
        print(f"{r[0]:16s} R0 DEV {_fmt(r[1])}  (subday {_fmt(r[3])})  2500-step VAL {_fmt(r[2])}")
    if len(xs) >= 4:
        print(f"n={len(xs)}  Spearman {spearmanr(xs, ys)[0]:.3f}  Kendall {kendalltau(xs, ys)[0]:.3f}")
    else:
        print(f"only {len(xs)} pairs finished")


def cmd_prune(a):
    st = load_state()
    n = 0
    for R in (0, 1, 2):
        nxt = st["rungs"].get(str(R + 1))
        if not nxt:
            continue
        keep = set(nxt["ids"])
        for name, cid, _ in rung_jobs(st, R):
            ck = ROOT / "data" / "ckpt" / f"{name}.pt"
            if cid not in keep and ck.exists() and parse(name)["status"] != "pending":
                ck.unlink(); n += 1
    log(f"prune: removed {n} checkpoints of non-promoted screens")


def cmd_flags(a):
    known = {ax["flag"] for ax in SP.AXES.values()} | set(SP.BASE) | {f for ax in SP.AXES.values() for f in ax.get("tie", {})}
    for s in SP.SCRIPTS:
        keys = SP.script_keys(s)
        if keys is None:
            print(f"{s}: not present yet")
            continue
        print(f"{s}: TSAR_DEV {'yes' if 'TSAR_DEV' in keys else 'NO (cannot screen)'}; "
              f"flags not in the search space: {' '.join(sorted(keys - known))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--n", type=int, default=300); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r0-steps", type=int, default=400); p.add_argument("--force", action="store_true")
    p = sub.add_parser("jobs"); p.add_argument("rung", type=int, choices=range(4))
    p.add_argument("--partial", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--r1-keep", type=float, default=0.1); p.add_argument("--r2-keep", type=int, default=8)
    p.add_argument("--r3-keep", type=int, default=3); p.add_argument("--top", type=int, default=60)
    p.add_argument("--no-preflight", action="store_true", help="write jobs for unpreflighted axis values (not advised)")
    p.add_argument("--stale-ok", action="store_true", help="accept passes recorded on an earlier sha256 of the script")
    p = sub.add_parser("ablate"); p.add_argument("--around", required=True)
    p = sub.add_parser("rank"); p.add_argument("--top", type=int, default=60)
    p = sub.add_parser("preflight"); p.add_argument("--dry", action="store_true")
    p.add_argument("--steps", type=int, default=3); p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--only", nargs="*", help="preflight ids to (re)run, e.g. pf_train_search_context_2048")
    p.add_argument("--redo", action="store_true", help="rerun cases that already passed on the current script")
    p = sub.add_parser("proxycheck"); p.add_argument("--score", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="R0 steps to validate (default: the search's, else 400)")
    sub.add_parser("prune"); sub.add_parser("flags")
    a = ap.parse_args()
    {"init": cmd_init, "jobs": cmd_jobs, "preflight": cmd_preflight, "ablate": cmd_ablate, "rank": cmd_rank, "proxycheck": cmd_proxycheck,
     "prune": cmd_prune, "flags": cmd_flags}[a.cmd](a)


if __name__ == "__main__":
    main()
