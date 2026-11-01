"""Resumable, memory-aware job queue for the M3 8GB box.

    uv run tools/jobq.py runs/jobs/rung2.txt [--max 1] [--reserve 1800] [--dry]

Jobs file, one per line (blank lines and # comments ignored):
    name | script.py | KEY=VAL KEY=VAL ... [| after=<name>] [| mem=<MB>] [| run=<TSAR_RUN>]

- A job is DONE when runs/<name>.log contains a `val_wql_rel:` line (`dev_wql_rel:` for TSAR_DEV search screens), so relaunching after a crash or
  reboot skips finished work and reruns only what was interrupted (its checkpoint is deleted first).
- Standby block (lessons L2: the queue must never idle): a line `# standby` marks every job below it as standby.
  A standby job is admitted only once no main job (above the marker) is pending or running, so it fills idle
  time and never delays main work. Add new main jobs ABOVE the marker (the file is re-read live).
- `run=` sets TSAR_RUN (default: name); use it for inference-only re-scores of another job's checkpoint,
  together with `after=` so the re-score waits for that job.
- Jobs start in file order. Default --max 1: two jobs at once thrashed swap on this box (23:22, 5x slower).
  Admission pauses (logged) while swap used > --swap-max MB (6144) or kern.memorystatus_level < --min-level
  (20), whatever is running; running jobs are never killed.
- A job is admitted only while fewer than --max jobs run AND the kernel's
  available memory (kern.memorystatus_level) minus its estimate stays above --reserve MB. The estimate is
  `mem=`, else 1.2x the peak footprint of the job's earlier run (runs/<name>.time), else 1.2x the
  peak_mem_mb of its earlier log (MPS only, an underestimate), else 1500 MB.
- Every job runs under nice -n 10 and /usr/bin/time -l (-> runs/<name>.time); the DONE/FAIL line reports
  its peak footprint (incl. MPS and swapped pages) and peak RSS. The whole queue holds a caffeinate assertion.
- Provenance: at START runs/<name>.prov.json gets git HEAD + dirty flag, sha256 of the script,
  tools/bench.py, prepare.py and tools/jobq.py, the job's TSAR_*/PYTORCH_* env and the start time; at
  DONE/FAIL it gains the sha256 of data/ckpt/<TSAR_RUN>.pt, the exit code and the end time.
- --dry prints the admission plan for the current free memory and exits without starting anything.
Status lines go to stdout; per-job output to runs/<name>.log.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SEALED = {"TSAR_SPLIT", "TSAR_EVAL_ONLY", "TSAR_CKPT_SHA256", "TSAR_TTA_STEPS"}   # never inherited from the shell


def parse(path):
    jobs, standby = [], False
    for line in Path(path).read_text().splitlines():
        if re.match(r"\s*#\s*standby\b", line, re.I):
            standby = True
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        job = {"name": parts[0], "script": parts[1], "env": {}, "after": None, "mem": None, "run": parts[0],
               "standby": standby}
        for kv in parts[2].split() if len(parts) > 2 else []:
            k, v = kv.split("=", 1)
            job["env"][k] = v
        for opt in parts[3:]:
            k, v = opt.split("=", 1)
            job[k] = int(v) if k == "mem" else v
        jobs.append(job)
    return jobs


def done(name):
    log = RUNS / f"{name}.log"
    return log.exists() and re.search(r"^(val|dev)_wql_rel:", log.read_text(errors="ignore"), re.M) is not None


def peak_mb(name):
    """(peak footprint, peak RSS) in MB from /usr/bin/time -l, or None if the job has no .time file."""
    f = RUNS / f"{name}.time"
    if not f.exists():
        return None
    txt = f.read_text(errors="ignore")
    get = lambda k: (lambda m: int(m.group(1)) / 2**20 if m else float("nan"))(re.search(rf"(\d+)\s+{k}", txt))
    return get("peak memory footprint"), get("maximum resident set size")


def sha256(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def prov_start(job, env):
    prov = {"name": job["name"], "script": job["script"], "run": job["run"], "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "sha256": {f: sha256(f) for f in (job["script"], "tools/bench.py", "prepare.py", "tools/jobq.py")},
            "env": {k: v for k, v in sorted(env.items()) if k.startswith(("TSAR_", "PYTORCH_"))},
            "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (RUNS / f"{job['name']}.prov.json").write_text(json.dumps(prov, indent=1) + "\n")


def prov_end(name, rc):
    f = RUNS / f"{name}.prov.json"
    if not f.exists():
        return
    prov = json.loads(f.read_text())
    prov.update(ckpt_sha256=sha256(f"data/ckpt/{prov['run']}.pt"), exit=rc, done=done(name),
                end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    f.write_text(json.dumps(prov, indent=1) + "\n")


def mem_estimate(job):
    if job["mem"]:
        return job["mem"]
    pk = peak_mb(job["name"])
    if pk and pk[0] == pk[0]:
        return pk[0] * 1.2
    log = RUNS / f"{job['name']}.log"
    if log.exists():
        m = re.search(r"^peak_mem_mb:\s+([\d.]+)", log.read_text(errors="ignore"), re.M)
        if m:
            return float(m.group(1)) * 1.2
    return 1500


def free_mb():
    """Kernel's available-memory estimate (what memory_pressure reports), in MB."""
    q = lambda k: int(subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout)
    return q("kern.memorystatus_level") / 100 * q("hw.memsize") / 2**20


def plan(a):
    """Dry run: which pending jobs main() would admit now, in file order. A job with a live process (its
    /usr/bin/time -o runs/<name>.time) counts as running under another queue (its memory is already out
    of `free`); each START is assumed to take its full estimate."""
    free = free_mb()
    why = pressure(a)
    print(f"free {free:.0f}MB, reserve {a.reserve:.0f}MB, max {a.max}" + (f"; PAUSED: {why}" if why else ""))
    slots, avail = (0 if why else a.max), free
    jobs = parse(a.jobs)
    main_open = any(not j["standby"] and not done(j["name"]) and not (RUNS / f"{j['name']}.failed").exists()
                    for j in jobs)
    for j in jobs:
        if done(j["name"]) or (RUNS / f"{j['name']}.failed").exists():
            continue
        est = mem_estimate(j)
        live = subprocess.run(["pgrep", "-f", f"{j['name']}.time"], capture_output=True).returncode == 0
        if live:                                                   # a live queue's job: already in `free`
            why = "RUNNING"
            slots -= 1
        elif j["after"] and not done(j["after"]):
            why = f"wait after={j['after']}"
        elif j["standby"] and main_open:
            why = "standby (after main jobs)"
        elif slots <= 0:
            why = "wait slot"
        elif slots < a.max and avail - est < a.reserve:      # main() always admits the first job
            why = f"wait memory ({avail - est:.0f}MB < reserve)"
        else:
            why = "START"
            slots, avail = slots - 1, avail - est
        print(f"  {j['name']:16s} {j['script']:16s} est {est:5.0f}MB  {why}")
        if why.startswith("wait memory"):
            slots = 0                                              # main() keeps file order
    return 0


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def level():
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)


def pressure(a):
    """Reason to hold admission, or None."""
    sw, lv = swap_used_mb(), level()
    if sw > a.swap_max or lv < a.min_level:
        return f"swap {sw:.0f}MB (max {a.swap_max:.0f}), memorystatus_level {lv} (min {a.min_level})"
    return None


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


class _Adopted:
    """A job started by an earlier runner; poll() mirrors Popen's."""
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        if _alive(self.pid):
            return None
        self.returncode = 0
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--reserve", type=float, default=1800)
    ap.add_argument("--swap-max", type=float, default=6144, help="pause admission above this swap use (MB)")
    ap.add_argument("--min-level", type=int, default=20, help="pause admission below this memorystatus_level")
    ap.add_argument("--dry", action="store_true", help="print the admission plan and exit")
    a = ap.parse_args()
    if a.dry:
        return plan(a)
    caff = subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
    running = {}                                                   # name -> (Popen, start, est)
    backups = {}                                                   # re-score name -> (ckpt, backup)
    for pidf in RUNS.glob("*.pid"):                                # adopt jobs a previous runner left running
        name, pid = pidf.stem, int(pidf.read_text() or 0)
        if not done(name) and _alive(pid):
            running[name] = (_Adopted(pid), time.time(), 0)
            print(f"{time.strftime('%H:%M:%S')} ADOPT {name} (pid {pid})", flush=True)
        else:
            pidf.unlink()
    paused = False                                                 # admission held for memory pressure
    while True:
        jobs = parse(a.jobs)                                       # re-read: jobs may be appended live
        for name, (p, t0, _) in list(running.items()):
            if p.poll() is not None:
                ok = done(name)
                if not ok:                                         # don't retry in a loop; delete the marker to retry
                    (RUNS / f"{name}.failed").write_text(f"exit {p.returncode}\n")
                prov_end(name, p.returncode)
                if name in backups:                                # re-score: put the parent's checkpoint back
                    src, bak = backups.pop(name)
                    shutil.move(bak, src)
                pk = peak_mb(name)
                mem = f" peak {pk[0]:.0f}MB rss {pk[1]:.0f}MB" if pk else ""
                print(f"{time.strftime('%H:%M:%S')} {'DONE' if ok else 'FAIL'} {name} {time.time() - t0:.0f}s{mem}", flush=True)
                del running[name]
                (RUNS / f"{name}.pid").unlink(missing_ok=True)
        pending = [j for j in jobs if j["name"] not in running and not done(j["name"])
                   and not (RUNS / f"{j['name']}.failed").exists()]
        if not pending and not running:
            print(f"{time.strftime('%H:%M:%S')} ALL_DONE", flush=True)
            break
        why = None
        if pending and len(running) < a.max:                       # only when a slot is free to admit into
            why = pressure(a)
            if bool(why) != paused:                                # log pause/resume once per change
                print(f"{time.strftime('%H:%M:%S')} {'PAUSE ' + why if why else 'RESUME'}", flush=True)
                paused = bool(why)
        main_open = any(not j["standby"] for j in pending) or \
            any(not j["standby"] for j in jobs if j["name"] in running)
        for j in pending if not why else []:
            if len(running) >= a.max:
                break
            if j["standby"] and main_open:                         # standby only fills idle time
                break
            if j["after"] and not done(j["after"]):
                continue
            est = mem_estimate(j)
            committed = sum(e for _, _, e in running.values())
            if running and free_mb() - est < a.reserve:
                break                                              # keep file order; wait for memory
            if not (ROOT / j["script"]).exists():
                print(f"missing script for {j['name']}: {j['script']}", flush=True)
                (RUNS / f"{j['name']}.failed").write_text("missing script\n")
                continue
            if j["run"] == j["name"]:
                (ROOT / "data" / "ckpt" / f"{j['name']}.pt").unlink(missing_ok=True)
            if j["env"].get("TSAR_SPLIT", "val") != "val":     # EVAL_ONLY on VAL is fine (audit v3.1 R-3)
                print(f"refused {j['name']}: jobq runs VAL screens only (audit_v3 B-1)", flush=True)
                (RUNS / f"{j['name']}.failed").write_text("refused: non-val split\n")
                continue
            inherited = {k: v for k, v in os.environ.items() if k not in SEALED}
            env = {**inherited, **j["env"], "TSAR_RUN": j["run"], "TSAR_SPLIT": "val"}
            prov_start(j, env)
            if j["run"] != j["name"]:                              # re-score: keep the parent's ckpt sha (audit v3.1 R-2)
                src = ROOT / "data" / "ckpt" / f"{j['run']}.pt"
                if src.exists():
                    backups[j["name"]] = (src, src.with_suffix(".pt.parent"))
                    shutil.copy2(src, backups[j["name"]][1])
            log = open(RUNS / f"{j['name']}.log", "w")
            p = subprocess.Popen(["nice", "-n", "10", "uv", "run", "--", "/usr/bin/time", "-l", "-o",
                                  str(RUNS / f"{j['name']}.time"), "python", j["script"]], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[j["name"]] = (p, time.time(), est)
            (RUNS / f"{j['name']}.pid").write_text(str(p.pid))
            print(f"{time.strftime('%H:%M:%S')} START {j['name']} ({j['script']}, est {est:.0f}MB, "
                  f"free {free_mb():.0f}MB, committed {committed:.0f}MB)", flush=True)
            time.sleep(15)                                         # let it allocate before the next check
        time.sleep(10)
    caff.terminate()


if __name__ == "__main__":
    sys.exit(main())
