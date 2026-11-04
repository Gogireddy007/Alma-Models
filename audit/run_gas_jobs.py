"""Run gas DEV-backtest jobs (seed|cfg|tag per line) 3 at a time; logs in audit/logs/."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "repo")
jobs = [l.rstrip("\n").split("|") for l in open(sys.argv[1]) if l.strip()]
def run(j):
    seed, cfg, tag = j
    env = dict(os.environ, OMP_NUM_THREADS="3")
    with open(f"{HERE}/logs/{tag}.log", "w") as fh:
        r = subprocess.run(["nice", "-n", "10", f"{REPO}/.venv/bin/python", "tools/gas/search/run.py", "bt", "--cfg", cfg,
                            "--seed", seed, "--tag", tag], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print("DONE", tag, r.returncode, flush=True)
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as ex:
    list(ex.map(run, jobs))
