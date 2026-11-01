"""Run one researcher smoke test under the team's compute rules.

    uv run tools/smoke.py [--timeout 120] -- env TSAR_STEPS=30 python train_x.py

- One smoke test at a time across all agents: an exclusive lock on runs/.smoke.lock (waits for it).
- Waits until kern.memorystatus_level >= 25 before starting.
- Kills the command after --timeout seconds (default 120) and reports wall time and peak RSS.
"""
import argparse, fcntl, os, resource, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def level():
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    with open(ROOT / "runs" / ".smoke.lock", "w") as lock:
        print("smoke: waiting for lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while level() < 25:
            print(f"smoke: memorystatus_level {level()} < 25, waiting", flush=True)
            time.sleep(30)
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)   # own process group: a timeout kills uv AND python
        try:
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20     # bytes on macOS
        print(f"smoke: exit {rc}, {time.time() - t0:.0f}s, peak RSS {peak:.0f} MB", flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
