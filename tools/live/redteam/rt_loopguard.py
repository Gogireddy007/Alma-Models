"""Red-team guard on the autoresearch loop's editable file tools/live/wm_core.py.

The loop may edit model/training code in wm_core.py, but the metric, the folds, the normalisation and the window
construction also live there (fold_rows, Norm, windows, pinball, pinball_np, predict, QL/NQ/NT). This script hashes
the source of those protected objects in every git revision of wm_core.py (plus the working tree) and reports any
change against the first (baseline) revision; it also checks run.py / wm_store.py (harness) blob changes.

    .venv/bin/python tools/live/redteam/rt_loopguard.py
"""
import ast
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROTECTED = ["fold_rows", "Norm", "windows", "pinball", "pinball_np", "predict", "QL", "NQ", "NT", "nparams",
             "matched_cfg"]
HARNESS = ["tools/live/wm_harness.py", "tools/live/search/run.py", "tools/live/wm_store.py", "tools/live/search/pillars.py",
           "tools/live/search/policy.py"]


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout


def protected_hashes(src):
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"__syntax__": "error"}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for n in names:
            if n in PROTECTED:
                out[n] = hashlib.sha256(ast.dump(node).encode()).hexdigest()[:12]
    return out


def main():
    revs = git("log", "--format=%h %cI %s", "--reverse", "--", "tools/live/wm_core.py").strip().splitlines()
    base, rows = None, []
    for line in revs:
        h = line.split()[0]
        src = git("show", f"{h}:tools/live/wm_core.py")
        ph = protected_hashes(src)
        if base is None:
            base = ph
        diff = sorted(k for k in set(base) | set(ph) if base.get(k) != ph.get(k))
        rows.append((line[:90], diff))
    wt = open(os.path.join(ROOT, "tools/live/wm_core.py")).read()
    ph = protected_hashes(wt)
    if base is None:
        base = ph
    rows.append(("WORKING TREE", sorted(k for k in set(base) | set(ph) if base.get(k) != ph.get(k))))
    # after a50f1f6 the protected objects live in wm_harness.py: they must NOT be (re)defined in wm_core.py
    moved = "a50f1f6"
    hist = [r[0].split()[0] for r in rows[:-1]]
    if moved in hist:
        k = hist.index(moved)
        rows = [(r[0], [n for n in protected_hashes(git("show", f"{r[0].split()[0]}:tools/live/wm_core.py")
                                                   if r[0] != "WORKING TREE" else wt) if n in
                        ("fold_rows", "Norm", "windows", "pinball_np", "predict")]) for r in rows[k:]]
    bad = [r for r in rows if r[1]]
    print(f"wm_core.py revisions {len(revs)} (+ working tree); protected objects changed vs baseline in {len(bad)}")
    for r in bad:
        print("  CHANGED", r)
    for f in HARNESS:
        n = len(git("log", "--format=%h", "--", f).split())
        dirty = bool(git("status", "--porcelain", "--", f).strip())
        print(f"harness {f}: {n} commits, uncommitted changes: {dirty}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
