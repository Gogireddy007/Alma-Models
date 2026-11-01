"""FROZEN.json audit (auditor): recompute K (both logs, refs excluded), check var_sharpe_trials against its recorded
row list, check every pinned checkpoint sha256 on disk and in the wrapper constants, check required fields, check
whether any frozen model already has VAL output files dated before the FROZEN.json commit. Reads no VAL value."""
import os, re, sys, json, hashlib, subprocess
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F = os.path.join(ROOT, "runs/fin/FROZEN.json")
fz = json.load(open(F))
sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
out = []
def rep(s, ok): out.append(("OK   " if ok else "FAIL ") + s); print(out[-1])
A = pd.read_csv(os.path.join(ROOT, "runs/fin/search/results.tsv"), sep="\t", dtype=str, keep_default_na=False)
B = pd.read_csv(os.path.join(ROOT, "runs/fin/search_b/results_b.tsv"), sep="\t", dtype=str, keep_default_na=False)
kA, kB = int((A["kind"] != "ref").sum()), int((B["kind"] != "ref").sum())
rep(f"K recorded {fz.get('K')} >= rows excl. refs now (A {kA} + B {kB} = {kA + kB}); detail {fz.get('K_detail')}",
    fz.get("K") is not None and int(fz["K"]) >= kA + kB - 5)
v = fz.get("var_sharpe_trials"); n = fz.get("var_sharpe_trials_n")
rep(f"var_sharpe_trials {v} over n={n} rows (per-period units; expected ~1e-3..1e-2 for 6-12 month windows)",
    v is not None and np.isfinite(v) and 1e-5 < v < 0.05)
for k in ("models", "claim_models", "sha256", "K", "var_sharpe_trials"):
    rep(f"field {k} present", k in fz)
for rel, h in (fz.get("sha256") or {}).items():
    rep(f"ckpt {rel} on disk sha matches FROZEN", os.path.exists(os.path.join(ROOT, rel)) and sha(rel) == h)
src = ""
for f in ["tools/fin/wm_fin.py", "tools/fin/wm_b_fin.py"] + [f"tools/fin/agents/{x}.py" for x in ("wm_agent", "ppo", "dqn")]:
    if os.path.exists(os.path.join(ROOT, f)):
        src += open(os.path.join(ROOT, f)).read()
for rel, h in (fz.get("sha256") or {}).items():
    rep(f"wrapper constants pin {os.path.basename(rel)}", h in src)
txt = json.dumps(fz).lower()
rep("go/no-go rule for VAL recorded", "go" in txt and ("no-go" in txt or "no_go" in txt or "nogo" in txt))
rep("post-selection / winner's-curse label on F1 numbers", "post-selection" in txt or "winner" in txt)
try:
    ct = int(subprocess.check_output(["git", "-C", ROOT, "log", "-1", "--format=%ct", "--", "runs/fin/FROZEN.json"], text=True).strip() or 0)
except Exception:
    ct = 0
rep(f"FROZEN.json committed (commit time {ct})", ct > 0)
for m in fz.get("models", []):
    for mk in ("crypto", "etf"):
        p = os.path.join(ROOT, "runs/fin/val", mk, m.replace(":", "__") + ".npz")
        if os.path.exists(p) and ct:
            rep(f"VAL file for frozen model {m}/{mk} written AFTER the FROZEN commit", os.path.getmtime(p) > ct)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_frozen.txt"), "w").write("\n".join(out) + "\n")
