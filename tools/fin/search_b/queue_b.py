"""Run a queue of (name, cfg) lines (JSONL: {"name":..., "cfg":{...}, "seed":0}) sequentially through loop_b.
Skips names already logged with fold EM (resumable). One process = one job (memory rule).
  nice -n 10 .venv/bin/python tools/fin/search_b/queue_b.py runs/fin/search_b/q_r1.jsonl
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_b as LB  # noqa

torch.set_num_threads(3)
done = set()
if os.path.exists(LB.TSV):
    for ln in open(LB.TSV):
        p = ln.split("\t")
        if len(p) > 6 and p[5] == "EM":
            done.add((p[2], p[4]))
for ln in open(sys.argv[1]):
    if not ln.strip():
        continue
    j = json.loads(ln)
    s = j.get("seed", 0)
    if (j["name"], str(s)) in done:
        continue
    LB.run_cfg(j["name"], j["cfg"], s, j.get("folds", ["E1", "E2", "E3"]), kind=j.get("kind", "wm_b"),
               log=lambda *a: None if str(a[0]).startswith("  step") else print(*a, flush=True))
