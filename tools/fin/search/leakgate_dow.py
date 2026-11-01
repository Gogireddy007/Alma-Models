"""Train the E3 fold model of (WM incumbent + dow=True) through Loop.fold_model (cached, reused by the loop) so the
dow family can pass test_leak --family rssm BEFORE any of its scores count (Am2). Prints the ckpt path."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import loop as LP
st = json.load(open(os.path.join(C.OUT, "loop_wm.json")))
cfg = dict(st["incumbent"], dow=True)
L = LP.Loop("wm", print)
L.fold_model(cfg, LP.FOLDS[2])
C.log_row({"kind": "leakgate", "name": f"wm_{LP.key(cfg)}_E3", "config": cfg, "seed": 0, "fold": "E3",
           "status": "trained", "note": "dow family leak-gate checkpoint (not scored here; scored by the loop)"})
print(os.path.join(C.CK, f"loop_{LP.key(cfg)}_none_s0_E3.pt"))
