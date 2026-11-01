"""Apply the Am1 holdout rule mechanically and write <pillar>_final.json.
wm: best F1 P1 among A's top-3 (ties -> lower P3); confirmed iff >= 0.6779 (logged d64 round-1 row), else START.
cl: best F1 S_C among top-3; confirmed iff S_C > 0, else START rule.
rl: best F1 held-out Sharpe among top-3; confirmed iff > F1 Sharpe of wm_voltgt (same WM), else START (RL0)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
p = sys.argv[1]
res = json.load(open(os.path.join(C.OUT, f"holdout_{p}.json")))
st = json.load(open(os.path.join(C.OUT, f"loop_{p}.json")))
start = next(h["key"] for h in st["hist"] if h["decision"] == "START")
if p == "wm":
    best = sorted(res, key=lambda r: (-r[2], r[3]))[0]
    ok = best[2] >= 0.6779
elif p == "cl":
    best = sorted(res, key=lambda r: (-r[2], r[3]))[0]
    ok = best[2] > 0
else:
    ref = [r for r in res if r[0] == "voltgt_ref"][0][2]
    cand = [r for r in res if r[0] != "voltgt_ref"]
    best = sorted(cand, key=lambda r: -r[2])[0]
    ok = best[2] > ref
out = {"key": best[0] if ok else start, "confirmed": bool(ok), "best_holdout": best, "start_key": start,
       "holdout_rows": res, "note": "F1 numbers are post-selection (RT-11)"}
json.dump(out, open(os.path.join(C.OUT, f"{p}_final.json"), "w"), indent=1)
print(json.dumps(out))
