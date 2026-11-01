"""VALIDATION (2025-07-01..2025-10-31), run ONCE for the frozen candidate and its twin. Reads the VAL cache only
(coed=2025-10-31; asserted). Same metric as the DEV search. Usage: run_val.py name1 name2 (names in FROZEN)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import wm_core as C, wm_data as D, wm_commodity as W

OUT = os.path.join(D.ROOT, "runs", "commodity", "search", "val_results.json")
assert not os.path.exists(OUT), "VALIDATION already run once"
P = C.Panel("val")
assert P.dates.max() <= D.VAL_END
res = {"ts": time.strftime("%Y-%m-%d %H:%M"), "models": {}}
for every in (5, 1):
    bo = C.backtest_origins(P, start=D.VAL_START, end=D.VAL_END, every=every)
    assert P.dates[bo.min()] >= D.VAL_START and P.dates[bo.max() + 20] <= D.VAL_END
    for name in sys.argv[1:] + ["rw"]:
        if name == "rw":
            q = C.rw_quantiles(P, bo)
        else:
            pr = W.make_predict_ens([(os.path.join(D.ROOT, p), s) for p, s in W.FROZEN[name]])
            ctxs = [np.exp(P.lp[t - 1024 + 1:t + 1, 0]) for t in bo]          # levels up to and incl. origin day
            lev = pr.batch(ctxs, 20, list(map(float, C.Q9)))
            q = np.log(lev) - P.lp[bo, 0][:, None, None]
        r = C.evaluate(P, bo, q)
        if every == 5:
            r["ci90"] = C.score_ci(P, bo, q)
        res["models"].setdefault(name, {})["weekly" if every == 5 else "daily"] = r
        print(name, every, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
res["origins_weekly"] = [str(P.dates[t].date()) for t in C.backtest_origins(P, start=D.VAL_START, end=D.VAL_END)]
json.dump(res, open(OUT, "w"), indent=1)
