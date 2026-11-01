"""DEV backtest score of the frozen ensembles through the harness wrapper (per-origin seeding)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, wm_core as C, wm_data as D, wm_commodity as W
P = C.Panel("dev")
res = {}
for every in (5, 1):
    bo = C.backtest_origins(P, every=every)
    for name in sys.argv[1:]:
        pr = W.make_predict_ens([(os.path.join(D.ROOT, p), s) for p, s in W.FROZEN[name]])
        lev = pr.batch([np.exp(P.lp[t - 1023:t + 1, 0]) for t in bo], 20, list(map(float, C.Q9)))
        q = np.log(lev) - P.lp[bo, 0][:, None, None]
        r = C.evaluate(P, bo, q)
        if every == 5:
            r["ci90"] = C.score_ci(P, bo, q)
        res.setdefault(name, {})["weekly" if every == 5 else "daily"] = r
        print(name, every, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
json.dump(res, open(os.path.join(D.ROOT, "runs", "commodity", "search", "dev_final.json"), "w"), indent=1)
