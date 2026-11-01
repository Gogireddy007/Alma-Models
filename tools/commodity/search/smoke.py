import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np
import wm_core as C, wm_models as Mo
C.set_threads(2)
P = C.Panel("dev")
for arch, kw in [("obs", {}), ("sv", {}), ("rs", {}), ("sv", {"N": 7, "spreads": True}), ("obs", {"N": 7, "spreads": True})]:
    for abl in ("none", "direct"):
        cols = list(range(kw.get("N", 1))) if kw.get("N", 1) == 1 else [P.names.index(n) for n in C.XC_SERIES]
        m = Mo.ARCHS[arch](abl=abl, **kw)
        og = C.train_origins(P, cols=cols)
        w = torch.from_numpy(C.windows(P, og[:64], cols=cols)).float()
        t = time.time(); lo, info = m.loss(w, C.L_CTX, C.H_MAX); lo.backward(); tl = time.time() - t
        bo = C.backtest_origins(P)
        t = time.time(); q = C.predict_origins(m, P, bo, cols); te = time.time() - t
        r = C.evaluate(P, bo, q)
        print(arch, kw, abl, m.n_params(), f"loss {lo.item():.3f} step {tl:.2f}s eval {te:.1f}s", len(og), {k: round(v, 3) for k, v in r.items()})
