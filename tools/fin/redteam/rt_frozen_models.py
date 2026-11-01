"""Frozen models through the registry (auditor): randomised-future (NaN flips) invariance + D-1 sensitivity for every
FROZEN model on crypto and ETF DEV days; continual walk for wm_fin_cl; preflight negative test (a wrong pin must abort).
DEV only."""
import os, sys, json, copy
import torch  # noqa: F401
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
import rt_arch as RA
H = R.H
sys.path.insert(0, os.path.join(R.FIN, "search"))
import common as C
import models as M
import run_eval as RE
rng = np.random.default_rng(11)
fz = json.load(open(os.path.join(R.ROOT, "runs/fin/FROZEN.json")))
P, E = C.dev_panel("crypto"), C.dev_panel("etf")
res = []
def rep(s, ok): res.append(("OK   " if ok else "FAIL ") + s); print(res[-1], flush=True)
print("LOAD_ERRORS", M.LOAD_ERRORS)
for mn in fz["models"]:
    base, frozen = mn.split(":")[0], mn.endswith(":frozen")
    if base == "wm_fin_cl" and not frozen:
        i0 = P.index("2025-02-03"); days = list(range(i0, i0 + 8)); j = i0 + 4
        r0 = H.run(P, M.get(base), days); r1 = H.run(RA.rand_future(P, j, rng), M.get(base), days)
        k = days.index(j)
        rep(f"{mn} continual walk: decisions <= j identical, later differ, updates {r0['updates'].tolist()}",
            R.same(R.out_of(r0)[:k + 1], R.out_of(r1)[:k + 1]) and not R.same(R.out_of(r0)[k + 1:], R.out_of(r1)[k + 1:]))
        continue
    for mk, PP in (("crypto", P), ("etf", E)):
        for d in RA.DAYS:
            i = min(PP.index(d), len(PP.days) - 1)
            m = M.get(base)
            o0 = R.out_of(H.run(PP, m, [i], online=False))
            o1 = R.out_of(H.run(RA.rand_future(PP, i, rng), M.get(base), [i], online=False))
            o2 = R.out_of(H.run(RA.rand_future(PP, i, rng, from_prev=True), M.get(base), [i], online=False))
            fin = np.isfinite(o0).mean()
            rep(f"{mn} {mk} {d}: future-invariant {R.same(o0, o1)}, D-1 sensitive {not R.same(o0, o2)}, finite {fin:.2f}",
                R.same(o0, o1) and (not R.same(o0, o2)))
# preflight negative test: corrupt one pin in memory
bad = copy.deepcopy(fz); k0 = next(iter(bad["sha256"])); bad["sha256"][k0] = "0" * 64
RE.frozen = lambda: bad
try:
    RE.preflight(fz["models"]); rep("preflight aborts on a wrong pin", False)
except SystemExit as e:
    rep(f"preflight aborts on a wrong pin ({str(e)[:90]})", True)
RE.frozen = lambda: fz
try:
    RE.preflight(fz["models"]); rep("preflight passes with the committed FROZEN.json", True)
except SystemExit as e:
    rep(f"preflight passes with the committed FROZEN.json ({e})", False)
open(os.path.join(R.HERE, "rt_frozen_models.txt"), "w").write("\n".join(res) + "\n")
