"""Reference rows for the DEV leaderboard: random walk variants (no training). Appends to results.tsv."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import wm_core as C
from run_exp import TSV, COLS

P = C.Panel("dev")
bo = C.backtest_origins(P)
r = np.diff(P.lp[:, 0])
H = C.H_MAX
sq = np.sqrt(np.arange(1, H + 1))[None, None, :]


def rows():
    yield "ref_rw250", C.rw_quantiles(P, bo, H)
    for n in (20, 60):
        sig = np.array([np.sqrt(np.mean(r[t - n:t] ** 2)) for t in bo])
        yield f"ref_rw{n}", C.Z9[None, :, None] * sig[:, None, None] * sq
    lam = 0.94
    v = np.zeros(len(r)); v[0] = r[:250].var()
    for i in range(1, len(r)):
        v[i] = lam * v[i - 1] + (1 - lam) * r[i - 1] ** 2       # v[i] uses returns < i : forecast of r[i]
    sig = np.sqrt(v[bo])                                         # r[bo] = lp[bo+1]-lp[bo], uses r[:bo]
    yield "ref_ewma94", C.Z9[None, :, None] * sig[:, None, None] * sq
    # momentum drift: median shifted by the sign of the past-20d return (classical trend-following reference)
    sig = np.array([np.sqrt(np.mean(r[t - 250:t] ** 2)) for t in bo])
    mom = np.sign(P.lp[bo, 0] - P.lp[bo - 20, 0])
    yield "ref_mom20", C.Z9[None, :, None] * sig[:, None, None] * sq + (mom * 0.1 * sig)[:, None, None] * sq


for name, q in rows():
    m = C.evaluate(P, bo, q)
    lo, hi = C.score_ci(P, bo, q)
    row = {k: "" for k in COLS}
    row.update(ts=time.strftime("%Y-%m-%d %H:%M"), name=name, arch="ref", abl="-", params=0, steps=0, train_s=0,
               ci5=round(lo, 4), ci95=round(hi, 4), status="ok", cfg="{}", note="no training")
    row.update({k: round(v, 4) for k, v in m.items() if k in COLS})
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(row[k]) for k in COLS) + "\n")
    print(name, json.dumps({k: round(v, 3) for k, v in m.items()}))
