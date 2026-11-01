"""Final refit of the frozen recipe (docs/gas_search.md section 0, freeze step 4): the chosen recipe trained with the
same steps and seeds on ALL storage.train_weeks() (released <= 2025-06-30), holdout calibration as in the backtest
(cal = 1: last 20% of the training weeks), one checkpoint per model holding its 3 seed members.

    nice -n 10 .venv/bin/python tools/gas/search/refit.py --cfg '<recipe json>' --name wm_gas --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R        # noqa: E402
import feats as FT     # noqa: E402
import data as D       # noqa: E402
import storage as S    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    cfg = {**R.DEFAULT, **json.loads(a.cfg)}
    z = R.load_cache()
    idx = np.arange(len(z["items"]))
    items = [z["items"][i] for i in idx]
    labs = [z["lab"][c["k"]] for c in items]
    assert max(l["avail"] for l in labs) <= D.TRAIN_END
    sc = R.scales_of(idx)
    fcsd = R.fc_err_sd(idx, D.TRAIN_END + np.timedelta64(1, "D"))
    f0 = FT.feats(items[0], sc)
    members = []
    t0 = time.time()
    for s in [int(x) for x in a.seeds.split(",")]:
        m, npar, cal = R.fit(cfg, items, labs, sc, s)
        members.append({"seed": s, "state": m.state_dict(), "cal": cal})
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)
    ck = {"cfg": cfg, "scales": sc, "fcsd": fcsd, "members": members, "g_dim": int(f0["g"].shape[0]),
          "u_dim": int(f0["u"].shape[1]), "n_params": int(npar), "train_end": str(D.TRAIN_END.date()),
          "n_train_weeks": int(len(set(c["k"] for c in items))),
          "train_release_range": [str(min(l["avail"] for l in labs).date()), str(max(l["avail"] for l in labs).date())]}
    out = os.path.join(R.OUT, "ckpt", f"{a.name}.pt")
    torch.save(ck, out)
    print(out, R.FT.__name__, flush=True)


if __name__ == "__main__":
    main()
