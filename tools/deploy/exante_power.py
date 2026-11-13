"""Sealed EX-ANTE forecast of the next German day-ahead auction (delivery day = last known price day + 1).
Inference only, same guards as the deploy scripts. Run it BEFORE the auction clears (12:00 CET on D-1) and commit the
output; tools/deploy/score_exante.py grades it once the prices are published.

    .venv/bin/python tools/deploy/exante_power.py     # -> runs/deploy_2026_10_11/power/exante_<D>.json + preds/
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, os.path.join(ROOT, "tools", "power"))
MODELS = ["wm_power", "wm_power_direct", "wm_power_blend", "lear", "arx", "naive_epf", "chronos2", "chronos2_cov",
          "chronos_bolt_base", "tirex"]


def main():
    C.no_training()
    watch = C.ParamWatch()
    C.track_torch_modules(watch)
    C.ensure_ckpts()
    pins = C.verify_checkpoints("power")
    import data as D
    import models as M

    dates, P, Cv, flags = D.panel()
    last = int(np.where(np.isfinite(P).sum(1) >= 23)[0][-1])
    assert last == len(dates) - 1, "the panel already holds days after the last complete price day"
    target = dates[last] + np.timedelta64(1, "D")
    # the protocol gives the model complete fundamentals of D-2 (= dates[last - 1]); never forecast on a partial day
    n_cov = int(np.isfinite(Cv[last - 1]).all(1).sum())
    if n_cov < 23:
        print(f"skip: fundamentals of {dates[last - 1]} incomplete ({n_cov}/24 h); run again after that day ends")
        return
    # one empty row for the target day: no price and no fundamental of it (or of D-1) is ever read
    dates2 = np.append(dates, target)
    P2 = np.vstack([P, np.full((1, 24), np.nan)])
    C2 = np.concatenate([Cv, np.full((1, 24, Cv.shape[-1]), np.nan)])
    i = len(dates2) - 1
    ctx = D.build_context(i, 736, dates2, P2, C2, flags)          # asserts the information set with timestamps
    auction = D.forecast_time_utc(pd.Timestamp(target))
    now = pd.Timestamp.now(tz="UTC")
    assert now < auction, f"too late: the {target} auction cleared at {auction}"
    out, q_all = {}, {}
    for mn in MODELS:
        p = M.get(mn)
        kw = {"dates": [{"ctx_dates": ctx["ctx_dates"], "target_day": ctx["target_day"]}]} if getattr(p, "wants_dates", False) else {}
        q = np.asarray(p.batch([ctx["prices"]], [ctx["covs"]], 24, list(C.LV9), **kw), float)[0]
        q_all[mn] = q
        out[mn] = {"base_median": float(np.mean(q[4])), "hourly_median": [round(float(x), 2) for x in q[4]],
                   "peak_median_08_20": float(np.mean(q[4, 8:20]))}
        print(f"{mn}: base median {out[mn]['base_median']:.2f}", flush=True)
    tag = str(target)
    path, sh = C.save_preds("power", f"exante_{tag}", levels=C.LV9, **{f"q_{k}": v for k, v in q_all.items()})
    res = {"project": "power", "kind": "ex-ante (sealed before the auction)", "delivery_day": tag,
           "auction_clears_utc": str(auction), "forecast_made_utc": C.stamp(), "last_price_day": str(dates[last]),
           "last_fundamentals_day": str(dates[last - 1]), "prev_day_base_price": float(np.nanmean(P[last])),
           "checkpoints": pins, "models": out, "preds_file": path, "preds_sha256": sh,
           "data_hourly_csv_sha256": C.sha_file(os.path.join(ROOT, "data", "power", "hourly.csv"))}
    res["param_watch"] = watch.check()
    assert all(r["unchanged"] for r in res["param_watch"])
    res["checkpoints_after"] = C.verify_checkpoints("power")
    print("->", C.write_json("power", f"exante_{tag}.json", res))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f}s")
