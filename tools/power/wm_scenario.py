"""APPLICATION DEMO (model simulation, NOT a validated forecast): wind scenarios through the frozen physical world model.

Question: "If tomorrow's wind output were X% lower than its last known (D-2, same hour) level, how would the model's
distribution of tomorrow's prices shift?"

Mechanism (an intervention on the latent state, not on the inputs): the physical WM rolls its latent system state
to day D and emits a residual-load estimate rl_hat(D,h). The scenario adds the lost wind to it,
    rl_hat(D,h) <- rl_hat(D,h) + X * wind(D-2,h),
and the price distribution is read out through the learned merit-order curve, which is monotone increasing in
residual load by construction (positive-weight sigmoid sum); the quantile spreads do not depend on rl_hat. So less
wind can never lower any quantile: the response is monotone by design, and this script checks it numerically.

Run on DEV target days (2025-01-01..2025-06-30, lead's panel via build_context; information set as in the harness):
    nice -n 10 .venv/bin/python tools/power/wm_scenario.py --model wm_power
Writes runs/power/scenario_demo.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

SCEN = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.3]          # wind change vs lagged level (-0.3 = 30% lower)


def p_up(q, levels, ref):
    """P(price > ref) from a quantile function (linear interpolation, clipped to the grid's tails)."""
    out = np.empty(q.shape[1])
    for h in range(q.shape[1]):
        out[h] = 1 - np.interp(ref[h], q[:, h], levels, left=0.0, right=1.0)
    return out


def main():
    import importlib.util
    import wm_power as WP
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wm_power")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-06-30")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("power_data_lead", os.path.join(HERE, "data.py"))
    LD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LD)
    dates, P, C = LD.train_panel()
    flags = {"dst_gap": np.array([], int), "dst_dup": np.array([], int)}
    ens = WP.Ensemble(WP.FROZEN[a.model])
    assert all(m["arch"] == "phys" for m in ens.meta["members"]), "scenario lever needs the physical WM"
    idx = [i for i in range(len(dates)) if a.start <= str(dates[i]) <= a.end and np.isfinite(P[i]).all()]
    ctxs = [LD.build_context(i, 40, dates, P, C, flags) for i in idx]
    prices = [c["prices"] for c in ctxs]
    covs = [c["covs"] for c in ctxs]
    dd = [{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in ctxs]
    lv = ens.QL
    res = {}
    Q = {}
    for x in SCEN:
        Q[x] = ens.quantiles_eur(prices, covs, dd, rl_shift_frac_wind=-x if x != 0 else None)   # wind -30% => rl += 0.3 wind
    base = Q[0.0]
    y1 = np.stack([p.reshape(-1, 24)[-1] for p in prices])
    for x in SCEN:
        q = Q[x]
        med_base = q[:, 6].mean(1)                                 # daily base price of the median path
        pu = np.array([p_up(q[n], lv, y1[n]).mean() for n in range(len(idx))])
        res[f"{x:+.0%}"] = {
            "mean_daily_median_EUR": float(med_base.mean()),
            "shift_vs_base_EUR": float((med_base - base[:, 6].mean(1)).mean()),
            "q10_mean_EUR": float(q[:, 2].mean()), "q90_mean_EUR": float(q[:, 10].mean()),
            "width80_mean_EUR": float((q[:, 10] - q[:, 2]).mean()),
            "mean_P_hour_up_vs_D-1": float(pu.mean()),
            "frac_days_daily_median_up_vs_D-1": float((med_base > y1.mean(1)).mean()),
        }
    # monotonicity: every quantile of every (day, hour) non-increasing in wind
    xs = sorted(SCEN)
    viol = 0
    tot = 0
    for x0, x1 in zip(xs[:-1], xs[1:]):
        d = Q[x1] - Q[x0]                                          # more wind -> should be <= 0
        viol += int((d > 1e-6).sum())
        tot += d.size
    # where does the shift land: by hour and by lagged-wind tercile
    shift30 = Q[-0.3][:, 6] - base[:, 6]                           # [n,24]
    wind_lag = np.stack([c.reshape(-1, 24, 4)[-2, :, 1] + c.reshape(-1, 24, 4)[-2, :, 2] for c in covs])   # D-2
    wl = np.nanmean(wind_lag, 1)
    terc = np.nanquantile(wl, [1 / 3, 2 / 3])
    groups = np.digitize(wl, terc)
    out = {"LABEL": "MODEL SIMULATION of the frozen world model; NOT a validated forecast. DEV days only.",
           "model": a.model, "members": ens.meta["members"], "days": [str(dates[idx[0]]), str(dates[idx[-1]])],
           "n_days": len(idx), "scenarios": res,
           "monotonicity": {"violations": viol, "comparisons": tot},
           "wind_-30%_median_shift_by_hour_EUR": [float(v) for v in shift30.mean(0)],
           "lagged_wind_tercile_bounds_MW": [float(t) for t in terc],
           "wind_-30%_daily_shift_by_lagged_wind_tercile_EUR": {
               ["low wind", "mid wind", "high wind"][g]: float(shift30[groups == g].mean()) for g in range(3)}}
    json.dump(out, open(os.path.join(ROOT, "runs", "power", "scenario_demo.json"), "w"), indent=1)
    print(json.dumps(out["scenarios"], indent=1))
    print("monotonicity", out["monotonicity"])
    print("by tercile", out["wind_-30%_daily_shift_by_lagged_wind_tercile_EUR"])


if __name__ == "__main__":
    main()
