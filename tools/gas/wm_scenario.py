"""Cold-snap scenario demo for the frozen gas world model (simulation only; NOT scored, NOT a forecast claim).

    nice -n 10 .venv/bin/python tools/gas/wm_scenario.py      # -> runs/gas/scenario_demo.json

Setting: one DEV winter target week (released <= 2025-06-30; no VAL/TEST data), origin 'thu'. Baseline = the
weather exactly as known at the origin. Scenario = every day of the target week made colder by +2 sd of the daily
HDD anomaly for that time of year (sd from the context's own observed history: same calendar window +-15 days,
previous 9 years). 1,000 Monte Carlo paths per arm: CPC forecast-error noise on the forecast days (training-set
sd by lead, stored in the checkpoint) rolled through each seed member's daily latent state, and one draw from
that path's predictive quantile function (residual uncertainty). Reported: daily L48 storage path (median and
10/90% bands of the member-averaged daily flows cumulated from the last released level), weekly dS distribution,
and P(draw larger than the 5-year-average draw) = P(dS_W < avg5_W) (withdrawal season).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "search"))
import storage as S      # noqa: E402
import wm_gas as WG      # noqa: E402

NPATHS = 1000
# Week choice rule (fixed after a first run on 2025-01-10, an already very cold week where P = 1.0 in both arms, i.e.
# uninformative): the DEV winter week (target week Dec 2024 - Feb 2025) whose BASELINE median deviation from avg5
# is closest to zero, so the scenario starts from a near-normal week. Realised values are never read.
WINTER = ("2024-12-01", "2025-02-28")


def hdd_anom_sd(c, days, half=15, years=9):
    od = pd.to_datetime(np.asarray(c["obs_daily_dates"]).astype(str))
    obs = pd.Series(np.asarray(c["obs_daily"], float)[:, 0], index=od)
    doy = obs.index.dayofyear
    an = []
    for x in days:
        for y in range(1, years + 1):
            ctr = x - pd.DateOffset(years=y)
            win = obs[(obs.index >= ctr - pd.Timedelta(days=half)) & (obs.index <= ctr + pd.Timedelta(days=half))]
            if len(win):
                an.append(win.values - win.mean())
    return float(np.std(np.concatenate(an)))


def simulate(ens, c, shift_hdd, rng):
    import torch
    b = ens.batch_tensors([c])
    sd = torch.tensor(ens.scales["dev_sd"], dtype=torch.float32)
    lead = np.rint(b["u"][0, :, 7].numpy() * 7).astype(int)
    fsd = np.asarray(ens.fcsd)[lead]
    fsd[lead == 0] = 0.0
    ql = np.array(WG.QS_GRID)
    ds, daily = [], []
    phys = b["phys"][0].numpy()
    for p in range(NPATHS):
        sh = np.zeros((1, 14, 4), np.float32)
        sh[0, 7:, 0] = shift_hdd
        sh[0, :, :2] += rng.standard_normal((14, 2)) * fsd
        sh[0, :, 0] = np.maximum(sh[0, :, 0], -phys[:, 0])
        sh[0, :, 1] = np.maximum(sh[0, :, 1], -phys[:, 1])
        qs, dl = 0, 0
        with torch.no_grad():
            for m, cal in ens.members:
                o = m(b, sd, phys_shift=torch.tensor(sh))
                qm = o["q"][0, 0].numpy().astype(float)
                if cal is not None:
                    qm = qm[6] + np.array(cal[c["kind"]][0])
                qs = qs + qm
                dl = dl + o["daily"][0].sum(-1).numpy()
        qs, dl = np.sort(qs / len(ens.members)), dl / len(ens.members)
        u = rng.uniform(0.025, 0.975)
        ds.append(np.interp(u, ql, qs))
        daily.append(dl * (ds[-1] / dl.sum()) if abs(dl.sum()) > 1e-6 else dl)   # path rescaled to its weekly draw
    return np.array(ds), np.array(daily)


def main():
    w, o, f = S.load()
    name = "wm_gas"
    ens = WG.GasEnsemble(*WG.FROZEN[name])
    cand = np.where((w.week >= WINTER[0]) & (w.week <= WINTER[1]))[0]
    best = None
    for k in cand:
        assert w.avail.iloc[k] <= pd.Timestamp("2025-06-30"), "scenario week must be DEV"
        cc = S.build_context(int(k), "thu", w, o, f)
        dev = abs(ens.quantiles([cc])[0, 0, 6] - cc["avg5"]["l48"])
        if best is None or dev < best[0]:
            best = (dev, int(k), cc)
    _, k, c = best
    days = [pd.Timestamp(x) for x in c["wx_days"]]
    sdh = hdd_anom_sd(c, days)
    rng = np.random.default_rng(0)
    out = {"model": name, "ckpt": ens.meta["ckpt"], "ckpt_sha256": ens.meta["ckpt_sha256"],
           "target_week": c["target_week"], "origin": c["origin"], "paths": NPATHS,
           "days": c["wx_days"], "wx_src": c["wx_src"].tolist(), "hdd_known": np.round(c["wx"][:, 0], 1).tolist(),
           "hdd_normal": np.round(c["wx_normal"][:, 0], 1).tolist(), "hdd_anom_sd_daily": round(sdh, 2),
           "avg5_l48": round(c["avg5"]["l48"], 1), "last_level_l48": float(c["lvl_hist"]["l48"][-1]), "arms": {}}
    for arm, sh in (("baseline", 0.0), ("cold_snap_+2sd_7d", 2 * sdh)):
        ds, daily = simulate(ens, c, sh, rng)
        path = out["last_level_l48"] + np.cumsum(daily, 1)
        out["arms"][arm] = {
            "hdd_shift_per_day": round(sh, 2),
            "dS_quantiles_10_50_90": np.round(np.quantile(ds, [0.1, 0.5, 0.9]), 1).tolist(),
            "dev_vs_avg5_median": round(float(np.median(ds) - c["avg5"]["l48"]), 1),
            "P_draw_larger_than_5yr_avg": round(float(np.mean(ds < c["avg5"]["l48"])), 3),
            "storage_path_p10": np.round(np.quantile(path, 0.1, 0), 1).tolist(),
            "storage_path_p50": np.round(np.quantile(path, 0.5, 0), 1).tolist(),
            "storage_path_p90": np.round(np.quantile(path, 0.9, 0), 1).tolist()}
    out["limitations"] = [
        "Simulation only, not scored; a counterfactual through a model fitted on 2015-2025 DEV weeks, not a forecast.",
        "The shift is applied to the degree-day inputs only; supply, LNG exports and prices are not shocked, and no "
        "feedback from price to demand is modelled. The response is the model's learned (monotone) weather "
        "sensitivity, extrapolated if +2 sd lies outside the training range for that week.",
        "Daily flows are latent: EIA publishes only weekly totals, so the daily path is not validated against data; "
        "each path's daily flows are rescaled to sum to its weekly draw.",
        "Uncertainty = CPC forecast-error noise on forecast days + the model's quantile spread; parameter uncertainty "
        "is only represented by the 3 seed members. Quantiles from the same model were only checked on DEV/VAL.",
        "The direct twin also takes weather as input and can be shocked the same way; the WM's addition is the "
        "daily latent state path, not the ability to answer what-if questions per se."]
    json.dump(out, open(os.path.join(ROOT, "runs", "gas", "scenario_demo.json"), "w"), indent=1)
    print(json.dumps({a: {k2: v for k2, v in d.items() if not k2.startswith("storage_path")} for a, d in out["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
