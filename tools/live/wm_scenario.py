"""
Alma-Live Gas N-order implication layer: scenario simulation on a frozen struct WM (docs/live_search.md 0.4, A2).

An event enters as a shock to a node of the latent state at rollout day k0 (in units of that node's training-sd);
the transition (physics balance + learned shock graph + residual) propagates it over the 7-day rollout. Outputs:
node trajectories (shocked - base), and readout shifts (median and 10/90% quantiles of r1, r5, lrv5, dsurp in real
units). Producer equities: the 2nd/3rd-order equity implication is the UNG r5 shift times a rolling 250-day beta of
XLE on UNG (data/fin/etf_daily.parquet, rows <= t-1; scenario/risk use only, never timing).

    .venv/bin/python tools/live/wm_scenario.py <ckpt.pt> <YYYY-MM-DD> [scenario]
Scenarios: lng_outage (net_exports -2 sd, day 1), cold_snap (demand +2 sd, day 1), supply_cut (supply -2 sd),
storage_shock (storage -2 sd).
"""
import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wm_core as C  # noqa: E402

SCEN = {"lng_outage": {"net_exports": -2.0}, "cold_snap": {"demand": 2.0}, "supply_cut": {"supply": -2.0},
        "storage_shock": {"storage": -2.0}}


def node_sd(m, P, rows, norm, L):
    with torch.no_grad():
        Xw, U, ST, _ = C.windows(P, rows, L, norm)
        S = m.rollout(m.enc(Xw), U)
    return S[:, :, :C.NN_].reshape(-1, C.NN_).std(0)


def xle_beta(d):
    try:
        import pandas as pd
        e = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "fin", "etf_daily.parquet"))
        x = e[e.symbol == "XLE"].set_index("t")["adjclose"]
        u_ = None
        import wm_store as WS
        R = WS.load_raw()
        o, lpa = WS._ung_adj_full(R)
        u_ = pd.Series(np.exp(lpa), index=pd.to_datetime(o))
        df = pd.concat([np.log(x).diff(), np.log(u_).diff()], axis=1, keys=["x", "u"]).dropna()
        df = df[df.index < pd.Timestamp(d)].tail(250)
        return float(np.cov(df.x, df.u)[0, 1] / df.u.var())
    except Exception:  # noqa: BLE001
        return float("nan")


def simulate(m, P, norm, row, shock, L=20, k0=0, sd=None):
    Xw, U, ST, _ = C.windows(P, np.array([row]), L, norm)
    sh = [None] * C.WD.H
    vec = torch.zeros(1, C.NN_)
    for name, mag in shock.items():
        vec[0, C.NODES.index(name)] = float(mag * (sd[C.NODES.index(name)] if sd is not None else 1.0))
    sh[k0] = vec
    with torch.no_grad():
        qb, Sb = m(Xw, U, ST, sample=False, return_states=True)
        qs, Ss = m(Xw, U, ST, sample=False, shocks=sh, return_states=True)
    qb = qb.numpy()[0] * norm.ys[:, None] + norm.ym[:, None]
    qs = qs.numpy()[0] * norm.ys[:, None] + norm.ym[:, None]
    lv = [1, C.NQ // 2, C.NQ - 2]
    out = {"nodes_delta": {n: [round(float(v), 4) for v in (Ss - Sb)[0, :, i]] for i, n in enumerate(C.NODES)},
           "readout_delta": {t: {"q10": round(float(qs[j, lv[0]] - qb[j, lv[0]]), 5),
                                 "q50": round(float(qs[j, lv[1]] - qb[j, lv[1]]), 5),
                                 "q90": round(float(qs[j, lv[2]] - qb[j, lv[2]]), 5)} for j, t in enumerate(C.WD.TARGETS)}}
    return out


def main(ck_path, day, scen=None):
    ck = torch.load(ck_path, weights_only=False)
    m = C.WM(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    P = C.WD.load_panel()
    norm = C.Norm.__new__(C.Norm)
    for k, v in ck["norm"].items():
        setattr(norm, k, np.array(v))
    L = ck["cfg"].get("L", 20)
    D = P["D"].astype("datetime64[D]")
    row = int(np.searchsorted(D, np.datetime64(day)))
    sd = node_sd(m, P, np.arange(max(L, row - 250), row), norm, L).numpy()
    beta = xle_beta(day)
    res = {"day": str(D[row]), "ckpt": os.path.basename(ck_path), "xle_beta_ung": round(beta, 3), "scenarios": {}}
    for name in ([scen] if scen else SCEN):
        r = simulate(m, P, norm, row, SCEN[name], L, sd=sd)
        r["producer_eq_r5_delta"] = round(r["readout_delta"]["r5"]["q50"] * beta, 5) if np.isfinite(beta) else None
        res["scenarios"][name] = r
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1))
