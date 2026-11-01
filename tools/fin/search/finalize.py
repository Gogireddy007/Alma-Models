"""Finalist stages (docs/fin_search.md Am1): holdout F1 (once per finalist), 3-seed F1 ensembles (candidate + twin),
refit on all DEV (<= 2025-06-30). No selection happens at the seed or refit stage.

  nice -n 10 .venv/bin/python tools/fin/search/finalize.py holdout_wm
  nice -n 10 .venv/bin/python tools/fin/search/finalize.py seeds_wm --key <cfgkey>
  nice -n 10 .venv/bin/python tools/fin/search/finalize.py refit_wm --key <cfgkey>
  nice -n 10 .venv/bin/python tools/fin/search/finalize.py holdout_cl | seeds_cl ... (see functions)
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402
import data as DT        # noqa: E402
import loop as LP        # noqa: E402

F1_FIT = "2024-06-30"
DEV_FIT = "2025-06-30"


def loop_state(p):
    return json.load(open(os.path.join(C.OUT, f"loop_{p}.json")))


def top3(p):
    st = loop_state(p)
    best = {}
    for h in st["hist"]:
        if h["decision"] in ("START", "KEEP") and h.get("em") is not None:
            best[h["key"]] = h
    return sorted(best.values(), key=lambda h: -h["em"])[:3]


def dataset(fit_end):
    tr, _ = DT.universe()
    path = os.path.join(C.CACHE, f"ds_w{WC.W_MAX}_{pd.Timestamp(fit_end).date()}.npz")
    if os.path.exists(path):
        return C.subset(dict(np.load(path)), fit_end)
    P = C.train_panel(fit_end) if fit_end == DEV_FIT else C.cut_panel(H.Panel("crypto"), pd.Timestamp(fit_end), tr)
    return C.subset(C.build_dataset(P, fit_end, coins=tr, wmax=WC.W_MAX), fit_end)


def member(cfg, fit_end, seed, abl):
    k = f"fin_{LP.key(cfg)}_{abl}_s{seed}_{fit_end}"
    path = os.path.join(C.CK, k + ".pt")
    if os.path.exists(path):
        return WC.load_members(path)[0][0], path
    D = dataset(fit_end)
    assert max(D["day_str"].tolist()) <= fit_end
    m = LP.train_model(D, cfg, seed, abl)
    WC.save_members(path, [m], {"cfg": cfg, "fit_end": fit_end, "seed": seed, "abl": abl, "params": WC.nparams(m),
                                "n_train": int(len(D["X"])), "last_label_day": max(D["day_str"].tolist())})
    return m, path


def eval_f1(fc, name, panels):
    out = {}
    for mk, P in panels.items():
        res = H.run(P, fc, C.bt_days(P), online=hasattr(fc, "observe"))
        C.save_preds(f"{name}_{mk}", res)
        out[mk] = res
    return out


def panels():
    return {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}


def holdout_wm(a):
    PN = panels()
    res = []
    for h in top3("wm"):
        cfg = h["cfg"]
        m, path = member(cfg, F1_FIT, 0, "none")
        name = f"hold_wm_{LP.key(cfg)}"
        r = eval_f1(WF.WMForecaster([m], {}), name, PN)
        c, e = C.p1_p3(PN["crypto"], r["crypto"]), C.p1_p3(PN["etf"], r["etf"])
        C.log_row({"kind": "holdout_wm", "name": name, "config": cfg, "seed": 0, "fold": "F1", "status": "ok",
                   "P1_all": c["all"]["acc"], "P1_train": c["train"]["acc"], "P1_heldout": c["heldout"]["acc"],
                   "P3_all": c["all"]["qs"], "E1_etf": e["all"]["acc"], "E3_etf": e["all"]["qs"],
                   "note": json.dumps({"loop_em": h["em"], "key": LP.key(cfg)})})
        res.append((LP.key(cfg), h["em"], c["all"]["acc"], c["all"]["qs"]))
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(C.OUT, "holdout_wm.json"), "w"), indent=1)


def find_cfg(p, key):
    for h in loop_state(p)["hist"]:
        if h["key"] == key:
            return h["cfg"]
    raise KeyError(key)


def seeds_wm(a):
    cfg = find_cfg("wm", a.key)
    PN = panels()
    for abl in ("none", "direct"):
        ms = [member(cfg, F1_FIT, s, abl)[0] for s in range(3)]
        name = f"f1ens_wm_{a.key}_{abl}"
        r = eval_f1(WF.WMForecaster(ms, {}), name, PN)
        c, e = C.p1_p3(PN["crypto"], r["crypto"]), C.p1_p3(PN["etf"], r["etf"])
        C.log_row({"kind": "f1ens_wm", "name": name, "config": dict(cfg, abl=abl), "seed": "0,1,2", "fold": "F1",
                   "status": "ok", "P1_all": c["all"]["acc"], "P1_train": c["train"]["acc"],
                   "P1_heldout": c["heldout"]["acc"], "P3_all": c["all"]["qs"], "E1_etf": e["all"]["acc"],
                   "E3_etf": e["all"]["qs"], "note": f"params {WC.nparams(ms[0])}"})
        print(name, c["all"]["acc"], c["all"]["qs"], e["all"]["acc"], flush=True)


def refit_wm(a):
    cfg = find_cfg("wm", a.key)
    for abl in ("none", "direct"):
        ms, paths = [], []
        for s in range(3):
            m, p = member(cfg, DEV_FIT, s, abl)
            ms.append(m); paths.append(p)
        out = os.path.join(C.CK, f"frozen_{'wm_fin' if abl == 'none' else 'wm_direct'}.pt")
        WC.save_members(out, ms, {"recipe": cfg, "abl": abl, "seeds": [0, 1, 2], "fit_end": DEV_FIT,
                                  "members_sha256": [C.sha(p) for p in paths], "params_per_member": WC.nparams(ms[0]),
                                  "key": a.key})
        C.log_row({"kind": "refit_wm", "name": os.path.basename(out), "config": dict(cfg, abl=abl), "seed": "0,1,2",
                   "fold": "refit", "status": "trained", "note": f"sha {C.sha(out)}"})
        print(out, C.sha(out), flush=True)


def wm_final_key():
    return json.load(open(os.path.join(C.OUT, "wm_final.json")))["key"]


def cl_rule(cfg):
    return {"steps": cfg["steps"], "lr": cfg["lr"], "l2_init": cfg["l2_init"], "replay_days": cfg["replay_days"],
            "params": cfg["params"],
            "meta": None if cfg["meta"] == "none" else {"eta": cfg["eta"], "lr_min": 1e-5, "lr_max": 1e-3}}


def holdout_cl(a):
    wk = wm_final_key()
    wcfg = find_cfg("wm", wk)
    m, _ = member(wcfg, F1_FIT, 0, "none")
    P = C.dev_panel("crypto")
    days = C.bt_days(P)
    sf = C.p1_p3(P, H.run(P, WF.WMForecaster([m], {}), days, online=False))
    res = []
    for h in top3("cl"):
        cfg = h["cfg"]
        name = f"hold_cl_{LP.key(cfg)}"
        r = H.run(P, WF.WMContinual([m], {}, cl_rule(cfg)), days)
        C.save_preds(f"{name}_crypto", r)
        sc = C.p1_p3(P, r)
        d = sc["all"]["day_acc"] - sf["all"]["day_acc"]
        C.log_row({"kind": "holdout_cl", "name": name, "config": cfg, "seed": 0, "fold": "F1", "status": "ok",
                   "P1_all": sc["all"]["acc"] - sf["all"]["acc"], "P3_all": sc["all"]["qs"] - sf["all"]["qs"],
                   "note": json.dumps({"loop_em": h["em"], "P1_cont": sc["all"]["acc"], "P1_frozen": sf["all"]["acc"],
                                       "ci": C.MT.boot_mean_ci(d[np.isfinite(d)]), "wm": wk})})
        res.append((LP.key(cfg), h["em"], sc["all"]["acc"] - sf["all"]["acc"], sc["all"]["qs"] - sf["all"]["qs"]))
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(C.OUT, "holdout_cl.json"), "w"), indent=1)


def seeds_cl(a):
    cfg = find_cfg("cl", a.key)
    wk = wm_final_key()
    ms = [member(find_cfg("wm", wk), F1_FIT, s, "none")[0] for s in range(3)]
    PN = panels()
    for mk, P in PN.items():
        r = H.run(P, WF.WMContinual(ms, {}, cl_rule(cfg)), C.bt_days(P))
        C.save_preds(f"f1ens_cl_{a.key}_{mk}", r)
        sc = C.p1_p3(P, r)
        C.log_row({"kind": "f1ens_cl", "name": f"f1ens_cl_{a.key}", "config": cfg, "seed": "0,1,2", "fold": "F1",
                   "status": "ok", **({"P1_all": sc["all"]["acc"], "P3_all": sc["all"]["qs"],
                                       "P1_heldout": sc["heldout"]["acc"]} if mk == "crypto" else
                                      {"E1_etf": sc["all"]["acc"], "E3_etf": sc["all"]["qs"]}),
                   "note": f"updates {int(r['updates'].sum())}"})
        print(mk, sc["all"]["acc"], sc["all"]["qs"], flush=True)


RLAG = None


def _rl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fin_agents_rl", os.path.join(C.FIN, "agents", "_rl.py"))
    RL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RL)
    return RL


def train_pol(RL, algo, m, S, cfg, seed):
    kw = dict(hor=cfg["hor"], gamma=cfg["gamma"], cm=cfg["cm"], log=lambda s: None)
    if algo == "dreamer":
        return RL.train_dreamer(m, S, cfg["lam"], seed, cfg["steps"], lr=cfg["lr"], **kw)
    if algo == "ppo":
        return RL.train_ppo(m.d, S, cfg["lam"], seed, cfg["steps"], lr=cfg["lr"], zd=m.zd, **kw)
    return RL.train_dqn(m.d, S, cfg["lam"], seed, cfg["steps"], lr=cfg["lr"], zd=m.zd, **kw)


def holdout_rl(a):
    RL = _rl()
    wk = wm_final_key()
    m, _ = member(find_cfg("wm", wk), F1_FIT, 0, "none")
    S = RL.real_states(m, dataset(F1_FIT))
    PN = panels()
    rv = eval_f1(WF.WMVolTgt(WF.WMForecaster([m], {})), f"hold_voltgt_{wk}", PN)
    ref = C.p2(PN["crypto"], rv["crypto"])["heldout"]["sharpe"]
    C.log_row({"kind": "holdout_rl", "name": f"hold_voltgt_{wk}", "config": {"wm": wk}, "seed": 0, "fold": "F1",
               "status": "ok", "SR_heldout": ref, "SR_etf": C.p2(PN["etf"], rv["etf"])["all"]["sharpe"]})
    res = [("voltgt_ref", None, ref)]
    for h in top3("rl"):
        cfg = h["cfg"]
        pol = train_pol(RL, "dreamer", m, S, cfg, 0)
        name = f"hold_rl_{LP.key(cfg)}"
        r = eval_f1(RL.RLAgent("dreamer", [m], [pol], {}), name, PN)
        s = C.p2(PN["crypto"], r["crypto"])
        C.log_row({"kind": "holdout_rl", "name": name, "config": cfg, "seed": 0, "fold": "F1", "status": "ok",
                   "SR_heldout": s["heldout"]["sharpe"], "SR_all": s["all"]["sharpe"],
                   "SR_etf": C.p2(PN["etf"], r["etf"])["all"]["sharpe"], "turn_heldout": s["heldout"]["turnover"],
                   "mdd_heldout": s["heldout"]["max_drawdown"],
                   "note": json.dumps({"loop_em": h["em"], "sr_period_heldout": s["heldout"]["sr_period"]})})
        res.append((LP.key(cfg), h["em"], s["heldout"]["sharpe"]))
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(C.OUT, "holdout_rl.json"), "w"), indent=1)


def seeds_rl(a):
    """3-seed F1 ensembles of wm_agent, ppo, dqn (same cfg) and wm_voltgt on the 3-seed WM."""
    RL = _rl()
    cfg = find_cfg("rl", a.key) if a.key != "RL0" else dict(LP.RL0)
    wk = wm_final_key()
    ms = [member(find_cfg("wm", wk), F1_FIT, s, "none")[0] for s in range(3)]
    D = dataset(F1_FIT)
    Ss = [RL.real_states(m, D) for m in ms]
    PN = panels()
    agents = {"voltgt": WF.WMVolTgt(WF.WMForecaster(ms, {}))}
    for algo in ("dreamer", "ppo", "dqn"):
        pols = [train_pol(RL, algo, m, S, cfg, s) for s, (m, S) in enumerate(zip(ms, Ss))]
        agents[algo] = RL.RLAgent(algo, ms, pols, {})
    for algo, ag in agents.items():
        name = f"f1ens_{algo}_{a.key}"
        r = eval_f1(ag, name, PN)
        s, se = C.p2(PN["crypto"], r["crypto"]), C.p2(PN["etf"], r["etf"])
        C.log_row({"kind": "f1ens_rl", "name": name, "config": dict(cfg, algo=algo), "seed": "0,1,2", "fold": "F1",
                   "status": "ok", "SR_heldout": s["heldout"]["sharpe"], "SR_all": s["all"]["sharpe"],
                   "SR_etf": se["all"]["sharpe"], "turn_heldout": s["heldout"]["turnover"],
                   "mdd_heldout": s["heldout"]["max_drawdown"],
                   "note": json.dumps({"sr_period_heldout": s["heldout"]["sr_period"],
                                       "sr_x2_heldout": C.p2(PN["crypto"], r["crypto"], 2.0)["heldout"]["sharpe"],
                                       "mean_abs_pos": float(np.abs(r["crypto"]["pos"]).mean())})})
        print(name, s["heldout"]["sharpe"], s["all"]["sharpe"], se["all"]["sharpe"], flush=True)


def refit_rl(a):
    RL = _rl()
    cfg = find_cfg("rl", a.key) if a.key != "RL0" else dict(LP.RL0)
    ms, _ = WC.load_members(os.path.join(C.CK, "frozen_wm_fin.pt"))
    D = dataset(DEV_FIT)
    Ss = [RL.real_states(m, D) for m in ms]
    for algo, nm in (("dreamer", "wm_agent"), ("ppo", "ppo"), ("dqn", "dqn")):
        pols = [train_pol(RL, algo, m, S, cfg, s) for s, (m, S) in enumerate(zip(ms, Ss))]
        out = os.path.join(C.CK, f"frozen_{nm}.pt")
        RL.save_policies(out, pols, dict(cfg, algo=algo, seeds=[0, 1, 2], fit_end=DEV_FIT,
                                         wm_ckpt="runs/fin/search/ckpt/frozen_wm_fin.pt",
                                         wm_sha256=C.sha(os.path.join(C.CK, "frozen_wm_fin.pt"))))
        C.log_row({"kind": "refit_rl", "name": os.path.basename(out), "config": dict(cfg, algo=algo),
                   "seed": "0,1,2", "fold": "refit", "status": "trained", "note": f"sha {C.sha(out)}"})
        print(out, C.sha(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 3)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
