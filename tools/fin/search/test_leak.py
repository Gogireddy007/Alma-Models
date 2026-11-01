"""Leak tests for the architect's frozen models (docs/fin_search.md section 0). DEV days only (no VAL/TEST day is
used). Output: runs/fin/search/test_leak.txt.

  nice -n 10 .venv/bin/python tools/fin/search/test_leak.py [--models wm_fin,wm_direct,...]

1. future invariance: multiply every daily value of days >= D (incl. the fill of D) and every hourly value of hours
   >= D 00:00 by U(0.3, 3) -> the output at D is bit-identical;
2. fill-bar invariance: perturb only fill(D) (and the hourly bar of D 00:00) -> bit-identical;
3. sensitivity: perturbing day D-1's daily row, or the last hour before D (hour D-1 23:00), changes the output;
4. continual: wm_fin_cl walked over 12 days on the original panel and on a panel perturbed from day j on: decisions
   for days <= j are bit-identical, later ones differ; every replay label day < the decision day (asserted in the
   model) and observe() records are dated before the next decision.
"""
import os
import sys
import copy
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import harness as H      # noqa: E402

DAYS = ["2025-02-03", "2025-03-17", "2025-04-28", "2025-06-09", "2025-06-30"]
ALL = ["wm_fin", "wm_direct", "wm_fin_cl", "wm_agent", "ppo", "dqn", "wm_voltgt"]


def out_of(res):
    if "pos" in res:
        return res["pos"]
    return np.concatenate([res["p_up"], res["q"].reshape(len(res["q"]), -1), res["p_ret_up"]], 1)


def perturb(P, i, mode, rng):
    Q = copy.deepcopy(P)
    if mode == "future":
        for f in Q.daily:
            Q.daily[f][i:] = Q.daily[f][i:] * rng.uniform(0.3, 3, Q.daily[f][i:].shape)
        if Q.hourly is not None:
            for f in Q.hourly:
                Q.hourly[f][24 * i:] = Q.hourly[f][24 * i:] * rng.uniform(0.3, 3, Q.hourly[f][24 * i:].shape)
    elif mode == "fill":
        Q.daily["fill"][i] *= rng.uniform(0.3, 3, Q.daily["fill"][i].shape)
        if Q.hourly is not None:
            for f in ("open", "high", "low", "close"):
                Q.hourly[f][24 * i] *= rng.uniform(0.3, 3, Q.hourly[f][24 * i].shape)
    elif mode == "prev_day":
        for f in Q.daily:
            Q.daily[f][i - 1] = Q.daily[f][i - 1] * rng.uniform(0.3, 3, Q.daily[f][i - 1].shape)
    elif mode == "prev_hour":
        for f in Q.hourly:
            Q.hourly[f][24 * i - 1] = Q.hourly[f][24 * i - 1] * rng.uniform(0.3, 3, Q.hourly[f][24 * i - 1].shape)
    return Q


def family_factories(family, ckpt):
    """Factories for one architecture family built from a fold checkpoint (not the frozen ones)."""
    import types
    import importlib.util
    import wm_core as WC
    import wm_fin as WF
    members = WC.load_members(ckpt)[0]
    F = {}
    if family == "rssm":
        direct = []
        for m in members:
            d = WC.RSSM(abl="direct", **{k: v for k, v in m.cfg().items() if k != "abl"})
            d.load_state_dict(m.state_dict())
            d.eval()
            direct.append(d)
        F["fam_rssm_none"] = lambda: WF.WMForecaster(members, {})
        F["fam_rssm_direct"] = lambda: WF.WMForecaster(direct, {})
    elif family == "cl":
        F["fam_cl_sgd"] = lambda: WF.WMContinual(members, {}, {"steps": 1, "lr": 3e-4, "l2_init": 1.0,
                                                               "replay_days": 60, "params": "all"})
    elif family == "meta":
        F["fam_cl_meta"] = lambda: WF.WMContinual(members, {}, {"steps": 1, "lr": 3e-4, "l2_init": 1.0,
                                                                "replay_days": 20, "params": "heads",
                                                                "meta": {"eta": 0.2, "lr_min": 1e-5, "lr_max": 1e-3}})
    elif family == "agents":
        spec = importlib.util.spec_from_file_location("rl", os.path.join(C.FIN, "agents", "_rl.py"))
        RL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RL)
        D = C.subset(dict(np.load(os.path.join(C.CACHE, "ds_w28_2024-06-30.npz"))), "2023-12-31")
        S = RL.real_states(members[0], D)
        pols = {"dreamer": RL.train_dreamer(members[0], S, 0.5, 0, 30, log=lambda s: None),
                "ppo": RL.train_ppo(members[0].d, S, 0.5, 0, 40, zd=members[0].zd, log=lambda s: None),
                "dqn": RL.train_dqn(members[0].d, S, 0.5, 0, 30, zd=members[0].zd, log=lambda s: None)}
        for k, pol in pols.items():
            F[f"fam_ag_{k}"] = (lambda k=k, pol=pol: RL.RLAgent(k, members[:1], [pol], {}))
        F["fam_ag_voltgt"] = lambda: WF.WMVolTgt(WF.WMForecaster(members, {}))
    ns = types.SimpleNamespace(names=list(F), get=lambda n: F[n]())
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL))
    ap.add_argument("--family", default="", help="rssm|cl|meta|agents: test a family on a fold checkpoint "
                                                  "(--ckpt) before its loop scores count")
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    import models as M
    out_name = "test_leak.txt"
    if a.family:
        M = family_factories(a.family, a.ckpt)
        a.models = ",".join(M.names)
        out_name = f"test_leak_{a.family}.txt"
    rng = np.random.default_rng(0)
    panels = {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}
    lines, ok_all = [], True

    def rep(s, ok=True):
        nonlocal ok_all
        ok_all &= ok
        lines.append(("OK   " if ok else "FAIL ") + s)
        print(lines[-1], flush=True)

    for mk, P in panels.items():
        idx = [int(np.searchsorted(P.days, np.datetime64(d, "ns"))) for d in DAYS]
        idx = [min(i, len(P.days) - 1) for i in idx]
        for mn in a.models.split(","):
            base = mn
            for i in idx:
                o0 = out_of(H.run(P, M.get(base), [i], online=False))
                o0b = out_of(H.run(P, M.get(base), [i], online=False))
                rep(f"{mk} {mn} {str(P.days[i])[:10]} deterministic", np.array_equal(o0, o0b, equal_nan=True))
                for mode in ("future", "fill") + (("prev_day", "prev_hour") if mk == "crypto" else ("prev_day",)):
                    Q = perturb(P, i, mode, rng)
                    o1 = out_of(H.run(Q, M.get(base), [i], online=False))
                    same = np.array_equal(o0, o1, equal_nan=True)
                    if mode in ("future", "fill"):
                        rep(f"{mk} {mn} {str(P.days[i])[:10]} {mode}: unchanged", same)
                    else:
                        nd = int((~np.isclose(o0, o1, equal_nan=True)).sum())
                        rep(f"{mk} {mn} {str(P.days[i])[:10]} {mode}: changed ({nd} values)", not same)
                    del Q
        cl_name = next((n for n in a.models.split(",") if n.startswith("wm_fin_cl") or n.startswith("fam_cl")), None)
        if cl_name:
            i0 = idx[2]
            days = list(range(i0, i0 + 12))
            r0 = H.run(P, M.get(cl_name), days)
            j = i0 + 6
            Q = perturb(P, j, "future", rng)
            r1 = H.run(Q, M.get(cl_name), days)
            o0, o1 = out_of(r0), out_of(r1)
            k = days.index(j)
            rep(f"{mk} {cl_name} online: decisions for days <= {str(P.days[j])[:10]} bit-identical "
                f"(updates {r0['updates'].tolist()})", np.array_equal(o0[:k + 1], o1[:k + 1], equal_nan=True))
            rep(f"{mk} {cl_name} online: later decisions differ (perturbed rows observed)",
                not np.array_equal(o0[k + 1:], o1[k + 1:], equal_nan=True))
            rep(f"{mk} {cl_name} online: updates made every day", bool((r0["updates"] > 0).all()))
    lines.append("ALL OK" if ok_all else "FAILURES")
    open(os.path.join(C.OUT, out_name), "w").write("\n".join(lines) + "\n")
    print(lines[-1])
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
