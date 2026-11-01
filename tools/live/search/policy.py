"""
Alma-Live Gas policy evaluation on DEV folds (docs/live_search.md A2). UNG close(d) -> close(d+1), 10 bp per side,
reported at 1x / 2x / 3x. Every policy row is appended to results.tsv (kind=agent or kind=rule) and counts in K.

    nice -n 10 .venv/bin/python tools/live/search/policy.py rules
    nice -n 10 .venv/bin/python tools/live/search/policy.py wm <wm_run_id> [policies comma list]
"""
import os
import sys
import glob
import json
import time
import datetime as dt

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as RN  # noqa: E402
C, H = RN.C, RN.H
WD = RN.WD
import wm_agent as A  # noqa: E402

DAILY = os.path.join(RN.OUT, "daily")


def fold_data(P, L=20):
    return RN.folds(P, L)


def pnl_stats(w, r):
    out = {}
    for m in (1, 2, 3):
        p = A.pnl(w, r, A.COST * m)
        out[f"sh{m}x"] = round(A.sharpe(p), 3)
    p = A.pnl(w, r)
    eq = np.cumsum(p)
    out["mdd"] = round(float((np.maximum.accumulate(eq) - eq).max()), 3)
    out["turn"] = round(float(np.abs(np.diff(np.concatenate([[0], w]))).mean() * 252), 1)
    out["ann"] = round(float(p.mean() * 252), 4)
    return out


def log_policy(name, kind, W, Rr, D, note="", cfg=None, seed=0):
    w = np.concatenate([W[Y] for Y in sorted(W)]); r = np.concatenate([Rr[Y] for Y in sorted(W)])
    st = pnl_stats(w, r)
    os.makedirs(DAILY, exist_ok=True)
    rid = f"{kind}_{name}_{dt.datetime.utcnow().strftime('%H%M%S')}"
    np.savez(os.path.join(DAILY, f"{rid}.npz"), w=w, r=r, D=np.concatenate([D[Y] for Y in sorted(W)]))
    per = {Y: round(A.sharpe(A.pnl(W[Y], Rr[Y])), 2) for Y in sorted(W)}
    RN.log_row(run_id=rid, utc=dt.datetime.utcnow().isoformat(timespec="seconds"), kind=kind, family=name, seed=seed,
               S_A=st["sh1x"], status="ok", cfg=json.dumps(cfg or {}, sort_keys=True),
               note=f"{note} {json.dumps(st)} folds={json.dumps(per)}")
    print(f"{rid:40s} S_A {st['sh1x']:+.3f}  2x {st['sh2x']:+.3f}  3x {st['sh3x']:+.3f}  mdd {st['mdd']}  turn {st['turn']}  folds {per}", flush=True)
    return rid, st


def rule_positions(P, rows, tr):
    X = P["X"]; F = WD.FEATS
    f = lambda n: X[rows, F.index(n)]  # noqa: E731
    sig = np.exp(f("u_lrv20"))
    tv = np.minimum(A.TV / sig, A.CAP)
    out = {}
    out["buy_hold_ung"] = np.ones(len(rows))
    out["cash"] = np.zeros(len(rows))
    out["short_ung_carry"] = -tv
    dev = X[:, F.index("dev")]
    sd = dev[tr].std()
    mon = np.array([int(str(d)[5:7]) for d in P["D"][rows]])
    out["short_ung_carry_wd"] = np.where(np.isin(mon, [12, 1, 2]) & (f("dev") < -sd), 0.0, -tv)
    out["tsmom_20"] = np.sign(f("u_r20")) * tv
    # 12-1 month momentum from the price window (lp_m250 is a level proxy; use lp - lp_250 ex last month)
    out["tsmom_252_21"] = np.sign(f("u_lp_m250") - f("u_r20") * 0.5) * tv
    out["eia_day_long"] = f("eia_tomorrow") * tv
    out["mr_1"] = -np.sign(f("u_r1")) * tv
    out["mr_5"] = -np.sign(f("u_r5")) * tv
    dows = np.argmax(X[:, [F.index(f"dow{i}") for i in range(5)]], 1)
    yr = P["Y"][:, 0]
    wk = np.array([np.sign(np.nanmean(yr[tr][dows[tr] == i])) for i in range(5)])
    out["weekday"] = wk[dows[rows]] * tv
    out["seasonal_rule"] = np.where(np.isin(mon, [9, 10, 11, 12]), 1.0, np.where(np.isin(mon, [1, 2, 3, 4]), -1.0, 0.0)) * tv
    return out


def rules():
    P = WD.load_panel()
    W, R, D = {}, {}, {}
    for Y, (tr, te) in fold_data(P).items():
        pos = rule_positions(P, te, tr)
        for k, v in pos.items():
            W.setdefault(k, {})[Y] = v
        R[Y] = P["Y"][te, 0]; D[Y] = P["D"][te]
    res = {}
    for k in W:
        res[k] = log_policy(k, "rule", W[k], R, D, note="rule baseline")[1]
    # feature-matched direct models -> same Kelly sizing
    refs = RN.load_refs()
    for name in ("linqr", "lgbm"):
        Wk = {}
        for Y, (tr, te) in fold_data(P).items():
            norm = H.Norm(P["X"], P["Y"], tr)
            q = refs[Y][name][:, 0] * norm.ys[0] + norm.ym[0]
            mu = q[:, C.NQ // 2]; sig = np.maximum((q[:, -1] - q[:, 0]) / 3.29, 1e-3)
            Wk[Y] = A.kelly(mu, sig)
        res[name + "_kelly"] = log_policy(name + "_kelly", "rule", Wk, R, D, note="feature-matched ref + Kelly sizing")[1]
    return res


def load_fold_models(rid, Y):
    out = []
    for p in sorted(glob.glob(os.path.join(RN.CK, f"{rid}_f{Y}_s*.pt"))):
        ck = torch.load(p, weights_only=False)
        m = C.WM(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        out.append((m, ck))
    return out


def wm_policies(rid, which=("kelly", "voltgt", "dreamer", "ppo_imag", "ppo_raw", "dqn_raw"), steps=1500, seed=0):
    P = WD.load_panel()
    W, R, D = {k: {} for k in which}, {}, {}
    for Y, (tr, te) in fold_data(P).items():
        ms = load_fold_models(rid, Y)
        norm = H.Norm(P["X"], P["Y"], tr)
        L = ms[0][1]["cfg"].get("L", 20)
        obs_tr = [A.wm_obs(m, P, tr, norm, L) for m, _ in ms]
        obs_te = [A.wm_obs(m, P, te, norm, L) for m, _ in ms]
        avg = lambda obs, i: np.mean([o[i] for o in obs], 0)  # noqa: E731
        Ztr, Zte = obs_tr[0][0], obs_te[0][0]
        mu_tr, sig_tr = avg(obs_tr, 3), avg(obs_tr, 4)
        mu_te, sig_te = avg(obs_te, 3), avg(obs_te, 4)
        dis_tr = np.std([o[3] for o in obs_tr], 0) if len(ms) > 1 else np.zeros(len(tr))
        dis_te = np.std([o[3] for o in obs_te], 0) if len(ms) > 1 else np.zeros(len(te))
        rtr = P["Y"][tr, 0]
        R[Y] = P["Y"][te, 0]; D[Y] = P["D"][te]
        Ftr, Fte = RN.flat(P, tr), RN.flat(P, te)
        fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-6
        Ftr, Fte = (Ftr - fm) / fs, (Fte - fm) / fs
        for k in which:
            if k == "kelly":
                W[k][Y] = A.kelly(mu_te, sig_te)
            elif k == "voltgt":
                W[k][Y] = A.voltgt(mu_te, sig_te)
            elif k == "dreamer":
                pol = A.train_wm_agent(Ztr, avg(obs_tr, 2), sig_tr, steps=steps, seed=seed)
                W[k][Y] = A.run_policy(pol, Zte, sig_te)
            elif k == "ppo_imag":
                pol = A.train_ppo(Ztr, None, sig_tr, steps=steps, seed=seed, imag_q=avg(obs_tr, 1), disagree=dis_tr)
                W[k][Y] = A.run_policy(pol, Zte, sig_te)
            elif k == "ppo_wmobs":
                pol = A.train_ppo(Ztr, rtr, sig_tr, steps=steps, seed=seed)
                W[k][Y] = A.run_policy(pol, Zte, sig_te)
            elif k == "dqn_wmobs":
                pol = A.train_dqn(Ztr, rtr, sig_tr, steps=steps, seed=seed)
                W[k][Y] = A.run_policy(pol, Zte, sig_te)
            elif k == "ppo_raw":
                s_tr = np.exp(P["X"][tr, WD.FEATS.index("u_lrv20")]); s_te = np.exp(P["X"][te, WD.FEATS.index("u_lrv20")])
                pol = A.train_ppo(Ftr, rtr, s_tr, steps=steps, seed=seed)
                W[k][Y] = A.run_policy(pol, Fte, s_te)
            elif k == "dqn_raw":
                s_tr = np.exp(P["X"][tr, WD.FEATS.index("u_lrv20")]); s_te = np.exp(P["X"][te, WD.FEATS.index("u_lrv20")])
                pol = A.train_dqn(Ftr, rtr, s_tr, steps=steps, seed=seed)
                W[k][Y] = A.run_policy(pol, Fte, s_te)
    res = {}
    for k in which:
        kind = "rule" if k in ("kelly", "voltgt") else "agent"
        res[k] = log_policy(k, kind, W[k], R, D, note=f"on WM {rid}", cfg={"wm": rid, "steps": steps}, seed=seed)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "rules":
        rules()
    elif sys.argv[1] == "wm":
        which = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("kelly", "voltgt", "dreamer", "ppo_imag", "ppo_raw", "dqn_raw")
        wm_policies(sys.argv[2], which, steps=int(os.environ.get("PSTEPS", 1500)))
