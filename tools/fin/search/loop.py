"""Autoresearch loop (docs/fin_search.md Am1). DEV-early folds only; the panel is cut at 2024-06-30 at load.

  nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar wm --budget 150 --until 02:00
  nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar cl --budget 80 --until 02:35
  nice -n 10 .venv/bin/python tools/fin/search/loop.py --pillar rl --budget 80 --until 03:05

Each proposal: one-knob mutation of the incumbent (the proposer reads the log: untried values only, knobs that were
kept before are preferred) -> screen on E3 -> if within tol of the incumbent's E3, also E1, E2 -> KEEP iff EM beats the
incumbent's EM by delta, else REVERT. Every proposal is a row in results.tsv (crashes included).
"""
import os
import sys
import json
import time
import copy
import hashlib
import argparse
import datetime
import subprocess
import traceback
import importlib.util

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402
import data as DT        # noqa: E402

_spec = importlib.util.spec_from_file_location("fin_agents_rl", os.path.join(C.FIN, "agents", "_rl.py"))
RL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RL)

LOOP_END = pd.Timestamp("2024-06-30")
FOLDS = [("E1", "2022-12-31", "2023-01-01", "2023-06-30"), ("E2", "2023-06-30", "2023-07-01", "2023-12-31"),
         ("E3", "2023-12-31", "2024-01-01", "2024-06-30")]

WM0 = {"d": 64, "zd": 16, "sub": 4, "w": 14, "mf": True, "lr": 1e-3, "steps": 500, "bs": 256, "etfp": 0.3,
       "w_q": 2.0, "w_p": 0.5, "w_blk": 0.5, "w_kl": 0.1, "w_rec": 0.5}
SPACE_WM = {"d": [32, 48, 64, 96, 128], "zd": [8, 16, 32], "sub": [1, 2, 4], "w": [7, 10, 14, 21, 28],
            "lr": [3e-4, 1e-3, 2e-3], "steps": [300, 500, 800], "bs": [128, 256], "etfp": [0.0, 0.3, 0.5],
            "mf": [False, True], "dow": [False, True], "w_q": [1.0, 2.0, 4.0], "w_p": [0.0, 0.5, 1.0, 2.0], "w_blk": [0.0, 0.5, 1.0],
            "w_kl": [0.03, 0.1, 0.3], "w_rec": [0.25, 0.5, 1.0]}
CL0 = {"steps": 1, "lr": 1e-4, "l2_init": 1.0, "replay_days": 60, "params": "all", "meta": "none", "eta": 0.2}
SPACE_CL = {"steps": [1, 2, 3], "lr": [3e-5, 1e-4, 3e-4, 1e-3], "l2_init": [0.1, 1.0, 10.0],
            "replay_days": [20, 60, 120], "params": ["all", "heads"], "meta": ["none", "lr_adapt"],
            "eta": [0.05, 0.2]}
RL0 = {"lam": 0.5, "hor": 10, "gamma": 0.9, "lr": 3e-4, "steps": 500, "cm": 1.0}
SPACE_RL = {"lam": [0.0, 0.25, 0.5, 1.0, 2.0], "hor": [5, 10, 20], "gamma": [0.8, 0.9, 0.97],
            "lr": [1e-4, 3e-4, 1e-3], "steps": [500, 1000], "cm": [1.0, 2.0]}
PILLAR = {"wm": (WM0, SPACE_WM, 0.003, 0.002), "cl": (CL0, SPACE_CL, 0.002, 0.001), "rl": (RL0, SPACE_RL, 0.3, 0.15)}


def key(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def wm_model_cfg(cfg):
    return {"d": cfg["d"], "zd": cfg["zd"], "sub": cfg["sub"], "w": cfg["w"], "mf": cfg["mf"],
            "dow": cfg.get("dow", False),
            "wts": {"q": cfg["w_q"], "p": cfg["w_p"], "blk": cfg["w_blk"], "bD": cfg["w_blk"], "kl": cfg["w_kl"],
                    "rec": cfg["w_rec"]}}


def train_model(D, cfg, seed, abl="none", log=None):
    """One RSSM member on dataset D (W_MAX=28 features; the model's last w days are used)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WC.RSSM(abl=abl, **wm_model_cfg(cfg))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    steps, bs = cfg["steps"], cfg["bs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    N = len(D["X"])
    w = cfg["w"]
    T = {k: torch.from_numpy(D[k]) for k in ("X", "aux", "tl", "y", "yr", "blk", "mblk")}
    T["aux"] = torch.from_numpy(WC.with_dow(D["aux"], D["day_str"], cfg.get("dow", False)))
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, N, bs))
        X = T["X"][idx][:, -w:].clone()
        if not cfg["mf"]:
            X[:, :, WC.F_MLV] = 0.0
            X[:, :, WC.F_MR] = 0.0
        etf = torch.from_numpy(rng.random(bs) < cfg["etfp"])
        X[etf, :, 7:15] = 0.0
        intra = torch.from_numpy(rng.random(bs) < 0.1)
        X[intra, :, 9:13] = 0.0
        X[intra, :, WC.F_MINTRA] = 0.0
        batch = (X, T["aux"][idx], T["tl"][idx], T["y"][idx], T["yr"][idx], T["blk"][idx], T["mblk"][idx])
        loss, parts = WC.loss_fn(model, batch, gen=g)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (s % 250 == 0 or s == steps - 1):
            log(f"  step {s} loss {float(loss):.4f}")
    model.eval()
    return model


def wait_resources(log):
    """<= 2 training jobs machine-wide (mine + family B's); wait if swap is nearly exhausted."""
    for _ in range(120):
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        free = float(sw.split("free =")[1].split("M")[0])
        ps = subprocess.run(["ps", "-axo", "pid,rss,command"], capture_output=True, text=True).stdout.splitlines()[1:]
        others = []
        for l in ps:
            f = l.split()
            if len(f) < 3 or "python" not in f[2] or int(f[0]) == os.getpid() or int(f[1]) < 150_000:
                continue
            cmd = " ".join(f[2:])
            if "queue" in cmd or "tools/fin/search/" in cmd or "Claude" in cmd:
                continue                       # queue drivers / my own tools / app helpers are not training jobs
            others.append(cmd[:80])
        if free > 400 and len(others) <= 1:
            return
        log(f"  waiting: swap free {free:.0f} MB, other jobs {len(others)}: {others}")
        time.sleep(30)


class Loop:
    def __init__(self, pillar, log):
        self.pillar, self.log = pillar, log
        P = H.Panel("crypto")
        self.P = C.cut_panel(P, LOOP_END)
        tr, _ = DT.universe()
        self.D = C.build_dataset(self.P, LOOP_END, coins=tr, wmax=WC.W_MAX)
        self.state_path = os.path.join(C.OUT, f"loop_{pillar}.json")
        self.models = {}
        self.utc = False

    def fold_days(self, a, b):
        return C.bt_days(self.P, pd.Timestamp(a), pd.Timestamp(b))

    # ---------------------------------------------------------------- WM fold models (cached on disk)
    def fold_model(self, cfg, fold, seed=0, abl="none"):
        name, fit_end, a, b = fold
        k = f"loop_{key(cfg)}_{abl}_s{seed}_{name}"
        if k in self.models:
            return self.models[k]
        path = os.path.join(C.CK, k + ".pt")
        if os.path.exists(path):
            m = WC.load_members(path)[0][0]
        else:
            m = train_model(C.subset(self.D, fit_end), cfg, seed, abl)
            WC.save_members(path, [m], {"cfg": cfg, "fit_end": fit_end, "seed": seed, "abl": abl, "fold": name})
        self.models[k] = m
        return m

    def score(self, cfg, fold):
        name, fit_end, a, b = fold
        days = self.fold_days(a, b)
        if self.pillar == "wm":
            m = self.fold_model(cfg, fold)
            s = C.p1_p3(self.P, H.run(self.P, WF.WMForecaster([m], {}), days, online=False))
            return s["all"]["acc"], {"P1_train": s["train"]["acc"], "P1_heldout": s["heldout"]["acc"],
                                     "P3": s["all"]["qs"]}
        wm = self.wm_cfg
        m = self.fold_model(wm, fold)
        if self.pillar == "cl":
            fk = ("frozen", name)
            if fk not in self.models:
                self.models[fk] = C.p1_p3(self.P, H.run(self.P, WF.WMForecaster([m], {}), days, online=False))
            sf = self.models[fk]
            rule = {"steps": cfg["steps"], "lr": cfg["lr"], "l2_init": cfg["l2_init"],
                    "replay_days": cfg["replay_days"], "params": cfg["params"],
                    "meta": None if cfg["meta"] == "none" else {"eta": cfg["eta"], "lr_min": 1e-5, "lr_max": 1e-3}}
            r = H.run(self.P, WF.WMContinual([m], {}, rule), days)
            sc = C.p1_p3(self.P, r)
            return sc["all"]["acc"] - sf["all"]["acc"], {"P1_cont": sc["all"]["acc"], "P1_frozen": sf["all"]["acc"],
                                                         "dQS": sc["all"]["qs"] - sf["all"]["qs"],
                                                         "updates": int(r["updates"].sum())}
        # rl: Dreamer-style agent on the fold WM (seed 0)
        sk = ("states", name)
        if sk not in self.models:
            self.models[sk] = RL.real_states(m, C.subset(self.D, fit_end))
        S = self.models[sk]
        pol = RL.train_dreamer(m, S, cfg["lam"], 0, cfg["steps"], lr=cfg["lr"], hor=cfg["hor"], gamma=cfg["gamma"],
                               cm=cfg["cm"], log=lambda s: None)
        ag = RL.RLAgent("dreamer", [m], [pol], {})
        r = H.run(self.P, ag, days, online=False)
        s = C.p2(self.P, r)
        return s["heldout"]["sharpe"], {"SR_all": s["all"]["sharpe"], "turn": s["heldout"]["turnover"],
                                        "sr_period_heldout": s["heldout"]["sr_period"],
                                        "mean_abs_pos": float(np.abs(r["pos"]).mean())}

    # ---------------------------------------------------------------- proposer
    def propose(self, inc, hist, space, rng):
        tried = {h["key"] for h in hist}
        kept = {}
        for h in hist:
            if h.get("decision") == "KEEP" and h.get("knob"):
                kept[h["knob"]] = kept.get(h["knob"], 0) + 1
        knobs = list(space)
        for _ in range(500):
            wts = np.array([1.0 + 2.0 * kept.get(k, 0) for k in knobs])
            kb = knobs[rng.choice(len(knobs), p=wts / wts.sum())]
            vals = space[kb]
            iv = inc.get(kb, vals[0])                    # knobs added later (dow) default to their first value
            cur = vals.index(iv) if iv in vals else 0
            near = [v for i, v in enumerate(vals) if abs(i - cur) == 1]
            far = [v for v in vals if v != iv and v not in near]
            for cand_vals in (near, far):
                cand_vals = list(cand_vals)
                rng.shuffle(cand_vals)
                for v in cand_vals:
                    c = dict(inc, **{kb: v})
                    if key(c) not in tried:
                        return c, kb
        return None, None

    def run(self, budget, until, seed=0, force=None):
        base, space, tol, delta = PILLAR[self.pillar]
        if self.pillar in ("cl", "rl"):
            self.wm_cfg = json.load(open(os.path.join(C.OUT, "loop_wm.json")))["incumbent"]
        st = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"hist": []}
        rng = np.random.default_rng(seed + len(st["hist"]))
        inc = st.get("incumbent")
        if inc is None:
            inc = dict(base)
            sc = {f[0]: self.score(inc, f) for f in FOLDS}
            st.update(incumbent=inc, inc_scores={k: v[0] for k, v in sc.items()},
                      inc_em=float(np.mean([v[0] for v in sc.values()])))
            st["hist"].append({"key": key(inc), "cfg": inc, "knob": None, "decision": "START", "em": st["inc_em"]})
            self.row(inc, "START", "EM", st["inc_em"], {k: v for k, v in sc.items()})
            json.dump(st, open(self.state_path, "w"), indent=1)
        n, streak = 0, 0
        now = lambda: (datetime.datetime.utcnow().strftime("%H:%M") if self.utc else
                       datetime.datetime.now().strftime("%H:%M"))
        while n < budget and now() < until and streak < 25:
            if force and n == 0:
                kb, v = force
                cand, knob = dict(inc, **{kb: v}), kb
                if key(cand) in {h["key"] for h in st["hist"]}:
                    cand, knob = self.propose(inc, st["hist"], space, rng)
            else:
                cand, knob = self.propose(inc, st["hist"], space, rng)
            if cand is None:
                self.log("search space exhausted")
                break
            wait_resources(self.log)
            n += 1
            t0 = time.time()
            try:
                e3, x3 = self.score(cand, FOLDS[2])
                if e3 < st["inc_scores"]["E3"] - tol:
                    dec, em, fold, extra = "REVERT(screen)", None, "E3", {"E3": [e3, x3]}
                else:
                    s1, s2 = self.score(cand, FOLDS[0]), self.score(cand, FOLDS[1])
                    em = float(np.mean([s1[0], s2[0], e3]))
                    extra = {"E1": list(s1), "E2": list(s2), "E3": [e3, x3]}
                    fold = "EM"
                    dec = "KEEP" if em > st["inc_em"] + delta else "REVERT"
                status = "ok"
            except Exception as ex:
                traceback.print_exc()
                dec, em, fold, extra, e3, status = "CRASH", None, "E3", {"err": f"{type(ex).__name__}: {ex}"}, None, "crash"
            h = {"key": key(cand), "cfg": cand, "knob": knob, "decision": dec, "e3": e3, "em": em,
                 "sec": round(time.time() - t0)}
            st["hist"].append(h)
            if dec == "KEEP":
                inc = cand
                st.update(incumbent=inc, inc_em=em, inc_scores={"E1": extra["E1"][0], "E2": extra["E2"][0], "E3": e3})
                streak = 0
            else:
                streak += 1
            self.row(cand, dec, fold, em if em is not None else e3, extra, status, knob)
            json.dump(st, open(self.state_path, "w"), indent=1, default=float)
            self.log(f"[{self.pillar} {n}] {knob}={cand.get(knob)} E3 {e3} EM {em} -> {dec} "
                     f"(inc EM {st['inc_em']:.4f}) {h['sec']}s")
        self.log(f"loop {self.pillar} stopped: n {n}, streak {streak}, time {datetime.datetime.now():%H:%M}")

    def row(self, cfg, dec, fold, val, extra, status="ok", knob=None):
        r = {"kind": f"loop_{self.pillar}", "name": f"{self.pillar}_{key(cfg)}", "config": cfg, "seed": 0,
             "fold": fold, "status": status, "note": json.dumps({"decision": dec, "knob": knob, "folds": extra},
                                                                  default=float)}
        if self.pillar == "wm":
            r["P1_all"] = val
        elif self.pillar == "cl":
            r["P1_all"] = val          # S_C (difference) for continual rows
        else:
            r["SR_heldout"] = val
        C.log_row(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar", required=True, choices=["wm", "cl", "rl"])
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--until", default="02:00")
    ap.add_argument("--utc", action="store_true", help="--until is a UTC HH:MM (RT-14)")
    ap.add_argument("--force", default="", help="knob=value of the first proposal (e.g. dow=True)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 3)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    lg = open(os.path.join(C.OUT, "logs", f"loop_{a.pillar}.log"), "a")

    def log(s):
        print(s, flush=True)
        lg.write(f"{datetime.datetime.now():%H:%M:%S} {s}\n")
        lg.flush()
    L = Loop(a.pillar, log)
    L.utc = a.utc
    force = None
    if a.force:
        k, v = a.force.split("=")
        force = (k, {"True": True, "False": False}.get(v, v))
    L.run(a.budget, a.until, force=force)


if __name__ == "__main__":
    main()
