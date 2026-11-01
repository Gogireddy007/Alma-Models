"""
Architect B (non-RSSM) Alma-Fin models, plug-in contract of tools/fin/README.md. docs/fin_search.md (B section).

  wm_b_fin     BNet world model (search_b/models_b.py; config in the checkpoint), ensemble of seeds, forecaster
  wm_b_fin_cl  wm_b_fin + frozen online meta-rule through observe(): affine logit recalibration (a, b) and a
               median shift, updated by plain SGD on the revealed day only (past bars only)
  wm_b_voltgt  agent: sign(mom_k) * min(1, c * sigma* / sigma_B), sigma_B from wm_b_fin's median (no RL)
  wm_b_resid   agent: risk-sensitive vol-target residual policy (search_b/policy_b.py) on top of wm_b_voltgt

Checkpoints in runs/fin/search_b/ckpt/; the factory asserts the sha256 recorded in B_CKPT (None = not frozen yet).
"""
import os
import sys
import hashlib
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "search_b"))
import feats_b as FB     # noqa: E402
import models_b as MB    # noqa: E402

# name -> {"ckpt", "sha", "type"}; the factory refuses a checkpoint whose sha256 differs.
# wm_b_moe4: c6_b2_moe4_s300 recipe (MLP 14-day window encoder + HAR skip + 4-expert MoE regime head + weekday
# one-hot, d=32, 300 AdamW steps, 3 seeds 0/1000/2000) REFIT on the 16 training coins, labels <= 2025-06-30 (no
# selection on the refit). Selected on DEV-early EM; F1 (post-selection, max of B's 3 F1 looks) 0.7152.
B_CKPT = {
    "wm_b_moe4": {"ckpt": "runs/fin/search_b/ckpt/wm_b_moe4_refit.pt",
                  "sha": "b560e92e45a1bb74c582d2b542e41cc10e2966c8a84d36dfe55b3df1f049e613", "type": "fc"},
}
B_ONLINE = {"lr": 0.02, "lr_q": 0.0}
B_VOLTGT = {"k": 4, "c": 1.0, "band": 0.0}
TGT = {"crypto": 0.40 / np.sqrt(365), "etf": 0.10 / np.sqrt(252)}


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_ckpt(rel, sh=None):
    p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    got = _sha(p)
    if sh is not None and got != sh:
        raise RuntimeError(f"checkpoint {rel} sha256 {got} != frozen {sh}")
    members, cfg, meta = MB.load(p)
    return members, cfg, dict(meta, ckpt=rel, ckpt_sha256=got)


class BForecaster:
    kind = "forecaster"

    def __init__(self, members, cfg, meta):
        self.members, self.cfg = members, cfg
        self.meta = dict(meta, n_members=len(members), params_per_member=MB.nparams(members[0]), cfg=cfg)
        torch.set_num_threads(3)

    def _raw(self, ctx):
        F = FB.features(ctx)
        t = lambda k: torch.from_numpy(np.ascontiguousarray(F[k]))[None]
        ok = torch.from_numpy(F["ok"])[None]
        ps, qs, lgs = [], [], []
        with torch.no_grad():
            for m in self.members:
                g = torch.Generator().manual_seed(1234)
                o = m(t("X"), t("hz"), t("har"), t("aux"), ok, gen=g)
                ps.append(torch.sigmoid(o["logit"][0]).numpy().astype(np.float64))
                qs.append(o["q"][0].numpy().astype(np.float64))
        p = np.mean(ps, 0)
        q = np.sort(np.mean(qs, 0), -1) + F["A"][:, None]
        return F, p, q

    def predict(self, ctx):
        F, p, q = self._raw(ctx)
        p[~F["ok"]] = np.nan
        q[~F["ok"]] = np.nan
        return {"p_up": p, "q": q, "p_ret_up": np.full(len(p), np.nan)}


class BContinual(BForecaster):
    """Online meta-rule (frozen before TEST): logit' = a*logit + b, a=1,b=0 at start; after day D is revealed, one SGD
    step on the BCE of that day's y_vol over the coins (label day D < next decision day D+1); the median-shift term
    mu (added to all quantiles) gets one SGD step on the pinball of the median, lr_q. Uses only predict() inputs and
    revealed rows."""

    def __init__(self, members, cfg, meta, rule=None):
        super().__init__(members, cfg, meta)
        self.rule = dict(B_ONLINE if rule is None else rule)
        self.meta["online_rule"] = self.rule
        self.a, self.b, self.mu = 1.0, 0.0, 0.0
        self.last = None

    def predict(self, ctx):
        F, p, q = self._raw(ctx)
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.nanmean(np.asarray(ctx["daily"]["logrv"], float)[-7:], 0)
        self.last = (ctx["day"], lg, q[:, 6].copy(), ref, F["ok"])
        p2 = 1 / (1 + np.exp(-(self.a * lg + self.b)))
        q2 = q + self.mu
        p2[~F["ok"]] = np.nan
        q2[~F["ok"]] = np.nan
        return {"p_up": p2, "q": q2, "p_ret_up": np.full(len(p2), np.nan)}

    def observe(self, rec):
        import pandas as pd
        day, lg, med, ref, ok = self.last
        assert pd.Timestamp(rec["day"]) == pd.Timestamp(day), "observe() for a day that was not just predicted"
        assert pd.Timestamp(rec["available_from"]) <= pd.Timestamp(day) + pd.Timedelta(days=1)
        lr = np.asarray(rec["row"]["logrv"], float)
        y = np.where(np.isfinite(lr) & np.isfinite(ref), (lr > ref).astype(float), np.nan)
        k = ok & np.isfinite(y) & np.isfinite(lg)
        if k.sum() == 0:
            return 0
        p = 1 / (1 + np.exp(-(self.a * lg[k] + self.b)))
        g = p - y[k]
        self.a -= self.rule["lr"] * float(np.mean(g * lg[k]))
        self.b -= self.rule["lr"] * float(np.mean(g))
        if self.rule.get("lr_q", 0) > 0:
            kk = k & np.isfinite(med) & np.isfinite(lr)
            if kk.any():
                self.mu -= self.rule["lr_q"] * float(np.mean(np.where(lr[kk] > med[kk] + self.mu, -0.5, 0.5)))
        return 1


class BVolTgt:
    kind = "agent"

    def __init__(self, fc, rule=None):
        self.fc = fc
        self.rule = dict(B_VOLTGT if rule is None else rule)
        self.meta = dict(fc.meta, agent_rule=self.rule)
        self.prev = None

    def target(self, ctx):
        F, p, q = self.fc._raw(ctx)
        tgt = TGT[ctx["market"]]
        sig = np.sqrt(np.exp(np.nan_to_num(q[:, 6], nan=np.log(tgt ** 2))))
        k = self.rule["k"]
        dirn = np.ones(len(sig)) if k < 0 else np.sign(F["mom"][:, k])
        base = dirn * np.minimum(1.0, self.rule["c"] * tgt / np.maximum(sig, 1e-6))
        return F, p, q, np.nan_to_num(base) * np.isfinite(np.asarray(ctx["daily"]["close"], float)[-1])

    def act(self, ctx):
        base = self.target(ctx)[3]
        band = self.rule.get("band", 0.0)
        if band > 0:      # no-trade band: replay the rule over the context's last 20 days from 0 (stateless)
            raise NotImplementedError("band rule not frozen")
        return base


def register(R):
    """B_CKPT[name] = {"ckpt": rel path, "sha": sha256 or None, "type": "fc" | "cl" | "voltgt", "rule": {...}}."""
    for name, e in B_CKPT.items():
        rel, sh, ty, rule = e["ckpt"], e.get("sha"), e.get("type", "fc"), e.get("rule")
        if ty == "cl":
            R[name] = (lambda rel=rel, sh=sh, rule=rule: BContinual(*load_ckpt(rel, sh), rule=rule))
        elif ty == "voltgt":
            R[name] = (lambda rel=rel, sh=sh, rule=rule: BVolTgt(BForecaster(*load_ckpt(rel, sh)), rule))
        else:
            R[name] = (lambda rel=rel, sh=sh: BForecaster(*load_ckpt(rel, sh)))
