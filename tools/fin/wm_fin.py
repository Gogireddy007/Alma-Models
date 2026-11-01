"""
Alma-Fin world-model forecasters (architect). docs/fin_search.md.

  wm_fin     RSSM, day heads on latents rolled through day D (claim model, rollout)
  wm_direct  matched twin: identical modules/data/steps/seeds, day heads on the filtered state at D-1
  wm_fin_cl  wm_fin + frozen online rule through observe() (continual); twin = "wm_fin_cl:frozen"
  wm_voltgt  agent (no RL): sign(30-day return) x min(1, sigma*/sigma_WM), sigma_WM = exp(median log RV / 2) of wm_fin

Each frozen model is a 3-seed ensemble in one checkpoint; its sha256 is asserted at load and recorded in meta.
"""
import os
import sys
import copy
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import wm_core as WC     # noqa: E402

CKDIR = os.path.join(ROOT, "runs", "fin", "search", "ckpt")
# filled at freeze (docs/fin_search.md section 3); the factory refuses a checkpoint whose sha256 differs
FROZEN_CKPT = {
    "wm_fin": ("runs/fin/search/ckpt/frozen_wm_fin.pt", "5b90558f4c271e4dc00f5ad8c5207b53d07c5676326b26c5c3d796dbd41b6b08"),
    "wm_direct": ("runs/fin/search/ckpt/frozen_wm_direct.pt", "38b39cf96bd921ccb8b08b87bdb2f00bf68e9d4875d29a8c2953c9ab33a708be"),
}
# frozen online rule (docs/fin_search.md section 3)
ONLINE = {"steps": 1, "lr": 0.0001, "l2_init": 1.0, "replay_days": 60, "params": "all", "meta": None}


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_frozen(name):
    rel, sh = FROZEN_CKPT[name]
    p = os.path.join(ROOT, rel)
    got = _sha(p)
    if sh is not None and got != sh:
        raise RuntimeError(f"{name}: checkpoint {rel} sha256 {got} != frozen {sh}")
    members, meta = WC.load_members(p)
    return members, dict(meta, ckpt=rel, ckpt_sha256=got)


class WMForecaster:
    kind = "forecaster"

    def __init__(self, members, meta):
        self.members = members
        self.meta = dict(meta, n_members=len(members), params_per_member=WC.nparams(members[0]))
        torch.set_num_threads(3)

    def _feats(self, ctx):
        m0 = self.members[0]
        X, aux, A, ok = WC.make_features(ctx["daily"], ctx.get("hourly"), ctx["market"], m0.w, m0.mf)
        return X, WC.with_dow(aux, ctx["day"], m0.dow), A, ok

    def predict(self, ctx):
        X, aux, A, ok = self._feats(ctx)
        q, p, pr = WC.forecast_members(self.members, X, aux, A)
        q[~ok], p[~ok], pr[~ok] = np.nan, np.nan, np.nan
        return {"p_up": p, "q": q, "p_ret_up": pr}


class WMContinual(WMForecaster):
    """Online rule: after day D's row is revealed (observe), take `steps` plain-SGD steps on the replay set = the last
    `replay_days` completed days (D-replay+1..D) with loss 2*pinball + 0.5*BCE(P1) + l2_init*||theta - theta_frozen||^2.
    The replay set is built only from contexts already passed to predict() and from revealed rows; every sample's
    label day is asserted < the next decision day."""

    def __init__(self, members, meta, rule=None):
        super().__init__([copy.deepcopy(m) for m in members], meta)
        self.rule = dict(ONLINE if rule is None else rule)
        self.meta["online_rule"] = self.rule
        self.anchor = [[p.detach().clone() for p in m.parameters()] for m in self.members]
        if self.rule.get("params", "all") == "heads":
            for m in self.members:
                for nm, p in m.named_parameters():
                    p.requires_grad_(nm.startswith("heads."))
        self.opt = [torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=self.rule["lr"])
                    for m in self.members]
        self.frozen = [copy.deepcopy(m).eval() for m in self.members]
        self.lrs = [float(self.rule["lr"])] * len(self.members)
        self.lr_trace = []
        self.buf = deque(maxlen=self.rule["replay_days"])
        self.last = None
        self.label_days = deque(maxlen=self.rule["replay_days"])

    def predict(self, ctx):
        day = ctx["day"]
        assert all(d < day for d in self.label_days), "replay holds a label dated >= the decision day"
        X, aux, A, ok = self._feats(ctx)
        lr = np.asarray(ctx["daily"]["logrv"], float)
        if not self.buf and lr.shape[0] > 100:          # backfill from the context (daily-only features, past labels)
            for k in range(self.rule["replay_days"] - 1, 0, -1):
                dk = {f: v[:lr.shape[0] - k] for f, v in ctx["daily"].items()}
                Xk, ak, Ak, okk = WC.make_features(dk, None, ctx["market"], self.members[0].w, self.members[0].mf)
                ak = WC.with_dow(ak, ctx["days"][lr.shape[0] - k], self.members[0].dow)
                lab = lr[lr.shape[0] - k]
                ref = np.nanmean(lr[lr.shape[0] - k - 7:lr.shape[0] - k], 0)
                self._push(Xk, ak, Ak, okk, lab, ref, ctx["days"][lr.shape[0] - k])
        self.last = (day, X, aux, A, ok, np.nanmean(lr[-7:], 0))
        q, p, pr = WC.forecast_members(self.members, X, aux, A)
        q[~ok], p[~ok], pr[~ok] = np.nan, np.nan, np.nan
        return {"p_up": p, "q": q, "p_ret_up": pr}

    def _push(self, X, aux, A, ok, logrv, ref, label_day):
        import pandas as pd
        tl = logrv - A
        y = np.where(np.isfinite(logrv) & np.isfinite(ref), (logrv > ref).astype(float), np.nan)
        keep = ok & np.isfinite(tl)
        if keep.sum() == 0:
            return False
        self.buf.append((X[keep], aux[keep], tl[keep].astype(np.float32), y[keep].astype(np.float32)))
        self.label_days.append(pd.Timestamp(label_day))
        return True

    @staticmethod
    def _loss(md, Xb, ab, tl, y):
        h, z, _, _ = md.filter(Xb, sample=False, losses=False)
        hins, _ = md.head_input(h, z, ab, sample=False, S=1,
                                shared_noise=WC.shared_noise(1) if md.abl == "none" else None)
        q, lp, _ = md.head_out(hins[0])
        l_q = WC.pinball(q, tl).mean()
        oky = torch.isfinite(y)
        l_p = torch.nn.functional.binary_cross_entropy_with_logits(lp[oky], y[oky]) if oky.any() else 0.0
        return 2.0 * l_q + 0.5 * l_p

    def observe(self, rec):
        import pandas as pd
        day, X, aux, A, ok, ref = self.last
        assert pd.Timestamp(rec["day"]) == pd.Timestamp(day), "observe() for a day that was not just predicted"
        assert pd.Timestamp(rec["available_from"]) <= pd.Timestamp(day) + pd.Timedelta(days=1)
        pushed = self._push(X, aux, A, ok, np.asarray(rec["row"]["logrv"], float), ref, day)
        if not self.buf:
            return 0
        meta = self.rule.get("meta")
        if meta and pushed:
            # self-improving rule: the newest day was never trained on; compare its loss under the online and the
            # frozen weights (a pure past prediction error) and adapt the learning rate multiplicatively
            nb = self.buf[-1]
            xb, abb, tlb, yb = (torch.from_numpy(v) for v in nb)
            with torch.no_grad():
                for k, (md, fr) in enumerate(zip(self.members, self.frozen)):
                    won = float(self._loss(md, xb, abb, tlb, yb)) < float(self._loss(fr, xb, abb, tlb, yb))
                    self.lrs[k] = float(np.clip(self.lrs[k] * np.exp(meta["eta"] if won else -meta["eta"]),
                                                meta.get("lr_min", 1e-5), meta.get("lr_max", 1e-3)))
                    for g in self.opt[k].param_groups:
                        g["lr"] = self.lrs[k]
            self.lr_trace.append(list(self.lrs))
        Xb = torch.from_numpy(np.concatenate([b[0] for b in self.buf]))
        ab = torch.from_numpy(np.concatenate([b[1] for b in self.buf]))
        tl = torch.from_numpy(np.concatenate([b[2] for b in self.buf]))
        y = torch.from_numpy(np.concatenate([b[3] for b in self.buf]))
        n = 0
        for md, opt, anc in zip(self.members, self.opt, self.anchor):
            md.train()
            for _ in range(self.rule["steps"]):
                loss = self._loss(md, Xb, ab, tl, y)
                reg = sum(((p - a) ** 2).sum() for p, a in zip(md.parameters(), anc))
                loss = loss + self.rule["l2_init"] * reg
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(md.parameters(), 1.0)
                opt.step()
                n += 1
            md.eval()
        return n


class WMVolTgt:
    kind = "agent"

    def __init__(self, fc, k=30):
        self.fc, self.k = fc, k
        self.meta = dict(fc.meta, rule=f"sign({k}-day return) x min(1, sigma*/sigma_WM)", vol_target=WC.VOL_TARGET)

    def act(self, ctx):
        q = self.fc.predict(ctx)["q"][:, WC.QS.index(0.5)]
        sig = np.sqrt(np.exp(q))
        tgt = WC.VOL_TARGET[ctx["market"]]
        pos = np.sign(np.nansum(ctx["daily"]["ret"][-self.k:], 0)) * np.minimum(1.0, tgt / np.maximum(sig, 1e-6))
        return np.nan_to_num(pos)


def register(R):
    R["wm_fin"] = lambda: WMForecaster(*load_frozen("wm_fin"))
    R["wm_direct"] = lambda: WMForecaster(*load_frozen("wm_direct"))
    R["wm_fin_cl"] = lambda: WMContinual(*load_frozen("wm_fin"))
    R["wm_voltgt"] = lambda: WMVolTgt(WMForecaster(*load_frozen("wm_fin")))
