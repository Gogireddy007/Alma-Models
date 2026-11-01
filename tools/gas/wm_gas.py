"""Harness wrappers for the architect's frozen gas storage world model and its matched direct twin
(docs/gas_search.md; runs/gas/FROZEN.json).

register(REGISTRY) adds every model in FROZEN whose checkpoint exists. A checkpoint holds the 3 seed members of one
model (same recipe, seeds 0,1,2), the training scales and the recipe; its sha256 is verified at load and exposed as
meta['ckpt'] / meta['ckpt_sha256'] for the harness preflight. Inference uses ONLY the harness context dicts (through
tools/gas/search/feats.py, the same feature builder used in training). Deterministic (no sampling); the Monte Carlo
rollouts live in tools/gas/wm_scenario.py (not scored).
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(HERE, "search")):
    if p not in sys.path:
        sys.path.insert(0, p)

QS_GRID = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
SERIES = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]
# name -> (checkpoint relative to ROOT, sha256). Filled at the freeze (docs/gas_search.md section 3).
# Recipe B (docs/gas_search.md section 3): WM assim=1 regime=1 d=32 wd=0.05 drop=0.3 cal=1, 1200 steps, seeds 0-2,
# refit on all storage.train_weeks() (released <= 2025-06-30).
FROZEN = {
    "wm_gas": ("runs/gas/search/ckpt/wm_gas.pt", "52e26f50f3576ffa928828b9c72b954b96ea9d2086cfc689c308b7a3a44d9e8a"),
    "wm_gas_direct": ("runs/gas/search/ckpt/wm_gas_direct.pt",
                      "7ebc0fbf1f79bc9ab5377283a3ceec17e66c8c6b62905668d74fa7fcc1a57457"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class GasEnsemble:
    def __init__(self, rel, sh):
        import torch
        import nets as N
        path = os.path.join(ROOT, rel)
        got = sha256(path)
        assert got == sh, f"sha256 mismatch for {rel}: {got}"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg, self.scales = ck["cfg"], ck["scales"]
        self.members = []
        for st in ck["members"]:
            m = N.GasNet(ck["g_dim"], ck["u_dim"], d=self.cfg["d"], rollout=(self.cfg["arch"] == "wm"),
                         assim=bool(self.cfg["assim"]), regime=bool(self.cfg["regime"]),
                         drop=self.cfg.get("drop", 0.0))
            m.load_state_dict(st["state"])
            m.eval()
            self.members.append((m, st.get("cal")))
        self.fcsd = ck.get("fcsd")
        self.meta = {"ckpt": rel, "ckpt_sha256": sh, "recipe": self.cfg, "seeds": [s["seed"] for s in ck["members"]],
                     "train_end": ck["train_end"], "n_train_weeks": ck["n_train_weeks"],
                     "n_params_per_member": ck["n_params"], "rollout": self.cfg["arch"] == "wm"}
        self._cache = {}

    def batch_tensors(self, ctxs):
        import torch
        import feats as FT
        fs = FT.stack([FT.feats(c, self.scales) for c in ctxs])
        b = {k: torch.tensor(v) for k, v in fs.items()}
        if self.cfg.get("drop_hh"):
            b["g"][:, -3:-1] = 0.0
        return b

    def quantiles(self, ctxs, phys_shift=None):
        """-> [n, 6, 13] Bcf on QS_GRID: quantile average over members, re-sorted."""
        import torch
        b = self.batch_tensors(ctxs)
        sd = torch.tensor(self.scales["dev_sd"], dtype=torch.float32)
        acc = 0
        with torch.no_grad():
            for m, cal in self.members:
                q = m(b, sd, phys_shift=phys_shift)["q"].numpy().astype(np.float64)
                if cal is not None:          # holdout-calibrated quantiles: median + residual quantiles (per kind)
                    med = q[:, :, 6]
                    q = np.stack([med[i][:, None] + np.array(cal[c["kind"]]) for i, c in enumerate(ctxs)])
                acc = acc + q
        return np.sort(acc / len(self.members), -1)

    def storage(self, ctxs, qs, series):
        key = tuple((c["target_week"], c["kind"]) for c in ctxs)
        if key not in self._cache:
            self._cache = {key: self.quantiles(ctxs)}
        q = self._cache[key][:, SERIES.index(series)]
        out = np.full((len(ctxs), len(qs)), np.nan)
        grid = [round(x, 4) for x in QS_GRID]
        for j, lv in enumerate(qs):
            if round(float(lv), 4) in grid:
                out[:, j] = q[:, grid.index(round(float(lv), 4))]
        return out


def _factory(name):
    def make():
        return GasEnsemble(*FROZEN[name])
    return make


def register(REGISTRY):
    for name, (rel, _) in FROZEN.items():
        if os.path.exists(os.path.join(ROOT, rel)):
            REGISTRY[name] = _factory(name)
