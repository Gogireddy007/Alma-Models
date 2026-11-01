"""wm_gas_ens: 14-member storage ensemble + OLS blend + adaptive intervals (audit 2026-11-01; recipe frozen on DEV,
see winning_models/gas_wm_gas_ens/MODEL_CARD.md).

  q_nn  = member-weighted quantile average of wm_gas (seeds 0-2), wm_gas_direct (0-2) and refits of the same two
          recipes with seeds 3-6 (all trained on weeks released <= 2025-06-30)
  q     = sort(0.8 q_nn + 0.2 q_st_ols_wx)
  L48 only, intervals: q' = median + s (q - median), s = clip((mean |median error| of the last 8 released weeks /
          ref_kind) ** 0.5, 0.4, 2.0), ref thu 15.39 / fri 15.27 Bcf (DEV out-of-fold); s = 1 with < 4 past weeks.
The past weeks' errors need this model's own forecasts of weeks k-1..k-8 at their origins; they are rebuilt with
storage.build_context (each asserted information-set safe) and every one of those weeks is asserted to be released
before the current origin, so s uses nothing unknown at the origin. Regions: blended quantiles, not recalibrated.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_W = "winning_models/"
MEMBERS = [(_W + "gas_wm_gas/checkpoints/wm_gas.pt", "52e26f50f3576ffa928828b9c72b954b96ea9d2086cfc689c308b7a3a44d9e8a"),
           (_W + "gas_wm_gas/checkpoints/wm_gas_direct.pt", "7ebc0fbf1f79bc9ab5377283a3ceec17e66c8c6b62905668d74fa7fcc1a57457"),
           (_W + "gas_wm_gas_ens/checkpoints/x_wm_s3456.pt", "1958ad360f7480c4a7d94e9aca5bf098d4153e87ae2ec1b1a73d494af2bbf1eb"),
           (_W + "gas_wm_gas_ens/checkpoints/x_dir_s3456.pt", "d6e47f2aa5a32e7f354b6cc77f6378297c8fbc3cb48b46e56dbb0f3a9adeb3bb")]
W_OLS, WIN, EXP, MIN_PAST, CLIP = 0.2, 8, 0.5, 4, (0.4, 2.0)
REF = {"thu": 15.39, "fri": 15.27}
SERIES = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]


class GasEns:
    def __init__(self, registry):
        import wm_gas as WG
        self.WG = WG
        self.ens = [WG.GasEnsemble(rel, sh) for rel, sh in MEMBERS]          # sha256 verified inside
        self.n = [len(e.members) for e in self.ens]
        self.ols = registry["st_ols_wx"]()
        self.meta = {"members": [{"ckpt": r, "ckpt_sha256": s} for r, s in MEMBERS], "n_members": sum(self.n),
                     "w_ols": W_OLS, "interval_rule": f"adaptive, window {WIN}, exponent {EXP}, ref {REF}"}
        self._raw = {}
        self._cache = {}

    def _blend(self, ctxs):
        """[n, 6, 13] blended quantiles (not recalibrated), cached per (target_week, kind)."""
        todo = [c for c in ctxs if (c["target_week"], c["kind"]) not in self._raw]
        if todo:
            nn = sum(e.quantiles(todo) * k for e, k in zip(self.ens, self.n)) / sum(self.n)
            ol = np.stack([np.asarray(self.ols.storage(todo, self.WG.QS_GRID, s), float) for s in SERIES], 1)
            q = np.sort((1 - W_OLS) * nn + W_OLS * ol, -1)
            for c, qi in zip(todo, q):
                self._raw[(c["target_week"], c["kind"])] = qi
        return np.stack([self._raw[(c["target_week"], c["kind"])] for c in ctxs])

    def _scale(self, c):
        import storage as S
        k, kind = int(c["k"]), c["kind"]
        odt = pd.Timestamp(c["origin"])
        past = [j for j in range(k - WIN, k) if j >= 1]
        w, o, f = S.load()
        pctx = []
        for j in past:
            assert S.release_dt(w, j) <= odt, "past week not yet released at the origin"
            pctx.append(S.build_context(j, kind, w, o, f))
        if len(pctx) < MIN_PAST:
            return 1.0
        med = self._blend(pctx)[:, 0, 6]
        released = np.asarray(c["ds_hist"]["l48"], float)[-len(past):]       # actuals of weeks k-8..k-1 (known)
        err = np.abs(med - released)
        err = err[np.isfinite(err)]
        if len(err) < MIN_PAST:
            return 1.0
        return float(np.clip((err.mean() / REF[kind]) ** EXP, *CLIP))

    def quantiles(self, ctxs):
        key = tuple((c["target_week"], c["kind"]) for c in ctxs)
        if key not in self._cache:
            q = self._blend(ctxs).copy()
            for i, c in enumerate(ctxs):
                s = self._scale(c)
                med = q[i, 0, 6]
                q[i, 0] = np.sort(med + s * (q[i, 0] - med))
            self._cache = {key: q}
        return self._cache[key]

    def storage(self, ctxs, qs, series):
        q = self.quantiles(ctxs)[:, SERIES.index(series)]
        grid = [round(x, 4) for x in self.WG.QS_GRID]
        out = np.full((len(ctxs), len(qs)), np.nan)
        for j, lv in enumerate(qs):
            if round(float(lv), 4) in grid:
                out[:, j] = q[:, grid.index(round(float(lv), 4))]
        return out


def register(REGISTRY):
    if all(os.path.exists(os.path.join(ROOT, r)) for r, _ in MEMBERS):
        REGISTRY["wm_gas_ens"] = lambda: GasEns(REGISTRY)
