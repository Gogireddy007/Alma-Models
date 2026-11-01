"""Harness wrappers for the architect's frozen power world model and its direct twin (docs/power_search.md).

register(REGISTRY) adds the models listed in FROZEN below. Each is a seed ensemble (quantile average, re-sorted) of
checkpoints whose sha256 is verified at load. Inference uses ONLY the harness arguments:
  prices of the last L context days (days D-L..D-1), covariates of days D-L-1..D-2 (the harness sets the D-1 rows to
  NaN; they are never read), and the calendar of the context days and of D (from `dates`, calendar facts only).
Deterministic (no sampling)."""
import datetime as dt
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# name -> list of (checkpoint path relative to ROOT, sha256). Frozen 2026-10-31 (docs/power_search.md section 3):
# recipe B = phys world model, d 64, L 14, lr 1e-3, aux 0.3, 1500 steps; refit on the lead's panel, targets <= 2025-06-30.
_C = "runs/power/search/ckpt/"
FROZEN = {
    "wm_power": [(_C + "frz_phys_none_s0.pt", "be295a716ed3204584077167fae530bf6e950a84022dc7457ac3509c0a4c8a4b"),
                 (_C + "frz_phys_none_s1.pt", "12bdf26936bb8483c58cb89f772fa212e52e0a4bf8e0d247ac445db31e26feec"),
                 (_C + "frz_phys_none_s2.pt", "3b751c979d2715df18da2663b7530d7acfe0c9ce002e83ca3d95a9e7158c6629")],
    "wm_power_direct": [(_C + "frz_phys_direct_s0.pt", "abc82742ae94b02f1fbc65312c19ec469acb51f0065b7d1d02402e6c12da8264"),
                        (_C + "frz_phys_direct_s1.pt", "f96cf84f210184ddaa440535484df18c623df2088e8b2089f6b04cbffc7d27fd"),
                        (_C + "frz_phys_direct_s2.pt", "f0b86f8ca693845fb485845405907493746af9a9d9e6b2077d731725eef15838")],
}
# DEV diagnostics only (registered when POWER_WM_DIAG=1): the same recipe's backtest models (trained on targets
# <= 2023-12-31), out-of-sample on the harness DEV window 2024-07-01..2025-06-30.
DIAG = {
    "wm_power_bt": [(_C + "r2_phys_none_lr1e3.pt", "bf1fb1e987ae3d2effd0546b5fe1b87bbf028eeb42bf5cbbf97d6ba910eeb5a8"),
                    (_C + "f_B_none_s1.pt", "aeda05fe35508a81296698eaabb8b93dd8597297db476ea0adcf704e8ef848c9"),
                    (_C + "f_B_none_s2.pt", "fc755e0649772e202feb62a263d185042881f82f395d3f6a9c51f62cf23771f7")],
    "wm_power_bt_direct": [(_C + "f_B_direct_s0.pt", "f17c0661b91beae1256ad137a57026fad8c615ef36febc56c2ae5561bbe2fa6c"),
                           (_C + "f_B_direct_s1.pt", "b1f668cc6e4c6a0324f33991a60d8e181cf5cedce96cd9adaaad8e030993a378"),
                           (_C + "f_B_direct_s2.pt", "2e29d46d0cab6e6f4773f95fbea15184deb5d95d40c8af8452d7f91db02f700f")],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _to_date(x):
    return dt.date.fromisoformat(str(np.datetime64(x, "D")))


def harness_features(prices, covs, ctx_dates, target_day, L):
    """Harness context -> arrays for wm_data.batch_from_arrays. covs columns: load, wind_on, wind_off, solar."""
    import wm_data as W
    P = np.asarray(prices, float).reshape(-1, 24)
    C = np.asarray(covs, float).reshape(-1, 24, 4)
    assert P.shape[0] == C.shape[0] == len(ctx_dates)
    assert _to_date(ctx_dates[-1]) == _to_date(target_day) - dt.timedelta(days=1)
    Pl = P[-L:]                                                   # D-L..D-1
    Cl = C[-L - 1:-1]                                             # D-L-1..D-2 (row D-1 excluded)
    F = np.stack([Cl[..., 0], Cl[..., 1] + Cl[..., 2], Cl[..., 3], Cl[..., 0] - Cl[..., 1] - Cl[..., 2] - Cl[..., 3]], -1)
    for k in range(4):                                            # rare raw gaps: same-slot value of the previous day
        m = ~np.isfinite(F[..., k])
        if m.any():
            F[..., k] = np.where(m, np.roll(np.nan_to_num(F[..., k], nan=0.0), 1, 0), F[..., k])
    if not np.isfinite(Pl).all():
        Pl = np.where(np.isfinite(Pl), Pl, np.nanmean(Pl, 0, keepdims=True))
    days = [_to_date(d) for d in ctx_dates[-L:]] + [_to_date(target_day)]
    cal = W.calendar_feats(days)
    return Pl, F, cal[:-1], cal[-1]


class Ensemble:
    def __init__(self, members):
        import torch
        import wm_models as M
        self.models, self.meta = [], {"members": []}
        for rel, sh in members:
            path = os.path.join(ROOT, rel)
            got = sha256(path)
            assert got == sh, f"sha256 mismatch for {rel}: {got}"
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]
            m = M.build(a["arch"], a["abl"], a["d"], a["L"])
            m.load_state_dict(ck["model"])
            m.eval()
            self.models.append((m, a["L"]))
            self.meta["members"].append({"path": rel, "sha256": sh, "arch": a["arch"], "abl": a["abl"],
                                         "seed": a["seed"], "steps": a["steps"], "train_end": a["train_end"]})
        self.QL = M.QL.numpy()

    def quantiles_eur(self, prices, covs, dates, rl_shift_frac_wind=None):
        """-> [n, 13, 24] EUR on the model grid. rl_shift_frac_wind: scenario lever (see wm_scenario.py)."""
        import torch
        import wm_data as W
        acc = 0
        for m, L in self.models:
            feats = [harness_features(p, c, d["ctx_dates"], d["target_day"], L) for p, c, d in zip(prices, covs, dates)]
            P = np.stack([f[0] for f in feats])
            F = np.stack([f[1] for f in feats])
            cal = np.stack([f[2] for f in feats])
            cald = np.stack([f[3] for f in feats])
            b, mu, sd = W.batch_from_arrays(P, F, cal, cald, L)
            with torch.no_grad():
                if rl_shift_frac_wind is None:
                    qz = m(b)["q"].numpy()
                else:
                    shift = torch.tensor(rl_shift_frac_wind * F[:, -1, :, 1] / W.FUND_SCALE[3], dtype=torch.float32)
                    qz = m(b, rl_shift=shift)["q"].numpy()
            acc = acc + (qz * sd[:, None, None] + mu[:, None, None])
        q = np.sort(acc / len(self.models), -1)
        return q.transpose(0, 2, 1)


def _factory(name):
    def make():
        ens = Ensemble({**FROZEN, **DIAG}[name])

        def batch(prices, covs, h, qs, dates=None):
            assert h == 24 and dates is not None
            q13 = ens.quantiles_eur(prices, covs, dates)
            out = np.full((len(prices), len(qs), 24), np.nan)
            for j, ql in enumerate(qs):
                k = np.where(np.isclose(ens.QL, ql))[0]
                if len(k):
                    out[:, j] = q13[:, k[0]]
                else:
                    out[:, j] = np.array([[np.interp(ql, ens.QL, q13[n, :, s]) for s in range(24)]
                                          for n in range(len(prices))])
            return out

        def predict(context_prices, context_covariates=None, horizon=24, quantiles=None, **kw):
            return batch([context_prices], [context_covariates], horizon, quantiles,
                         **{k: [v] for k, v in kw.items()})[0]
        predict.batch = batch
        predict.wants_dates = True
        predict.uses_covariates = True
        predict.meta = ens.meta
        return predict
    return make


def register(REGISTRY):
    for name in FROZEN:
        REGISTRY[name] = _factory(name)
    if os.environ.get("POWER_WM_DIAG") == "1":
        for name in DIAG:
            REGISTRY[name] = _factory(name)
