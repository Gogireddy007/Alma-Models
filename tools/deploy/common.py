"""Shared pieces of the Oct/Nov 2026 deployment of the frozen Alma models (docs/deploy_2026_10_11.md).

Deployment = inference only. Nothing in tools/deploy/ fits, fine-tunes or re-selects an Alma model:
  * `no_training()` makes every torch optimizer step and every autograd backward raise, for the whole process;
  * `verify_checkpoints()` checks every frozen checkpoint against BOTH its winning_models/*/SHA256SUMS pin and (for
    gas, power, fin) tools/forward/FREEZE.json, before and after the run;
  * every Alma model's state_dict is hashed on load and again at the end (`ParamWatch`) - the hashes must match.
Competitor baselines keep their pre-registered protocols (e.g. LEAR refits its lasso every day on the rolling
history available at the origin; OLS / HAR were fitted once on the training period by their own code). Those are the
competitors' definitions, not additional training of Alma.

Windows (target dates, inclusive):
  OCT = 2026-10-01 .. 2026-10-31   ("last month"; note: inside the original TEST year, which was already scored)
  NOV = 2026-11-01 .. data cutoff  ("this month"; strictly after the 2026-11-01 forward freeze)
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("ALMA_DEPLOY_OUT") or os.path.join(ROOT, "runs", "deploy_2026_10_11")   # daily runs: runs/deploy_daily/<date>
WINDOWS = {"oct_2026": (pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-31")),
           "nov_2026": (pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-30"))}
LV9 = np.round(np.arange(0.1, 0.91, 0.1), 2)
QS13 = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [QS13.index(q) for q in LV9]

CKPT_DIRS = {"oil": ["oil_wm_sv"], "power": ["power_wm_power"], "gas": ["gas_wm_gas", "gas_wm_gas_ens"],
             "fin": ["fin_wm_fin"]}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# --------------------------------------------------------------------------- no-training lock
class TrainingAttempted(RuntimeError):
    pass


def no_training():
    """Make any gradient-based training impossible in this process (deployment is inference only)."""
    import torch

    def _blocked(*a, **k):
        raise TrainingAttempted("training is disabled in deployment (tools/deploy/common.no_training)")
    torch.optim.Optimizer.step = _blocked
    torch.Tensor.backward = _blocked
    torch.autograd.backward = _blocked
    torch.set_grad_enabled(False)
    return True


class ParamWatch:
    """Hash the parameters of every torch module that gets loaded; compare at the end."""

    def __init__(self):
        self.mods = []

    def add(self, name, module):
        self.mods.append((name, module, self.digest(module)))

    @staticmethod
    def digest(module):
        h = hashlib.sha256()
        for k, t in sorted(module.state_dict().items()):
            h.update(k.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def check(self):
        out = []
        for name, m, d0 in self.mods:
            d1 = self.digest(m)
            out.append({"module": name, "sha_load": d0, "sha_end": d1, "unchanged": d0 == d1})
        return out


def track_torch_modules(watch):
    """Register every nn.Module whose load_state_dict is called (all Alma wrappers load their weights that way)."""
    import torch.nn as nn
    orig = nn.Module.load_state_dict

    def wrapped(self, *a, **k):
        r = orig(self, *a, **k)
        watch.add(type(self).__name__ + f"#{len(watch.mods)}", self)
        return r
    nn.Module.load_state_dict = wrapped


# --------------------------------------------------------------------------- checkpoint pins
def verify_checkpoints(project):
    """Every checkpoint in winning_models/<dir>/ must match SHA256SUMS and (if listed) tools/forward/FREEZE.json."""
    freeze = json.load(open(os.path.join(ROOT, "tools", "forward", "FREEZE.json")))["sha256"]
    rows = []
    for d in CKPT_DIRS[project]:
        cd = os.path.join(ROOT, "winning_models", d, "checkpoints")
        pins = {}
        for line in open(os.path.join(cd, "SHA256SUMS")):
            if line.strip():
                s, f = line.split()
                pins[os.path.basename(f.lstrip("*"))] = s
        for f in sorted(os.listdir(cd)):
            if not f.endswith(".pt"):
                continue
            got = sha_file(os.path.join(cd, f))
            rel = f"winning_models/{d}/checkpoints/{f}"
            r = {"file": rel, "sha256": got, "sha256sums_ok": pins.get(f) == got,
                 "forward_freeze_ok": (freeze[rel] == got) if rel in freeze else None}
            if not r["sha256sums_ok"] or r["forward_freeze_ok"] is False:
                raise SystemExit(f"refused: checkpoint pin mismatch {r}")
            rows.append(r)
    for rel, s in freeze.items():                     # frozen wrapper code must be unchanged too
        if rel.endswith(".py") and rel.split("/")[1] == {"oil": "commodity"}.get(project, project):
            if sha_file(os.path.join(ROOT, rel)) != s:
                raise SystemExit(f"refused: frozen code changed: {rel}")
            rows.append({"file": rel, "sha256": s, "forward_freeze_ok": True})
    return rows


def ensure_ckpts():
    """The wrappers load runs/<project>/search/ckpt/; a fresh clone keeps the checkpoints in winning_models/."""
    import shutil
    for proj, src in (("gas", "gas_wm_gas"), ("power", "power_wm_power"), ("commodity", "oil_wm_sv")):
        d = os.path.join(ROOT, "runs", proj, "search", "ckpt")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(os.path.join(ROOT, "winning_models", src, "checkpoints")):
            if f.endswith(".pt") and not os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(ROOT, "winning_models", src, "checkpoints", f), d)


# --------------------------------------------------------------------------- scoring
def pinball9(y, q9):
    """y [...], q9 [..., 9] at LV9 -> mean pinball over the 9 levels."""
    d = np.asarray(y)[..., None] - q9
    return np.maximum(LV9 * d, (LV9 - 1) * d).mean(-1)


def boot_ci(d, B=2000, block=1, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(d)
    block = max(1, min(block, n))
    bs = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]
        bs.append(d[idx].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


MIN_EFF_N = 10      # below this many independent scoring units no significance verdict is issued


def compare(per, cand, key, block, higher_better=False, overlap=1):
    """per[model][metric] = per-unit arrays; returns summary with cand-minus-other diffs and bootstrap CIs.

    overlap = how many consecutive units share information (e.g. h-day-ahead forecasts from daily origins overlap
    h-1 days). Effective n = n / max(block, overlap). Below MIN_EFF_N the verdict is 'insufficient_data' whatever the
    bootstrap says: a CI from 2 days, or from 22 overlapping 20-day returns, is not evidence."""
    out = {}
    for mn, v in per.items():
        out[mn] = {k: float(np.nanmean(x)) if np.isfinite(x).any() else None for k, x in v.items()}
        out[mn]["n"] = int(np.isfinite(v[key]).sum())
        if mn != cand:
            dd = per[cand][key] - v[key]
            n_eff = np.isfinite(dd).sum() / max(block, overlap, 1)
            ci = boot_ci(dd, block=max(block, overlap))
            out[mn][f"{cand}_minus_this_{key}"] = float(np.nanmean(dd))
            out[mn][f"ci95_{key}"] = ci
            out[mn]["n_eff"] = float(n_eff)
            if n_eff < MIN_EFF_N or not np.isfinite(ci).all():
                out[mn][f"verdict_{key}"] = "insufficient_data"
                continue
            better = (ci[0] > 0) if higher_better else (ci[1] < 0)
            worse = (ci[1] < 0) if higher_better else (ci[0] > 0)
            out[mn][f"verdict_{key}"] = ("cand_better" if better else "cand_worse" if worse else "not_significant")
    return out


def git_state():
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"head": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain").splitlines()}


def save_preds(project, name, **arrays):
    d = os.path.join(OUT, project, "preds")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.npz")
    np.savez_compressed(p, **arrays)
    return os.path.relpath(p, ROOT), sha_file(p)


def write_json(project, name, obj):
    d = os.path.join(OUT, project)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    json.dump(obj, open(p, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    return p


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
