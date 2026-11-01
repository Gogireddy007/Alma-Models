"""Unit check of wm.ssm.detect_periods (docs/period_lever.md): synthetic sine / multi-season / AR(1) / random walk /
white noise / trend series, CPU, a few seconds. Prints the hit rate (right period in a slot) and the false-fire
rate (any period on a series with none). uv run tools/period_check.py"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm.ssm import detect_periods, PER_LW  # noqa: E402

rng = np.random.default_rng(0)
L, N = PER_LW, 400
t = np.arange(L)


def ar1(phi, n=N):
    e = rng.normal(size=(n, L)); y = np.zeros_like(e)
    for i in range(1, L):
        y[:, i] = phi * y[:, i - 1] + e[:, i]
    return y


def run(name, y, truth):
    y = y.astype(np.float32)
    p = detect_periods(torch.from_numpy(y)).numpy()
    fired = (p > 0).any(1).mean()
    if truth:
        hit = np.mean([any(abs(q - tr) <= max(1, 0.03 * tr) for q in row if q > 0) for row, tr in
                       zip(p, np.broadcast_to(truth, (len(p),)))])
        top = np.mean([abs(row[0] - tr) <= max(1, 0.03 * tr) for row, tr in zip(p, np.broadcast_to(truth, (len(p),)))])
        wrong = np.mean([(row > 0).any() and not any(abs(q - tr) <= max(1, 0.03 * tr) or (q > tr and abs(q / tr - round(q / tr)) < 0.05)
                                                     for q in row if q > 0) for row, tr in zip(p, np.broadcast_to(truth, (len(p),)))])
        print(f"{name:34s} hit={hit:.3f} top1={top:.3f} fired={fired:.3f} wrong={wrong:.3f}  e.g. {p[0].astype(int).tolist()}")
        return hit
    print(f"{name:34s} false_fire={fired:.3f}  e.g. {p[(p > 0).any(1)][0].astype(int).tolist() if fired else '-'}")
    return fired


ph = rng.uniform(0, 2 * np.pi, (N, 1))
res = {}
for per, snr in ((24, 1.0), (24, 0.5), (24, 0.3), (12, 1.0), (7, 1.0), (48, 1.0), (60, 1.0), (96, 1.0), (48, 0.5)):
    res[f"sine{per}_a{snr}"] = run(f"sine p={per} amp/noise={snr}", snr * np.sin(2 * np.pi * t / per + ph) + rng.normal(size=(N, L)), per)
prof = rng.normal(size=(N, 24))
res["profile24"] = run("random 24-profile + AR(0.7)", prof[:, t % 24] + 0.7 * ar1(0.7), 24)
res["sine24_ar"] = run("sine 24 + AR(0.9) + trend", np.sin(2 * np.pi * t / 24 + ph) + 0.5 * ar1(0.9) + 0.02 * t, 24)
res["mult_24_7"] = run("24 + 7 (two seasons)", np.sin(2 * np.pi * t / 24 + ph) + np.sin(2 * np.pi * t / 7) + 0.5 * rng.normal(size=(N, L)), 24)
pp = detect_periods(torch.from_numpy((np.sin(2 * np.pi * t / 24 + ph) + np.sin(2 * np.pi * t / 7) + 0.5 * rng.normal(size=(N, L))).astype(np.float32))).numpy()
print(f"{'  both 24 and 7 found':34s} {np.mean([(np.abs(r - 24) <= 1).any() and (np.abs(r - 7) <= 0).any() for r in pp]):.3f}")
short = np.sin(2 * np.pi * t / 24 + ph) + 0.5 * rng.normal(size=(N, L)); short[:, :L - 100] = np.nan
res["short100"] = run("sine 24, only last 100 observed", short, 24)
miss = np.sin(2 * np.pi * t / 24 + ph) + 0.5 * rng.normal(size=(N, L)); miss[rng.random((N, L)) < 0.2] = np.nan
res["missing20"] = run("sine 24, 20% missing", miss, 24)
print("-- null series (no period)")
for name, y in (("white noise", rng.normal(size=(N, L))), ("AR(1) phi=0.5", ar1(0.5)), ("AR(1) phi=0.9", ar1(0.9)),
                ("AR(1) phi=0.97", ar1(0.97)), ("random walk", rng.normal(size=(N, L)).cumsum(1)),
                ("linear trend + noise", 0.05 * t + rng.normal(size=(N, L))),
                ("exp growth + noise", np.exp(t / 80) + 0.3 * rng.normal(size=(N, L))),
                ("spikes (Poisson 0.05)", rng.poisson(0.05, (N, L)) * 10.0),
                ("level shift", (t > 150) * 3.0 + rng.normal(size=(N, L))),
                ("AR(2) quasi-cycle phi=(1.6,-0.8)", None)):
    if y is None:
        e = rng.normal(size=(N, L)); y = np.zeros_like(e)
        for i in range(2, L):
            y[:, i] = 1.6 * y[:, i - 1] - 0.8 * y[:, i - 2] + e[:, i]
    res[name] = run(name, y, None)

# -- causality probe (docs/audit_sprint.md item 1): the harness's per-origin windows (train_wm.add_period_features:
# the PER_LW points ending at patch j, NaN-padded on the left) must not read points after the origin. Perturb every
# point after patch c and check that periods, context phases and lead phases of origins j < c are unchanged.
from wm.ssm import phase_features, PER_K  # noqa: E402
PATCH, K_STEPS, Lw = 8, 8, 320                           # a training window (CONTEXT 256 + MAX_H 64)


def per_feats(x):
    B, L = x.shape
    n = L // PATCH
    xp = torch.nn.functional.pad(x, (PER_LW - PATCH, 0), value=float("nan"))
    per = detect_periods(xp.unfold(1, PER_LW, PATCH).reshape(B * n, PER_LW)).view(B, n, PER_K)
    s_tok = PATCH * torch.arange(n)
    s_fut = PATCH * (torch.arange(n)[:, None] + 1 + torch.arange(K_STEPS)[None])
    return (per, phase_features(per, s_tok[None].expand(B, n)),
            phase_features(per[:, :, None].expand(B, n, K_STEPS, PER_K), s_fut[None].expand(B, n, K_STEPS)))


tt = np.arange(Lw)
x = np.sin(2 * np.pi * tt / 24 + rng.uniform(0, 6, (64, 1))) + np.sin(2 * np.pi * tt / 7) * (rng.random((64, 1)) < 0.5) \
    + 0.5 * rng.normal(size=(64, Lw))
x[:8, :120] = np.nan                                     # left padding, as short series have
x = torch.from_numpy(x.astype(np.float32))
worst, fired = 0.0, 0
for c in (10, 20, 33, 39):
    x2 = x.clone()
    x2[:, c * PATCH:] = x2[:, c * PATCH:] * 3 + 10 * torch.randn(64, Lw - c * PATCH) + 50 * torch.sin(torch.arange(Lw - c * PATCH) / 3.0)
    a, b = per_feats(x), per_feats(x2)
    d = max((u[:, :c] - v[:, :c]).abs().max().item() for u, v in zip(a, b))
    later = (a[0][:, c:] != b[0][:, c:]).any().item()
    worst, fired = max(worst, d), fired + int((a[0][:, :c] > 0).sum())
    print(f"causality c={c:2d}: max change at origins j<{c} = {d:.1e}; origins >= c changed: {later}")
assert worst == 0.0 and fired > 0, "detect_periods / phase_features read points after the origin"
print(f"causality: OK (exact, {fired} detected period slots in the unchanged origins)")
