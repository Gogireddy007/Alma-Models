"""
wm/infer_tricks.py: inference-time tricks for a frozen Alma world model (no retraining). docs/infer_streams.md.

Everything here reads ONLY the context the bench passes for one forecast window (<= 2048 points, all before the
origin). No target, no other window's context, no future statistic. Decisions that need a score (expert selection,
interval calibration) are taken per window from in-context backtests: the same predictor is run on the window's own
context truncated by b points and scored against those b observed points.

The harness (train_wm.py) is a script, so it cannot be imported. load_harness() executes the COMMITTED
train_wm.py (git HEAD, so a concurrent edit of the working tree cannot break a run) in a private namespace, up to
the line that would start training/eval, with TSAR_EVAL_ONLY pointing at the checkpoint (its stored config is
restored by train_wm's own eval-only code) and with bench.load_train stubbed out (TRAIN data are not needed to
forecast, and loading them costs ~2 GB). wm/ssm.py is imported from the working tree (its edits are default-off).

Forecaster building blocks (all return [n, NQ, h] raw-space quantiles, sorted):
  chunk(ctxs, h, m, C)            one rollout (h <= MAX_H) from the last C points (C any multiple of PATCH: the
                                   SSM is recurrent, so a longer context is just a longer filter run)
  rollout(ctxs, h, m, C, long)    long horizons: 'paths' (Chronos-2 style quantile paths, = train_wm
                                   predict_paths with C in place of CONTEXT) or 'median' feedback
  Expert(spec).predict(...)       one complete forecaster: context C, integer downsampling factor k, season
                                   override, origin shift s, sign-flip TTA + non-negative clip (train_wm TTA=1).
Combiners (in tools/eval_tricks.py, which caches every expert's forecasts and backtests per task):
  sel / soft   per-window expert selection / weighting by in-context backtest pinball
  cal          per-window interval scaling q' = m + c (q - m), c chosen on in-context backtests
  avg          quantile averaging (Vincentisation) of experts
"""

import io
import math
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

H = None            # the harness namespace (dict) once loaded


def load_harness(ckpt, device="cpu", source="HEAD"):
    """Execute train_wm.py (git `source` revision, or the working tree if source is None) up to training, load
    the frozen checkpoint, return the namespace. Prints train_wm's own startup lines (wm:, wm_causal:)."""
    global H
    import bench as BN                                        # noqa: E402  (tools/)
    BN.load_train = lambda *a, **k: []                         # forecasting needs no TRAIN data
    path = os.path.join(ROOT, "train_wm.py")
    if source:
        src = subprocess.check_output(["git", "-C", ROOT, "show", f"{source}:train_wm.py"], text=True)
        rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", source], text=True).strip()
    else:
        src, rev = open(path).read(), "worktree"
    cut = src.index("\nCKPT = os.path.join(P.CACHE_DIR")
    os.environ["TSAR_EVAL_ONLY"] = ckpt
    os.environ["TSAR_DEVICE"] = device
    os.environ.setdefault("TSAR_SPLIT", "val")
    for k in ("TSAR_WM_TTA", "TSAR_WM_LONG", "TSAR_WM_LONG_PATHS", "TSAR_WM_RECAL"):   # we implement these here
        os.environ.pop(k, None)
    ns = {"__file__": path, "__name__": "train_wm_harness"}
    exec(compile(src[:cut], f"train_wm.py@{rev}", "exec"), ns)
    ck = ns["CK0"]
    ns["model"].load_state_dict(ck["model"])
    if ns["EMA"]:
        ns["ema"].load_state_dict(ck.get("ema", ck["model"]))
    ns["ema"].eval()
    ns["_ck_step"], ns["_rev"] = ck["step"], rev
    del ns["CK0"]
    print(f"infer_tricks: harness train_wm.py@{rev}, checkpoint {ckpt} step {ns['_ck_step']} on {device}",
          flush=True)
    H = ns
    return ns


# ---------------------------------------------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def chunk(ctxs, h, m, C=256):
    """One rollout from the last C points of each context (right-aligned, NaN-padded). [n, NQ, h], h <= MAX_H."""
    PATCH, MAX_H = H["PATCH"], H["MAX_H"]
    assert h <= MAX_H and C % PATCH == 0
    m = max(1, int(m))
    EB = 512 if C <= 512 else max(64, 512 * 512 // C)
    out = []
    for s in range(0, len(ctxs), EB):
        part = ctxs[s:s + EB]
        x = np.full((len(part), C), np.nan, dtype=np.float32)
        for i, c in enumerate(part):
            c = c[-C:]
            x[i, C - len(c):] = c
        q = H["forecast"](torch.from_numpy(x).to(H["DEVICE"]), h, m)            # [B, h, NQ]
        out.append(q.permute(0, 2, 1).float().cpu().numpy())
    q = np.concatenate(out) if out else np.zeros((0, H["NQ"], h), np.float32)
    return np.sort(np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0), axis=1)


@torch.no_grad()
def rollout(ctxs, h, m, C=256, long="paths"):
    """Full horizon: chunk 1 is one rollout; beyond MAX_H, 'paths' (train_wm predict_paths with context C) or
    'median' feedback (train_wm predict_one)."""
    MAX_H, NQ = H["MAX_H"], H["NQ"]
    ctxs = [np.asarray(c, dtype=np.float32) for c in ctxs]
    q = chunk(ctxs, min(MAX_H, h), m, C)
    if h <= MAX_H:
        return q
    out = [q]
    if long == "median":
        for s in range(MAX_H, h, MAX_H):
            ctxs = [np.concatenate([c, q[i, NQ // 2]])[-C:] for i, c in enumerate(ctxs)]
            q = chunk(ctxs, min(MAX_H, h - s), m, C)
            out.append(q)
        return np.concatenate(out, -1)
    PATH_IDX, PATH_W, n = H["PATH_IDX"], H["PATH_W"], len(ctxs)
    S = len(PATH_IDX)
    paths = [list(ctxs) for _ in PATH_IDX]
    for s in range(MAX_H, h, MAX_H):
        hc = min(MAX_H, h - s)
        for p, qi in enumerate(PATH_IDX):
            paths[p] = [np.concatenate([c, q[i, qi]])[-C:] for i, c in enumerate(paths[p])]
        comp = chunk([c for pp in paths for c in pp], hc, m, C)
        q = H["mixture_quantiles"](comp.reshape(S, n, NQ, hc).transpose(1, 0, 2, 3), PATH_W)
        out.append(q)
    return np.concatenate(out, -1)


def nonneg(ctxs):
    out = np.zeros(len(ctxs), dtype=bool)
    for i, c in enumerate(ctxs):
        v = np.asarray(c, dtype=np.float32)
        v = v[np.isfinite(v)]
        out[i] = v.size > 0 and bool((v >= 0).all())
    return out


# ---------------------------------------------------------------------------------------------------------------
# Down / up-sampling (stream 4)
# ---------------------------------------------------------------------------------------------------------------

def downsample(c, k):
    """Block means of k consecutive points, aligned to the END of the context (the last block ends at the
    origin); the oldest len % k points are dropped. A block with no observed value is NaN."""
    c = np.asarray(c, dtype=np.float32)
    n = (len(c) // k) * k
    if n == 0:
        return c[-1:].copy()
    b = c[len(c) - n:].reshape(-1, k)
    cnt = np.isfinite(b).sum(1)
    s = np.where(np.isfinite(b), b, 0).sum(1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype(np.float32)


def upsample(q, k, h, last_block=None):
    """Coarse quantiles q [n, NQ, h'] (block means) -> fine [n, NQ, h] by linear interpolation between block
    centres. Block j (0-based, future) covers fine steps j*k .. j*k+k-1, centre j*k + (k-1)/2. Before the first
    centre we interpolate from the last observed block mean (centre at -(k+1)/2) when given, else hold."""
    n, NQ, hc = q.shape
    centres = np.arange(hc) * k + (k - 1) / 2.0
    t = np.arange(h, dtype=np.float64)
    if last_block is not None:
        xs = np.concatenate([[-(k + 1) / 2.0], centres])
        out = np.empty((n, NQ, h), np.float32)
        for i in range(n):
            lb = last_block[i]
            for j in range(NQ):
                ys = np.concatenate([[lb if np.isfinite(lb) else q[i, j, 0]], q[i, j]])
                out[i, j] = np.interp(t, xs, ys)
        return out
    flat = q.reshape(-1, hc)
    out = np.stack([np.interp(t, centres, r) for r in flat]).astype(np.float32)
    return out.reshape(n, NQ, h)


# ---------------------------------------------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------------------------------------------

class Expert:
    """A complete forecaster parsed from a spec such as 'c256' (= the current best setup: context 256, paths,
    sign-flip TTA + non-negative clip), 'c512', 'c256_d2', 'c256_m168', 'c256_s4', 'c256_raw', 'c256_med'.
      c<C>   model context in points (multiple of PATCH; default 256)
      d<k>   downsample by k (block means), forecast ceil(h/k) coarse steps with season round(m/k), upsample
      m<M>   season override (seasonal-naive reference period); 'mx<j>' multiplies the bench season by j
      s<s>   origin shift: forecast from ctx[:-s] for h + s steps and keep the last h (multi-origin, stream 3)
      raw    no TTA (else TTA=1: sym + clip, as TSAR_WM_TTA=1)
      med    median feedback for h > MAX_H instead of paths
    """

    def __init__(self, spec):
        self.spec = spec
        self.C, self.k, self.mo, self.mx, self.s, self.tta, self.long = 256, 1, None, 1, 0, True, "paths"
        for tok in spec.split("_"):
            if tok == "raw":
                self.tta = False
            elif tok == "med":
                self.long = "median"
            elif tok.startswith("mx"):
                self.mx = int(tok[2:])
            elif tok[0] == "c":
                self.C = int(tok[1:])
            elif tok[0] == "d":
                self.k = int(tok[1:])
            elif tok[0] == "m":
                self.mo = int(tok[1:])
            elif tok[0] == "s":
                self.s = int(tok[1:])
            else:
                raise ValueError(f"expert spec {spec!r}: unknown token {tok!r}")

    def season(self, m):
        m = self.mo if self.mo else m * self.mx
        return max(1, int(round(m / self.k)))

    def _one(self, ctxs, h, m):
        """One pass (no TTA) on prepared contexts."""
        k, s = self.k, self.s
        if s:
            ctxs = [c[:-s] if len(c) > s + 1 else c[:1] for c in ctxs]
            h = h + s
        if k > 1:
            co = [downsample(c, k) for c in ctxs]
            hc = int(math.ceil(h / k))
            q = rollout(co, hc, self.season(m), self.C, self.long)
            last = np.array([cc[-1] for cc in co], dtype=np.float64)
            q = upsample(q, k, h, last)
        else:
            q = rollout(ctxs, h, self.season(m), self.C, self.long)
        return q[..., s:] if s else q

    def predict(self, ctxs, h, m):
        ctxs = [np.asarray(c, dtype=np.float32) for c in ctxs]
        q = self._one(ctxs, h, m)
        if self.tta:
            qm = self._one([-c for c in ctxs], h, m)
            q = 0.5 * (q + (-qm)[:, ::-1])
            el = nonneg(ctxs)
            q = np.where(el[:, None, None], np.maximum(q, 0), q)
        return np.sort(np.ascontiguousarray(q, dtype=np.float32), axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Scoring helpers (identical maths to prepare._score) and per-window losses for in-context decisions
# ---------------------------------------------------------------------------------------------------------------

QL = None


def levels():
    global QL
    if QL is None:
        import prepare as P
        QL = np.asarray(P.QUANTILES, dtype=np.float64)
    return QL


def window_pinball(q, y):
    """Per-window sum over quantiles and steps of the pinball loss (NaN targets ignored). q [n, NQ, b], y [n, b]."""
    tau = levels()[None, :, None]
    mask = np.isfinite(y)
    err = np.nan_to_num(y)[:, None, :] - q
    pb = np.maximum(tau * err, (tau - 1) * err) * mask[:, None, :]
    return pb.sum(axis=(1, 2))


def scale_about_median(q, c):
    """q' = med + c (q - med), c [n] or scalar; keeps the median and the order of the levels for c >= 0."""
    med = q[:, q.shape[1] // 2: q.shape[1] // 2 + 1, :]
    c = np.asarray(c, dtype=np.float32)
    c = c[:, None, None] if c.ndim == 1 else c
    return med + c * (q - med)


# ---------------------------------------------------------------------------------------------------------------
# Test-time system identification (stream 1): per-window gradient steps on the window's own context
# ---------------------------------------------------------------------------------------------------------------
#
# Per row (forecast window) we learn, starting from zero (= the frozen model exactly):
#   spectrum delta: every mixer's decay rate exp(a_log + dlog) and angular rate a_im * exp(dang), [P] each; the
#                   same per-series system is used to assimilate the context and to imagine the future
#   readout FiLM:   decoder input (o_k + ref_proj(ref_k)) * (1 + g) + beta, [D] each
# by a few Adam steps on the training loss itself (multi-origin pinball), computed on the last C + MAX_H points
# of the window's context: every in-context origin whose 64-step future is (partly) observed in the context.
# Only the shared frozen weights + the row's own past are used. The adapted model then forecasts as usual
# (paths / TTA wrap it; the -x pass of TTA adapts on -x separately).

ADAPT = None        # active per-row parameters during a rollout: dict(dlog, dang, g, beta, n)


def _ssm_forward(ctx, b, j, prm, rowmap):
    """WMModel.forward for family ssm (abl none, SPEC/FB/LOOP off) with per-row spectrum / FiLM deltas.
    prm tensors are [R, ...]; rowmap [B] maps each context row to its parameter row."""
    fam, ref_skip = H["ema"].family, H["ema"].ref_skip
    cfg = fam.cfg
    P, K = cfg.patch, cfg.k_steps
    mods = []
    for li, mx in enumerate(fam.mixers):
        dl = prm["dlog"][rowmap, li] if prm is not None else 0.0
        da = prm["dang"][rowmap, li] if prm is not None else 0.0
        mods.append((torch.exp(mx.a_log + dl), mx.a_im * torch.exp(da)))        # [B, P] (or [P])
    r = fam.embed(ctx.tok)
    layers = []
    for (mx, md) in zip(fam.mixers, mods):
        x, gate = mx.pre(r)
        xp = torch.nn.functional.pad(x, (0, 0, mx.kc - 1, 0))
        win = xp.unfold(1, mx.kc, 1).transpose(-1, -2)
        xc = mx.conv(win)
        mb = (md[0][:, None] if md[0].dim() == 2 else md[0], md[1][:, None] if md[1].dim() == 2 else md[1])
        lr_, li_, vr, vi = mx.coeffs(xc, mb)
        hr = torch.zeros_like(vr[:, 0]); hi = torch.zeros_like(hr)
        Hr, Hi = [], []
        for t in range(xc.shape[1]):
            hr, hi = lr_[:, t] * hr - li_[:, t] * hi + vr[:, t], lr_[:, t] * hi + li_[:, t] * hr + vi[:, t]
            Hr.append(hr); Hi.append(hi)
        Hr, Hi = torch.stack(Hr, 1), torch.stack(Hi, 1)
        layers.append((Hr, Hi, win[:, :, 1:]))
        r = r + mx.post(Hr, Hi, xc, gate)
    N = len(b)
    ref = ctx.ref[b, j].view(N, K, P)
    U = fam.step_emb[None] + fam.drive(ref)
    st = [(Hr[b, j], Hi[b, j], Bf[b, j]) for Hr, Hi, Bf in layers]
    dmods = [(md[0][b] if md[0].dim() == 2 else md[0], md[1][b] if md[1].dim() == 2 else md[1]) for md in mods]
    outs = []
    for k in range(K):
        rr = U[:, k]
        for li, mx in enumerate(fam.mixers):
            hr, hi, buf = st[li]
            rr, hr, hi, buf = mx.step(rr, hr, hi, buf, dmods[li])
            st[li] = (hr, hi, buf)
        outs.append(fam.norm_f(rr))
    O = torch.stack(outs, 1) + fam.ref_proj(ref)
    if prm is not None:
        rows = rowmap[b]
        O = O * (1 + prm["g"][rows, None]) + prm["beta"][rows, None]
    q = fam.decode_head(O).view(N, K, P, cfg.nq)
    return (q + ref[..., None] * ref_skip).reshape(N, K * P, cfg.nq)


def _check_adaptable():
    fam = H["ema"].family
    if H["WM_NAME"] != "ssm" or fam.cfg.abl != "none" or fam.spec or fam.fb_on or fam.loop != 1:
        raise SystemExit("test-time adaptation is implemented for family ssm, abl none, SPEC/FB/LOOP off")


def adapt(ctxs, m, C, steps, lr, wd=0.0, verbose=False):
    """Fit per-row deltas on each context's last C + MAX_H points. Returns the parameter dict (detached)."""
    _check_adaptable()
    PATCH, MAX_H, DEV = H["PATCH"], H["MAX_H"], H["DEVICE"]
    fam = H["ema"].family
    W = C + MAX_H
    n = len(ctxs)
    x = np.full((n, W), np.nan, dtype=np.float32)
    for i, c in enumerate(ctxs):
        c = np.asarray(c, dtype=np.float32)[-W:]
        x[i, W - len(c):] = c
    x = torch.from_numpy(x).to(DEV)
    L, D, Pm = len(fam.mixers), fam.cfg.d_model, fam.S
    prm = {"dlog": torch.zeros(n, L, Pm, device=DEV), "dang": torch.zeros(n, L, Pm, device=DEV),
           "g": torch.zeros(n, D, device=DEV), "beta": torch.zeros(n, D, device=DEV)}
    for v in prm.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(prm.values(), lr=lr)
    with torch.no_grad():
        ctx = H["make_context"](x, torch.full((n,), max(1, int(m)), device=DEV))
        nP = ctx.tok.shape[1]
        pad = torch.full((n, MAX_H), float("nan"), device=DEV)
        fut = torch.cat([x, pad], 1).unfold(1, MAX_H, PATCH)[:, 1:nP + 1]
        valid = (ctx.cnt > 0)[..., None] & ~torch.isnan(fut)
        tgt = H["to_model_space"](torch.nan_to_num(fut), ctx.loc[..., None], ctx.scale[..., None])
        keep = valid.any(-1).view(-1)
        bj = torch.nonzero(keep).squeeze(1)
        b, j = bj // nP, bj % nP
        mask = valid.reshape(n * nP, MAX_H)[keep].float()
        tg = tgt.reshape(n * nP, MAX_H)[keep]
        Qt = torch.tensor(H["P"].QUANTILES, device=DEV)
        rowmap = torch.arange(n, device=DEV)
        # per-row normaliser so every row's loss is a mean over its own observed targets
        cnt_row = torch.zeros(n, device=DEV).index_add_(0, b, mask.sum(1)).clamp(min=1)
    if len(bj) == 0:
        return {k: v.detach() for k, v in prm.items()}
    p_frozen = list(H["ema"].parameters())
    for p_ in p_frozen:
        p_.requires_grad_(False)
    hist = []
    for it in range(steps):
        pred = _ssm_forward(ctx, b, j, prm, rowmap)
        err = tg.unsqueeze(-1) - pred
        pb = (torch.maximum(Qt * err, (Qt - 1) * err) * mask.unsqueeze(-1)).sum((1, 2))
        loss_row = torch.zeros(n, device=DEV).index_add(0, b, pb) / cnt_row / len(Qt)
        loss = loss_row.sum()
        if wd:
            loss = loss + wd * sum((v ** 2).sum() for v in prm.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(loss_row.mean().item())
    if verbose:
        print(f"    adapt n={n} origins={len(bj)} loss {hist[0]:.4f} -> {hist[-1]:.4f}", flush=True)
    return {k: v.detach() for k, v in prm.items()}


@torch.no_grad()
def forecast_adapted(x, h, m):
    """Drop-in for train_wm.forecast using the active ADAPT parameters (rows mapped by row index % n)."""
    B = x.shape[0]
    ctx = H["make_context"](x, torch.full((B,), int(m), device=x.device))
    nP = ctx.tok.shape[1]
    rowmap = torch.arange(B, device=x.device) % ADAPT["n"]
    q = _ssm_forward(ctx, torch.arange(B, device=x.device), torch.full((B,), nP - 1, device=x.device),
                     ADAPT, rowmap)
    return H["from_model_space"](q[:, :h], ctx.loc[:, -1, None, None], ctx.scale[:, -1, None, None])


class AdaptExpert(Expert):
    """Expert + test-time adaptation: extra tokens t<steps> (Adam steps), l<lr*1000> (default 10 = 0.01).
    e.g. 'c256_t5', 'c256_t10_l20'. Windows are processed in groups of GROUP so the paths' S*n rows fit one
    forward (row index % n gives the window)."""
    GROUP = 48

    def __init__(self, spec):
        toks = spec.split("_")
        self.steps = next(int(t[1:]) for t in toks if t[0] == "t")
        lt = [t for t in toks if t[0] == "l"]
        self.lr = int(lt[0][1:]) / 1000 if lt else 0.01
        super().__init__("_".join(t for t in toks if t[0] not in "tl"))
        self.spec = spec

    def _one(self, ctxs, h, m):
        global ADAPT
        if self.k != 1 or self.s:
            raise SystemExit("AdaptExpert: d/s tokens not supported")
        out = []
        orig = H["forecast"]
        for s in range(0, len(ctxs), self.GROUP):
            part = ctxs[s:s + self.GROUP]
            with torch.enable_grad():
                prm = adapt(part, self.season(m), self.C, self.steps, self.lr)
            ADAPT = dict(prm, n=len(part))
            H["forecast"] = forecast_adapted
            try:
                out.append(rollout(part, h, self.season(m), self.C, self.long))
            finally:
                H["forecast"] = orig
                ADAPT = None
        return np.concatenate(out)


def make_expert(spec):
    return AdaptExpert(spec) if any(t[0] == "t" and t[1:].isdigit() for t in spec.split("_")) else Expert(spec)
