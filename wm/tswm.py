"""
Family `tswm`: the Timestral backbone, re-implemented from its Apache-2.0 source and trained from scratch (no
pretrained weights or outputs), with a WORLD-MODEL HORIZON BLOCK. TSAR_WM_ABL=direct is the faithful Timestral
Decoupled Horizon Forecasting (DHF) decoder and serves as the control. The anatomy, with file:line citations, is in
docs/timestral_anatomy.md.

Backbone (timestral/layers.py, model.py), the same in every mode:
  embed    ResidualBlock(tok_dim, 4D, D) = out(silu(hidden(x))) + residual(x), with bias (model.py:41, layers.py:30).
           The harness token is [patch values in its own asinh frame, observed mask, 3 frame features]. Timestral's
           is [values, missing]; the frame features are the harness's (its NORM=last frame moves per patch).
  blocks   pre-norm RMSNorm (eps 1e-6). Time attention: fused QKV with no bias, partial RoPE on the first half of
           each head (rope_partial_factor 0.5, theta 1e4, the sin/cos concatenated-halves form), SDPA, then the
           output multiplied by sigmoid(W_g x) before W_O ("sdpa_sigmoid"). Then the SwiGLU MLP w3(gate * silu(up)),
           hidden = 8/3 D (512 at D = 192, exactly as Timestral-5M), no bias. Layer kinds follow Timestral's
           schedule: every 4th layer is "time_variate". On univariate rows the variate attention has one key, so
           softmax = 1 and it reduces exactly to W_O(W_V x * sigmoid(W_g x)); only W_V, W_g, W_O are built (the
           Q/K weights would be dead). Final RMSNorm.
  masks    encoder: causal over valid keys (a patch is valid once the series has started, cnt > 0), diagonal always
           open (masks.py:25). RoPE positions are 0..n-1; Timestral anchors them at the first observed patch, which
           is a no-op under RoPE (scores depend only on position differences).
  head     ResidualBlock(D, 4D, PATCH*NQ, no bias), then Timestral's monotone_cumsum (model.py:57): the median is
           free, the other knots fan out by cumsum(width_factor * dp * softplus(raw)), dp = 0.1, width_factor
           TSAR_WM_TSWM_WF (3.0, as the released configs). The 9 harness levels are produced directly.
  frame    TSAR_WM_TSWM_FRAME (default loc): the advancing refine frame (scaler.py:165, forecaster.py:458) made
           consistent with the harness frame. Lead k's head output is read in a frame whose location is the LAST
           imagined median value of lead k-1 (the harness NORM=last rule run over the imagined path), converted
           to the origin frame as asinh(sinh(q_k) * s_k + l_k). Unlike Timestral, the frame is computed from the
           model's own (detached) medians in training too, so training and inference use the same frames.
           `adv` also advances the scale (running mean |step| over context + imagined medians, floored at 0.5 of
           the origin scale); `origin` disables it (every lead in the origin frame).

Horizon block. For origin (b, j) the K lead tokens sit at RoPE positions j+1 .. j+K and run through the SAME blocks
as the encoder (shared weights, as in DHF). In every layer a lead token attends to the valid encoder keys 0..j of its
row (the memory read; masks.py:43) and causally to the lead tokens before it and itself (masks.py:44). Lead input:
x_k = forecast_query + drive(ref_k) (drive zero-initialised, so at init the token is exactly Timestral's single
learned query; ref_k is the harness's causal seasonal-naive patch, an exogenous "action" every family receives).

Ablations (TSAR_WM_ABL):
  direct  Timestral DHF. All K tokens are decoded in parallel from identical inputs; nothing is carried from lead k
          to lead k+1 except what attention reads from the other queries' hidden states in the same pass.
  none    the world model: an imagined latent trajectory. The final hidden state z_k of lead k (after the full
          stack and the final norm) is fed back through a learned transition T (D x D, zero-init) into lead k+1's
          input: x_{k+1} = forecast_query + drive(ref_{k+1}) + T(z_k), with z_{-1} = the encoder's state at the
          origin patch j. Leads are decoded sequentially (with a KV cache of the imagined tokens), so the whole
          stack is the transition function and the state is carried and rolled; each lead still reads the
          context through attention. Same attention pattern and the same per-token MACs as direct; params +D^2+D
          (2.5% at width 1). At init T = 0, so none starts as exactly direct.
  copy    no transition: every lead gets T(z_{-1}) (the origin state) plus its own input, in parallel. Same params
          as none.
  zero    the rolled-out states z_k are zeroed before the head (rollout as none; same params as none, usable
          eval-only on a none checkpoint).
Optional aux (TSAR_WM_TSWM_LAT=<w>, default 0 = off): latent prediction, z_k -> Linear -> the stop-gradient encoder
state at patch j+1+k (when it is inside the training window), MSE. Adds a D x D predictor.

Internal patch: the harness PATCH (8) is kept. Timestral's 32 is tied to its 32k context (1024 tokens); with
CONTEXT = 256 a 32-point patch would leave 8 context tokens and K = 2 lead tokens, which removes most of what the
attention and the horizon dynamics can do. K can grow freely (RoPE positions, no per-lead tables), so TSAR_MAX_H and
TSAR_WM_LONG=paths need no change here.

Compute: decode runs on a dense (row, origin) grid when most origins of the batch are requested (training: no gather
of encoder keys, logits [B, n, K, H, n]) and on gathered per-origin keys otherwise (eval: one origin per row; the
harness's MAC count). TSAR_WM_TSWM_PATH=auto|dense|gather forces one (identical maths; used by the tests).

Transition (TSAR_WM_TSWM_TRANS, 2026-10-31; default linear = the T above, bit-identical to d6371a2):
  linear  x_{k+1} = forecast_query + drive(ref_{k+1}) + T(z_k), T a zero-initialised D x D Linear.
  ssm     a selective diagonal complex state-space recurrence ACROSS LEADS replaces T (LeadSSM below, the S6/S5 step
          of wm/ssm.py): a state s of P complex modes is carried from lead to lead alongside the stack,
              dt_k = softplus(W_dt z_k + b_dt)                         [P], input-dependent step
              s_k  = exp(dt_k * a) * s_{k-1} + dt_k * (B z_k),  a = -exp(a_log) + i a_im  (|lambda| < 1, stable)
              y_k  = C [Re s_k, Im s_k] + Dskip * z_k                  -> x_{k+1} = forecast_query + drive + y_k
          with s_{-2} = 0 and z_{-1} the encoder state at the origin, so lead k+1 sees the whole imagined path
          z_{-1..k} through s (a second, explicit carried state next to the stack's KV cache) instead of z_k only.
          C and Dskip are zero-initialised, so y = 0 at init and `none` again starts as exactly `direct`. Init of
          a: radii 0.6..0.995 per lead, periods log-spaced over 2..64 leads (as wm/ssm.py). copy: one step from
          s = 0 on z_{-1}, the same y for every lead (no carried state). zero: as none. Refused with direct (no
          transition to replace). TSAR_WM_TSWM_SSM_STATE = P (default D // 3: 32 at width 1).
          Params (TSAR_WM_INFO=1): ssm is 5PD + 5P + D = 15,616 against T's 9,312, so none/copy/zero = 374,665
          at width 1 (+1.7% vs linear none, within the +3% rule; no matched control needed); MACs 12.26M vs 12.21M.
          At the 5M shape (D 192, P 64): 61,952 vs 37,056 -> 4,505,993 (+0.6%).

Context read (TSAR_WM_TSWM_READ, 2026-10-31, E7 of docs/almats_research_program.md; default all = bit-identical):
  all     every lead token attends to the valid encoder keys 0..j (DHF, masks.py:43).
  first   only lead 0 reads the context; leads >= 1 attend to the lead tokens only (themselves and, causally, the
          earlier leads; in none/zero that is the KV cache of the imagined tokens). This closes the per-lead
          bypass of the carried state: past lead 0, the future is reachable only through what earlier leads
          computed. Applied identically in every mode. `direct` + first stays a meaningful, exactly matched
          control: its leads >= 1 still reach lead 0's context read through Dec-to-Dec attention on the hidden
          states of the same pass (the DHF path), but nothing is fed back from a lead's output into the next
          lead's input. So none-vs-direct at READ=first isolates the fed-back state (T or the lead SSM) when
          the context can no longer be re-read; same masks, same params as at READ=all.

Env keys: TSAR_WM_TSWM_FRAME (loc|adv|origin), TSAR_WM_TSWM_WF (3.0), TSAR_WM_TSWM_LAT (0), TSAR_WM_TSWM_PATH (auto),
TSAR_WM_TSWM_TRANS (linear|ssm), TSAR_WM_TSWM_SSM_STATE (D // 3), TSAR_WM_TSWM_READ (all|first).
Params (incl. ref_skip) / MACs per forecast (TSAR_WM_INFO=1):
  width 1 (D 96, 2 layers, 4 heads, MLP 256): none/copy/zero 368,361 / 12.2M; direct 359,049 (-2.5%) / 12.1M.
  ladder (none; direct): 1.6667 -> 969,769 (944,009), 3.3333 -> 3,741,769 (3,639,049), 5.3333 -> 9,427,465
  (9,164,809).
  Timestral-5M shape (TSAR_WM_WIDTH=2 TSAR_N_LAYERS=8 TSAR_N_HEADS=3: D 192, 8 layers, 3 heads, MLP 512, layers 4
  and 8 time_variate): none/copy/zero 4,481,097 / 173.6M; direct 4,444,041 / 173.4M. The backbone (4.06M) has the
  released 5M's shape; the released model has 5.2M because its 32-point patch and 21-knot head make the embedding
  and the head larger. At this size train with TSAR_WM_ORIGINS=1024 (batch 128 with all origins needs ~11 GB).
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily, refuse

_LAT = float(os.environ.get("TSAR_WM_TSWM_LAT", "0"))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class ResidualBlock(nn.Module):
    """output(silu(hidden(x))) + residual(x) (timestral/layers.py:30)."""

    def __init__(self, i, h, o, bias):
        super().__init__()
        self.hidden_layer = nn.Linear(i, h, bias=bias)
        self.output_layer = nn.Linear(h, o, bias=bias)
        self.residual_layer = nn.Linear(i, o, bias=bias)

    def forward(self, x):
        return self.output_layer(F.silu(self.hidden_layer(x))) + self.residual_layer(x)


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        gate, up = self.w12(x).chunk(2, -1)
        return self.w3(gate * F.silu(up))


def _rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


class TimeAttn(nn.Module):
    """Sigmoid-gated multi-head attention with partial RoPE (timestral/layers.py:143). The attention itself is done
    by the caller, because encoder and lead tokens see different keys."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.wQKV = nn.Linear(d, 3 * d, bias=False)
        self.wO = nn.Linear(d, d, bias=False)
        self._gate = nn.Linear(d, d, bias=False)

    def qkv(self, xn, cos, sin):
        """xn [..., T, D] -> q, k, v [..., T, H, hd], RoPE on q and k (cos/sin [..., T, 1, hd])."""
        q, k, v = self.wQKV(xn).unflatten(-1, (3, self.h, self.hd)).unbind(-3)
        return q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin, v

    def out(self, o, xn):
        """o [..., T, H, hd] attention output -> W_O(o * sigmoid(W_g xn))."""
        return self.wO(o.flatten(-2) * torch.sigmoid(self._gate(xn)))


class VariateAttn(nn.Module):
    """Timestral's variate attention on a single-row (univariate) series: one key, softmax = 1."""

    def __init__(self, d):
        super().__init__()
        self.wV = nn.Linear(d, d, bias=False)
        self.wO = nn.Linear(d, d, bias=False)
        self._gate = nn.Linear(d, d, bias=False)

    def forward(self, xn):
        return self.wO(self.wV(xn) * torch.sigmoid(self._gate(xn)))


class Block(nn.Module):
    def __init__(self, d, heads, hidden, variate):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = TimeAttn(d, heads)
        self.var_norm = RMSNorm(d) if variate else None
        self.var = VariateAttn(d) if variate else None
        self.mlp_norm = RMSNorm(d)
        self.mlp = SwiGLU(d, hidden)

    def tail(self, x):
        """Everything after the time attention: the variate sublayer (time_variate kind) and the MLP."""
        if self.var is not None:
            x = x + self.var(self.var_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LeadSSM(nn.Module):
    """Selective diagonal complex SSM carried across leads (TSAR_WM_TSWM_TRANS=ssm). step(z, state) -> (y, state);
    state None = zero. Real arithmetic on (re, im) pairs, as wm/ssm.py (MPS has no complex autograd)."""

    def __init__(self, D, P):
        super().__init__()
        self.P = P
        self.BC = nn.Linear(D, 3 * P)                                    # [dt | B re | B im], one matmul
        self.C = nn.Linear(2 * P, D, bias=False)
        self.Dskip = nn.Parameter(torch.zeros(D))
        with torch.no_grad():
            self.BC.weight[:P].normal_(std=0.02)
            self.BC.bias.zero_()
            self.BC.bias[:P] = math.log(math.e - 1)                     # softplus(dt) = 1 lead at init
            self.C.weight.zero_()                                        # y = 0 at init: none starts as direct
        radius = torch.linspace(0.6, 0.995, P)[torch.randperm(P, generator=torch.Generator().manual_seed(1))]
        period = torch.exp(torch.linspace(math.log(2.0), math.log(64.0), P))
        self.a_log = nn.Parameter(torch.log(-torch.log(radius)))
        self.a_im = nn.Parameter(2 * math.pi / period)

    def step(self, z, state=None):
        P = self.P
        g = self.BC(z)
        dt = F.softplus(g[..., :P])
        mag = torch.exp(-dt * torch.exp(self.a_log))
        ang = dt * self.a_im
        lr, li = mag * torch.cos(ang), mag * torch.sin(ang)
        vr, vi = dt * g[..., P:2 * P], dt * g[..., 2 * P:]
        if state is None:
            hr, hi = vr, vi
        else:
            sr, si = state
            hr, hi = lr * sr - li * si + vr, lr * si + li * sr + vi
        return self.C(torch.cat([hr, hi], -1)) + self.Dskip * z, (hr, hi)


def monotone_quantiles(raw, width_factor):
    """timestral/model.py:57 for 9 equally spaced levels (dp = 0.1): median free, cumsum of softplus increments."""
    m = raw.shape[-1] // 2
    inc = width_factor * 0.1 * F.softplus(raw)
    med = raw[..., m:m + 1]
    up = torch.cumsum(inc[..., m + 1:], -1)
    down = torch.flip(torch.cumsum(torch.flip(inc[..., :m], [-1]), -1), [-1])
    return torch.cat([med - down, med, med + up], -1)


class TSWMFamily(WMFamily):
    name = "tswm"
    aux_weights = {"lat": _LAT} if _LAT > 0 else {}
    env_keys = ("TSAR_WM_TSWM_FRAME", "TSAR_WM_TSWM_WF", "TSAR_WM_TSWM_LAT", "TSAR_WM_TSWM_PATH",
                "TSAR_WM_TSWM_TRANS", "TSAR_WM_TSWM_SSM_STATE", "TSAR_WM_TSWM_READ")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, H, P, NQ = cfg.d_model, cfg.n_heads, cfg.patch, cfg.nq
        if NQ != 9:
            refuse(self.name, cfg.abl, f"the monotone head assumes 9 equally spaced levels, got {NQ}")
        self.frame = os.environ.get("TSAR_WM_TSWM_FRAME", "loc")
        if self.frame not in ("loc", "adv", "origin"):
            raise SystemExit(f"TSAR_WM_TSWM_FRAME={self.frame!r} not in (loc, adv, origin)")
        if self.frame != "origin" and os.environ.get("TSAR_NORM", "last") != "last":
            raise SystemExit("TSAR_WM_TSWM_FRAME=loc/adv advances the harness NORM=last frame; use FRAME=origin "
                             "with another TSAR_NORM")
        self.wf = float(os.environ.get("TSAR_WM_TSWM_WF", "3.0"))
        self.path = os.environ.get("TSAR_WM_TSWM_PATH", "auto")
        if self.path not in ("auto", "dense", "gather"):
            raise SystemExit(f"TSAR_WM_TSWM_PATH={self.path!r} not in (auto, dense, gather)")
        self.trans_kind = os.environ.get("TSAR_WM_TSWM_TRANS", "linear")
        if self.trans_kind not in ("linear", "ssm"):
            raise SystemExit(f"TSAR_WM_TSWM_TRANS={self.trans_kind!r} not in (linear, ssm)")
        if self.trans_kind != "ssm" and os.environ.get("TSAR_WM_TSWM_SSM_STATE"):
            raise SystemExit("TSAR_WM_TSWM_SSM_STATE needs TSAR_WM_TSWM_TRANS=ssm")
        if self.trans_kind == "ssm" and cfg.abl == "direct":
            refuse(self.name, cfg.abl, "direct has no transition for TSAR_WM_TSWM_TRANS=ssm to replace")
        self.read = os.environ.get("TSAR_WM_TSWM_READ", "all")
        if self.read not in ("all", "first"):
            raise SystemExit(f"TSAR_WM_TSWM_READ={self.read!r} not in (all, first)")
        hd = D // H
        rope_dim = 2 * (hd // 4)                                         # rope_partial_factor 0.5, even
        if rope_dim == 0:
            raise SystemExit(f"tswm: head_dim {hd} too small for partial RoPE")
        inv = torch.zeros(hd // 2)
        idx = torch.arange(0, rope_dim, 2, dtype=torch.float32)
        inv[:rope_dim // 2] = 1.0 / (10000.0 ** (idx / rope_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        hidden = 8 * round(8 / 3 * D / 8)
        self.embed = ResidualBlock(cfg.tok_dim, 4 * D, D, bias=True)
        self.forecast_query = nn.Parameter(torch.zeros(D))
        self.layers = nn.ModuleList(Block(D, H, hidden, variate=(i % 4 == 3)) for i in range(cfg.n_layers))
        self.out_norm = RMSNorm(D)
        self.head = ResidualBlock(D, 4 * D, P * NQ, bias=False)
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.trans = None
        if cfg.abl != "direct" and self.trans_kind == "ssm":
            self.trans = LeadSSM(D, int(os.environ.get("TSAR_WM_TSWM_SSM_STATE", max(1, D // 3))))
        elif cfg.abl != "direct":
            self.trans = nn.Linear(D, D)                                 # the world-model transition feedback
            nn.init.zeros_(self.trans.weight); nn.init.zeros_(self.trans.bias)
        self.lat_pred = nn.Linear(D, D) if _LAT > 0 else None

    def param_groups(self):
        g = {"backbone": self.layers, "head": self.head}
        g["transition"] = self.trans if self.trans is not None else nn.Identity()
        return g

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _rope(self, pos):
        """pos [...] (float) -> cos, sin [..., 1, hd] (sin/cos concatenated halves, layers.py:58)."""
        f = pos[..., None] * self.inv_freq
        e = torch.cat([f, f], -1)[..., None, :]
        return e.cos(), e.sin()

    # ---- encoder ---------------------------------------------------------------------------------------------
    def encode(self, ctx):
        x = self.embed(ctx.tok)                                          # [B, n, D]
        B, n, _ = x.shape
        valid = ctx.cnt > 0                                              # started (attend_missing_patches)
        ar = torch.arange(n, device=x.device)
        mask = ((ar[None, :] <= ar[:, None])[None] & valid[:, None, :]) | torch.eye(n, dtype=torch.bool,
                                                                                    device=x.device)[None]
        cos, sin = self._rope(ar.float())                                # [n, 1, hd]
        sc = 1.0 / math.sqrt(self.layers[0].attn.hd)
        kv = []
        for blk in self.layers:
            xn = blk.attn_norm(x)
            q, k, v = blk.attn.qkv(xn, cos, sin)                         # [B, n, H, hd]
            kv.append((k, v))
            s = torch.einsum("bihd,bshd->bhis", q, k) * sc
            a = s.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
            x = x + blk.attn.out(torch.einsum("bhis,bshd->bihd", a, v), xn)
            x = blk.tail(x)
        return {"kv": kv, "valid": valid, "o": self.out_norm(x)}

    # ---- one decoder layer on lead tokens ----------------------------------------------------------------------
    def _dec_layer(self, blk, x, cos, sin, Kt, Vh, emask, cache):
        """x [G, M, T, D] lead tokens. Kt [G, H, hd, n] / Vh [G, H, n, hd]: the encoder keys/values of each group,
        laid out once per decode so the K sequential steps never copy them. emask [G, 1, M*T, n] valid encoder keys
        per query. cache: None (the T tokens are all the leads: causal among them) or a list [K_prev, V_prev]
        ([G, M, H, U, hd]) of earlier imagined tokens (sequential decode; extended in place)."""
        G, M, T, _ = x.shape
        H, hd = blk.attn.h, blk.attn.hd
        xn = blk.attn_norm(x)
        q, k, v = blk.attn.qkv(xn, cos, sin)                             # [G, M, T, H, hd]
        q2, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)   # [G, M, H, T, hd]
        if cache is not None:
            if cache[0] is not None:
                k, v = torch.cat([cache[0], k], 3), torch.cat([cache[1], v], 3)
            cache[0], cache[1] = k, v
        sc = 1.0 / math.sqrt(hd)
        qh = q.permute(0, 3, 1, 2, 4).reshape(G, H, M * T, hd)
        se = (qh @ Kt).mul_(sc).masked_fill_(~emask, float("-inf"))      # [G, H, M*T, n]
        n = se.shape[-1]
        se = se.view(G, H, M, T, n).permute(0, 2, 1, 3, 4)               # [G, M, H, T, n]
        sd = (q2 @ k.transpose(-1, -2)).mul_(sc)                         # [G, M, H, T, U]
        if cache is None:
            tri = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            sd = sd.masked_fill(~tri, float("-inf"))
        a = torch.cat([se, sd], -1).softmax(-1)
        ae = a[..., :n].permute(0, 2, 1, 3, 4).reshape(G, H, M * T, n)
        o = (ae @ Vh).view(G, H, M, T, hd).permute(0, 2, 3, 1, 4) + (a[..., n:] @ v).transpose(2, 3)
        return blk.tail(x + blk.attn.out(o, xn))                         # o [G, M, T, H, hd]

    def _run_leads(self, x, pos, kv, emask, cache, read=None):
        """read: None (every token reads the context) or a bool [T]: which of the T lead tokens may read it
        (TSAR_WM_TSWM_READ=first)."""
        cos, sin = self._rope(pos)
        G, M, T, _ = x.shape
        em = emask[:, None, :, None, :].expand(-1, -1, -1, T, -1)
        if read is not None:
            em = em & read[None, None, None, :, None]
        em = em.reshape(G, 1, M * T, emask.shape[-1])
        for li, blk in enumerate(self.layers):
            Kt, Vh = kv[li]
            x = self._dec_layer(blk, x, cos, sin, Kt, Vh, em, None if cache is None else cache[li])
        return self.out_norm(x)

    # ---- decoder -----------------------------------------------------------------------------------------------
    def decode(self, enc, ctx, b, j):
        cfg, abl = self.cfg, self.cfg.abl
        P, K, NQ = cfg.patch, cfg.k_steps, cfg.nq
        kv, valid, o = enc["kv"], enc["valid"], enc["o"]
        B, n, D = o.shape
        N = len(b)
        dense = self.path == "dense" or (self.path == "auto" and 4 * N >= B * n)
        dev = o.device
        if dense:                                                        # every (row, origin) slot of the batch
            M = n
            jj = torch.arange(n, device=dev)[None].expand(B, n)          # [G=B, M=n]
            kvg = [(k_.permute(0, 2, 3, 1).contiguous(), v_.transpose(1, 2).contiguous()) for k_, v_ in kv]
            emask = (torch.arange(n, device=dev)[None, None, :] <= jj[..., None]) & valid[:, None, :]
            ref = ctx.ref.view(B, n, K, P)
            z0 = o                                                       # [B, n, D]
        else:                                                            # one group per requested origin
            M = 1
            jj = j[:, None]
            kvg = [(k_[b].permute(0, 2, 3, 1).contiguous(), v_[b].transpose(1, 2).contiguous()) for k_, v_ in kv]
            emask = ((torch.arange(n, device=dev)[None, :] <= j[:, None]) & valid[b])[:, None, :]
            ref = ctx.ref[b, j].view(N, 1, K, P)
            z0 = o[b, j][:, None]                                        # [N, 1, D]
        lead = torch.arange(K, device=dev)
        pos = (jj[..., None] + 1 + lead).float()                         # [G, M, K]
        U = self.forecast_query + self.drive(ref)                        # [G, M, K, D]
        ssm = self.trans_kind == "ssm"
        if abl in ("direct", "copy"):
            X = U if abl == "direct" else U + (self.trans.step(z0)[0] if ssm else self.trans(z0))[:, :, None]
            rd = (lead == 0) if self.read == "first" else None
            Z = self._run_leads(X, pos, kvg, emask, None, rd)            # parallel DHF decode
        else:                                                            # none / zero: imagined trajectory
            cache = [[None, None] for _ in self.layers]
            z, zs, st = z0, [], None
            for k in range(K):
                if ssm:                                                  # carried SSM state across leads
                    y, st = self.trans.step(z, st)
                else:
                    y = self.trans(z)
                rd = (lead[k:k + 1] == 0) if self.read == "first" else None
                z = self._run_leads((U[:, :, k] + y)[:, :, None], pos[:, :, k:k + 1], kvg, emask,
                                    cache, rd)[:, :, 0]
                zs.append(z)
            Z = torch.stack(zs, 2)                                       # [G, M, K, D]
        if dense:
            sel = b * n + j
            Z = Z.reshape(B * n, K, D)[sel]
        else:
            Z = Z[:, 0]                                                  # [N, K, D]
        aux = {}
        if self.lat_pred is not None and self.training:
            tgt_idx = j[:, None] + 1 + lead                              # encoder patch of lead k
            ok = (tgt_idx < n) & valid[b[:, None], tgt_idx.clamp(max=n - 1)]
            if ok.any():
                tgt = o[b[:, None], tgt_idx.clamp(max=n - 1)].detach()
                err = (self.lat_pred(Z) - tgt).pow(2).mean(-1)
                aux["lat"] = (err * ok).sum() / ok.sum()
            else:
                aux["lat"] = Z.sum() * 0.0
        if abl == "zero":
            Z = torch.zeros_like(Z)
        q = monotone_quantiles(self.head(Z).view(N, K, P, NQ), self.wf)
        return self._to_origin_frame(q, ctx, b, j), aux

    def _to_origin_frame(self, q, ctx, b, j):
        """Lead k's quantiles are read in the advancing frame of the imagined path, then mapped to the origin frame
        (model space of (b, j)), where the harness scores them."""
        if self.frame == "origin" or q.shape[1] < 2:
            return q
        N, K, P, NQ = q.shape
        med = torch.sinh(q[..., NQ // 2].detach().clamp(-15, 15))       # [N, K, P] in lead-frame units
        if self.frame == "loc":                                          # scale fixed: l_k = sum_{i<k} med_i[-1]
            L = F.pad(torch.cumsum(med[:, :-1, -1], 1), (1, 0))[..., None, None]
            return torch.cat([q[:, :1], torch.asinh(torch.sinh(q[:, 1:].clamp(-15, 15)) + L[:, 1:])], 1)
        loc = q.new_zeros(N)
        scl = q.new_ones(N)
        last = q.new_zeros(N)                                            # the last observation = loc0 = 0
        c0 = (ctx.cnt[b, j] - 1).clamp(min=1)
        tot = c0.clone()                                                 # sum |step| so far, origin-scale units
        locs, scls = [], []
        for k in range(K):
            locs.append(loc); scls.append(scl)
            y = med[:, k] * scl[:, None] + loc[:, None]                  # imagined median, origin units
            if self.frame == "adv":
                steps = torch.diff(torch.cat([last[:, None], y], 1), dim=1).abs().sum(1)
                tot = tot + steps
                scl = (tot / (c0 + P * (k + 1))).clamp(0.5, 1e6)
            last = y[:, -1]
            loc = last
        L = torch.stack(locs, 1)[..., None, None]
        S = torch.stack(scls, 1)[..., None, None]
        out = torch.asinh(torch.sinh(q.clamp(-15, 15)) * S + L)
        return torch.cat([q[:, :1], out[:, 1:]], 1)                      # lead 0 is in the origin frame exactly
