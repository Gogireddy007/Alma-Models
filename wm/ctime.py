"""
Family `ctime`: continuous-time latent dynamics. An ODE-RNN filter over the observed patches and a closed-form
neural-ODE rollout over the forecast, where every forecast step is the latent state at its actual time offset.

The vector field. For a state x [D] and a piecewise-constant input u, the field evaluated at the start of an
interval gives, per pair of latent dims (D/2 damped oscillators),
    decay rate  lam = softplus(.)          learned, state- and input-dependent time constants (LTC-style)
    frequency   om  = pi * tanh(.)         rad per step, up to Nyquist (ROT=0 sets om = 0: pure relaxation)
    equilibrium x_eq = tanh(.)
from one hidden layer on [LN(x), u]. With these coefficients frozen over the interval, dx/dt = A (x - x_eq) with
A = blockdiag([[-lam, -om], [om, -lam]]) has the exact solution
    x(t0 + tau) = x_eq + exp(-lam tau) R(om tau) (x(t0) - x_eq)
so the state can be evaluated at any real time offset in closed form (an exponential integrator; no ODE solver, no
step-size error inside an interval). The coefficients are re-evaluated at every patch boundary, so across patches
the dynamics are nonlinear. The state stays bounded: |x - x_eq| only decays and rotates, and |x_eq| < 1.

  encode  e_t = causal Transformer(tokens)_t (N_LAYERS layers, the same layer as gru). ODE-RNN filter, causal:
              x_t^- = flow(x_{t-1}^+, u = drive(seasonal-naive patch t from origin t-1), tau = PATCH)
              x_t^+ = GRUCell(e_t, x_t^-)                   observation update (jump)
          The SAME field that rolls the forecast propagates the belief between observations, so it is trained on
          every observed transition, not only on the forecast leads.
  decode  from x_j^+, lead patch k uses input u_k = drive(ref_k) and the state at every step of the patch:
              x(kP + tau) = flow(x(kP), field(x(kP), u_k), tau),   tau = 1..PATCH
          x((k+1)P) is carried to lead k+1. Each forecast step is decoded from its own state:
              q(t) = Linear(LN(x(t) + ref_proj(ref_k) + ref_pt(ref(t))))  -> NQ   (one head for every step)
          so the within-patch shape comes from the continuous trajectory (sub-patch interpolation), not from a
          PATCH*NQ linear map. There is no step embedding: the lead is identified by its time only.

Ablations (TSAR_WM_ABL). The field is shared by the filter (encoder) and the rollout, so no mode deletes it:
  none    the rollout above, coefficients re-evaluated from the carried state at each patch boundary.
  direct  no carried state: lead k jumps straight from the origin, x(kP + tau) = flow(x_j, field(x_j, u_k), kP + tau).
          Same parameters (0% difference) and the same MACs (K field evaluations); the lead's state depends on the
          origin and its own input only, never on lead k-1.
  copy    no transition in the rollout: x(kP + tau) = x_j + drive(ref_k) for every tau (constant within the patch;
          the per-step seasonal input ref_pt and the harness skip remain). The filter still uses the field, so the
          parameter count equals none (reported, not reduced).
  zero    rolled states zeroed before the decoder; identical parameters (eval-only on a none checkpoint).

Env keys (all optional):
  TSAR_WM_CTIME_ROT   1 = damped oscillators (default), 0 = pure relaxation (real time constants only, LTC-like)
  TSAR_WM_CTIME_HID   field hidden width as a multiple of d_model (2.0)

Width ladder: every module scales like d^2, so the d_model ladder of docs/wm_families.md applies.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily

_e = os.environ.get
ROT = int(_e("TSAR_WM_CTIME_ROT", 1))
HID = float(_e("TSAR_WM_CTIME_HID", 2.0))


class CTimeFamily(WMFamily):
    name = "ctime"
    aux_weights = {}
    env_keys = ("TSAR_WM_CTIME_ROT", "TSAR_WM_CTIME_HID")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P, NQ = cfg.d_model, cfg.patch, cfg.nq
        if D % 2:
            raise SystemExit("ctime needs an even d_model (pairs of oscillator dims)")
        H = max(8, 8 * round(HID * D / 8))
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        # vector field (shared by the filter and the rollout)
        self.f_norm = nn.LayerNorm(D)
        self.f_in = nn.Linear(2 * D, H)
        self.f_out = nn.Linear(H, 2 * D)          # [rate (D/2) | freq (D/2) | equilibrium (D)], one matmul
        with torch.no_grad():                    # spread the initial time constants and frequencies (S4D-like)
            h2 = D // 2
            rate = torch.logspace(math.log10(1 / 128), 0, h2)                  # per step: 128 steps .. 1 step
            om = torch.linspace(0, 0.5, h2)                                     # om / pi at init
            self.f_out.bias[:h2].copy_(torch.log(torch.expm1(rate)))
            self.f_out.bias[h2:D].copy_(torch.atanh(om))
            self.f_out.weight[:D].mul_(0.1)
        self.jump = nn.GRUCell(D, D)
        self.drive = nn.Linear(P, D)
        self.ref_proj = nn.Linear(P, D)
        self.ref_pt = nn.Linear(1, D)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, NQ))
        self.register_buffer("tau", torch.arange(1, P + 1, dtype=torch.float32))

    def param_groups(self):
        return {"transition": nn.ModuleList([self.f_norm, self.f_in, self.f_out]),
                "jump": self.jump}

    def field(self, x, u):
        h2 = x.shape[-1] // 2
        o = self.f_out(F.gelu(self.f_in(torch.cat([self.f_norm(x), u], -1))))
        lam = F.softplus(o[:, :h2]) + 1e-4
        om = math.pi * torch.tanh(o[:, h2:2 * h2]) * ROT
        return lam, om, torch.tanh(o[:, 2 * h2:])

    @staticmethod
    def flow(x, lam, om, xeq, t):
        """Exact solution of the frozen linear field at times t from x. x, xeq [..., D]; lam, om [..., D/2]; t [T] or
        any shape broadcastable to [..., T]. Returns [..., T, D]. Oscillator pairs are (i, i + D/2)."""
        h2 = x.shape[-1] // 2
        t = t[..., None]                                                   # [..., T, 1]
        d = (x - xeq)[..., None, :]
        a, b = d[..., :h2], d[..., h2:]
        g = torch.exp(-lam[..., None, :] * t)                              # [..., T, D/2]
        ph = om[..., None, :] * t
        gc, gs = g * torch.cos(ph), g * torch.sin(ph)
        return xeq[..., None, :] + torch.cat([gc * a - gs * b, gs * a + gc * b], -1)

    def encode(self, ctx):
        P = self.cfg.patch
        B, n = ctx.tok.shape[:2]
        e = self.encoder(self.embed(ctx.tok) + self.pos[:, :n], mask=self.causal[:n, :n])   # [B, n, D]
        u = self.drive(ctx.ref[:, :, :P])                                  # input over patch t+1 from origin t
        tP = self.tau[-1:]
        x = e.new_zeros(B, self.cfg.d_model)
        xs = []
        for t in range(n):
            if t > 0:
                x = self.flow(x, *self.field(x, u[:, t - 1]), tP)[:, 0]
            x = self.jump(e[:, t], x)
            xs.append(x)
        return torch.stack(xs, 1)                                          # [B, n, D]

    def decode(self, enc, ctx, b, j):
        cfg, abl = self.cfg, self.cfg.abl
        P, K, N = cfg.patch, cfg.k_steps, len(b)
        x0 = x = enc[b, j]
        ref = ctx.ref[b, j]                                                # [N, MAX_H]
        tP = self.tau[-1:]
        if abl == "copy":
            S = (x0[:, None] + self.drive(ref.view(N, K, P)))[:, :, None].expand(N, K, P, -1)
        else:
            starts, coefs = [], []
            for k in range(K):
                u = self.drive(ref[:, k * P:(k + 1) * P])
                c = self.field(x0 if abl == "direct" else x, u)
                starts.append(x); coefs.append(c)
                if abl != "direct":
                    x = self.flow(x, *c, tP)[:, 0]                         # carried boundary state x((k+1)P)
            lam, om, xeq = (torch.stack(v, 1) for v in zip(*coefs))        # [N, K, .]
            if abl == "direct":                                            # jump from the origin to t = kP + tau
                t = torch.arange(K, device=x0.device)[:, None] * P + self.tau  # [K, P]
                S = self.flow(x0[:, None].expand(N, K, -1), lam, om, xeq, t)
            else:                                                          # every step of every rolled patch
                S = self.flow(torch.stack(starts, 1), lam, om, xeq, self.tau)
        if abl == "zero":
            S = torch.zeros_like(S)
        r = ref.view(N, K, P)
        S = S + self.ref_proj(r)[:, :, None] + self.ref_pt(r[..., None])
        return self.head(S), {}
