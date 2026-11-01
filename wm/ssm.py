"""
Family `ssm`: a selective state-space world model. One dynamical system both ASSIMILATES the observed patches and
IMAGINES the future ones: the SSM backbone that encodes the context is the same recurrence that is continued, from
its own state at the origin, through the K lead patches, driven by the lead inputs.

Why it is a world model and not a GRU tweak: there is no Transformer and no separate transition module. The world
state is the stack of SSM states (one per layer) plus each layer's causal-conv buffer; observed patch tokens and
imagined lead tokens update that state with the same parameters and the same equations. Forecasting = keep
running the filter with no observations, only actions (lead step embedding + the seasonal-naive patch).

Mixer (per layer, Mamba/S6-style in pure PyTorch, no custom kernels; MPS-friendly real arithmetic):
  r -> LN -> in_proj (D -> 2E) = [x, gate]; x -> causal depthwise conv over patches (kernel TSAR_WM_SSM_CONV) -> SiLU
  selective step (input-dependent timestep per mode, S6-style), MIMO diagonal complex state of P modes (S5-style,
  so the per-origin state is 2P reals instead of Mamba's E*N, which is what makes rolling 5k origins affordable):
      dt_t = softplus(W_dt x_t + b_dt)                         [P]
      lambda_t = exp(dt_t * a),  a = -exp(a_log) + i a_im       |lambda_t| < 1 always (stable)
      h_t = lambda_t * h_{t-1} + dt_t * (B x_t)                 B: E -> P complex
      y_t = C [Re h_t, Im h_t] + Dskip * x_t                    C: 2P -> E
  out = out_proj(y * SiLU(gate)) (E -> D), residual r + out. Final LN -> o_t (the decoder's view of the state).
  Init: a_im gives periods log-spaced over 2..64 patches at dt = 1; radii |exp(a)| over 0.6..0.995 per patch.

  encode  embed tokens; N_LAYERS mixers scanned over all patches (sequential scan of n <= 40 steps). Returns, per
          layer, the state h_j and the conv buffer (the last CONV-1 pre-conv inputs) at every patch j, and o_j.
  decode  imagined token for lead k: u_k = step_emb[k] + drive(ref_k). Starting from (h_j, conv buffer at j) in
          every layer, run the SAME mixers K steps on u_0..u_{K-1}; decoder head(o_k + ref_proj(ref_k)), head =
          Linear(D, D), GELU, Linear(D, PATCH*NQ). The harness adds the per-quantile ref skip.

Ablations (TSAR_WM_ABL):
  none    the rollout above (state and conv buffer carried from lead k to k+1).
  direct  no carried state: every lead takes ONE step of the same system from the origin's state and conv buffer
          with its own input u_k (h_k = lambda(u_k) h_j + dt(u_k) B u_k; the selective dt lets step_emb[k] stretch
          the step to k patches, so this is a learned direct jump). Identical parameters and MACs to none.
  copy    no transition: o_k = o_j + u_k. The SSM is shared with the encoder, so no parameter can be removed
          (params identical to none; reported as such).
  zero    imagined o_k zeroed before the decoder (same parameters as none).

Env keys: TSAR_WM_SSM_EXPAND (inner width E = 8 * round(EXPAND * d_model / 8), default 2.25), TSAR_WM_SSM_STATE
(complex modes P per layer, default d_model), TSAR_WM_SSM_CONV (causal conv kernel over patches, default 4; 1 = none).
Params at width 1 (d_model 96, E 216, P 96): 365,937 in every mode (including ref_skip). Params scale ~d^2.
MACs: torch's counter sees only the matmuls; the scan's elementwise complex updates (about 10 * P per step per layer,
~0.1M per forecast) are not counted.

Round-2 architect options (2026-10-31, runs/proposals/architect_r2.md). Each is default-off; with all three off the
module builds the same parameters in the same order and runs the same ops as before (bit-identical). New modules
are created after every existing one, so turning one on does not change the init of the others.
  TSAR_WM_SSM_SPEC=0|1  system identification for imagination. A small head on the origin state o_j (Linear(D, 16),
          GELU, Linear(16, n_layers * 2P), last layer zero-init) shifts every layer's spectrum for the rollout of
          that origin: a_log += dlog, angle *= exp(0.5 tanh(dang)). The encoder scan is unchanged; the imagined
          continuation runs the same system with per-series damping and frequencies (Koopman's MOD, which was worth
          0.016 in that family). |lambda| < 1 still holds. Used by none, zero and direct (one step), not copy.
          +8,080 params at width 1.
  TSAR_WM_SSM_FB=0|1    closed-loop imagination. The rollout's input for lead k >= 1 also gets fb(q_{k-1}), an
          embedding (Linear(PATCH*NQ, D), zero-init) of the model's own quantile forecast for lead k-1 (clamped to
          +-20, gradients flow through it). The decoder head is applied inside the loop. none / zero only; direct
          and copy have no previous lead and ignore it (the parameters still exist, so counts match).
          +7,008 params at width 1.
  TSAR_WM_SSM_LOOP=1|2|3  weight-tied depth. The mixer stack is applied LOOP times with shared weights; every pass
          has its own SSM state and conv buffer, so the world state grows LOOP x at the same parameters. Pass p >= 1
          adds a learned pass embedding (zero-init) to the residual stream first. Applies to encode and to every
          ablation. +(LOOP-1)*D params; about LOOP x the step time and activation memory.

Period lever (2026-10-31, docs/period_lever.md). Harness keys, default off; built after every module above and
zero-init, so with both off the module is bit-identical and with them on no other init changes.
  TSAR_PERIOD=1      the harness detects up to 4 periods at every origin (detect_periods below) and passes phase
          coordinates for the context patches (ctx.tok_phase) and for each origin's lead patches (ctx.fut_phase).
          One shared zero-init Linear(16, D) embeds both: added to the token embedding in encode and to every lead
          input u_k in decode (none, direct, copy, zero alike), so the latent rollout runs on a clock that is
          continuous with the context. +1,632 params at width 1.
  TSAR_PERIOD_BIN=1  phase binning (template() below): a per-origin cycle template of encoder outputs in 16 phase bins
          per period, read by each lead at its own phase, into u_k (tmpl_in) and the decoder head input (tmpl_out),
          both zero-init, every ablation. +18,624 params at width 1.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily

# ---------------------------------------------------------------------------------------------------------------
# Period lever (2026-10-31, docs/period_lever.md): parameter-free period detection + phase coordinates. The harness
# (train_wm.py, TSAR_PERIOD=1) calls detect_periods / phase_features on the window ending at every origin; the
# family reads ctx.tok_phase / ctx.fut_phase (TSAR_PERIOD) and ctx.periods (TSAR_PERIOD_BIN). Pure functions, kept
# here so tools can unit-test them without running the training script.
# ---------------------------------------------------------------------------------------------------------------
PER_LW = 256           # detection window in points (= the default inference CONTEXT, so train and eval match)
PER_K = 4              # at most this many periods per origin
PER_FEAT = 4           # features per period slot: sin, cos, active, active * log(p) / log(PER_LW)
PER_BINS = 16          # phase bins per period for the cycle template (TSAR_PERIOD_BIN)
_DFT_CACHE = {}


def _dft(L, device):
    key = (L, str(device))
    if key not in _DFT_CACHE:
        t = torch.arange(L, dtype=torch.float64)
        k = torch.arange(1, L // 2 + 1, dtype=torch.float64)
        ang = 2 * math.pi * t[:, None] * k[None, :] / L
        _DFT_CACHE[key] = torch.cat([torch.cos(ang), torch.sin(ang)], 1).float().to(device)   # [L, 2m]
    return _DFT_CACHE[key]


@torch.no_grad()
def detect_periods(w, K=PER_K, alpha=0.01, min_cycles=3.0, min_obs=64, phi_max=0.99, upgrade=0.2, coherence=0.2):
    """Parameter-free period detector (TinyCast / Fracast-0 style) with an AR(1) null and a coherence check.

    w: [M, L] raw windows, NaN = missing or padding, oldest first. Returns [M, K] integer-valued periods in points,
    strongest first, 0 = empty slot. Per row, observed points only:
      1. centre, scale, weighted linear detrend, standardise, clip at +-5 sd (spikes), zero the unobserved points;
      2. lag-1 autocorrelation phi (clamped to +-phi_max) = the AR(1) null; periodogram I(k), k = 1..L/2, by one
         DFT matmul; whitened periodogram Iw(k) = I(k) * |1 - phi exp(-i 2 pi k / L)|^2 (I / AR(1) spectrum);
      3. Fisher's g-test on Iw, sequentially: g = max Iw / sum Iw over the remaining bins, p ~= m (1 - g)^(m - 1)
         with m the effective number of bins (observed points / 2, scaled by the share of bins left), accepted
         while p < alpha / K (Bonferroni over the K slots) and the row has >= min_obs observed points; the first
         non-significant peak stops the row. Only bins whose period gives >= min_cycles cycles in the observed
         span are candidates (the minimum-cycle null); the peak and its two neighbours leave the pool;
      4. integer period from the comb autocorrelation C(P) = mean of a(P), a(2P), .., a(iP) over every multiple
         of P up to lag L/2 (a = pair-count normalised autocorrelation of the cleaned series): the P in the peak
         bin's period range [L/(k+1), L/(k-1)] with the highest C (long lags make it sharp to +-1 point), then
         moved to a multiple mP (m = 2..8) when C(mP) > C(P) + upgrade, C(mP) > 2.5 C(P) and C(P) >= coherence / 4 (a non-sinusoidal profile whose strongest
         line is a harmonic: phase at mP still fixes phase at P);
      5. coherence: C(P) >= coherence, the peak bin's centre period is admissible, and the best P is not in the top 10% of the
         clipped end of a bin range that extends past the longest admissible period (a true period beyond the
         min-cycle limit, e.g. 96 in a 256 window, would land there with a wrong value). A fixed-phase cycle keeps its autocorrelation at every multiple, an AR(2)
         quasi-cycle (random phase drift) does not. A period failing 5, or a duplicate or integer fraction
         (harmonic) of an accepted one, takes no slot and the search goes on (2K iterations to fill K slots).
    A red-noise (AR(1)) or random-walk series has a flat whitened periodogram, so it fires at about rate alpha."""
    M, L = w.shape
    dev = w.device
    w = w.float()
    obs = ~torch.isnan(w)
    of = obs.float()
    n = of.sum(1)
    nn_ = n.clamp(min=1)
    v = torch.where(obs, w, torch.zeros_like(w))
    v = (v - (v.sum(1) / nn_)[:, None]) * of
    v = v / v.abs().amax(1, keepdim=True).clamp(min=1e-30)
    t = torch.arange(L, device=dev, dtype=torch.float32)
    mt = (of * t).sum(1) / nn_
    mv = v.sum(1) / nn_
    tc = (t - mt[:, None]) * of
    slope = (tc * v).sum(1) / (tc * tc).sum(1).clamp(min=1e-6)
    r = (v - mv[:, None] - slope[:, None] * (t - mt[:, None])) * of
    sd = ((r * r).sum(1) / nn_).sqrt()
    r = (r / sd.clamp(min=1e-12)[:, None]).clamp(-5, 5) * of
    pair = of[:, 1:] * of[:, :-1]
    phi = ((r[:, 1:] * r[:, :-1] * pair).sum(1) / (r * r).sum(1).clamp(min=1e-12)).clamp(-phi_max, phi_max)
    Fq = r @ _dft(L, dev)
    m = L // 2
    I = Fq[:, :m] ** 2 + Fq[:, m:] ** 2                                              # [M, m], bin k = index + 1
    kk = torch.arange(1, m + 1, device=dev, dtype=torch.float32)
    om = 2 * math.pi * kk / L
    Iw = I * (1 + phi[:, None] ** 2 - 2 * phi[:, None] * torch.cos(om)[None]) + 1e-12
    # autocorrelation at lags 0..m, normalised by the number of observed pairs at each lag
    Rr = torch.fft.irfft(torch.fft.rfft(r, n=2 * L).abs() ** 2, n=2 * L)[:, :m + 1]
    Rc = torch.fft.irfft(torch.fft.rfft(of, n=2 * L).abs() ** 2, n=2 * L)[:, :m + 1]
    acf = Rr / Rc.clamp(min=8)
    acf = torch.where(Rc >= 8, acf / acf[:, :1].clamp(min=1e-12), torch.full_like(acf, -1.0))   # [M, m + 1]
    lag = torch.arange(m + 1, device=dev, dtype=torch.float32)
    li = torch.arange(m + 1, device=dev)
    Wc = ((li[:, None] % li[None].clamp(min=1) == 0) & (li[None] >= 2) & (li[:, None] >= 1)).float()   # [lag, P]
    val = (acf > -1).float()
    comb = ((acf * val) @ Wc) / (val @ Wc).clamp(min=1)                              # [M, m + 1] C(P)
    comb = torch.where((val @ Wc) > 0, comb, torch.full_like(comb, -9.0))
    pmax = torch.minimum(n / min_cycles, torch.full_like(n, float(m)))               # longest admissible period
    rem = torch.ones_like(Iw, dtype=torch.bool)
    cand = (n[:, None] * kk[None] / L >= min_cycles) & (L / kk[None] >= 2)
    periods = torch.zeros(M, K, device=dev)
    cnt = torch.zeros(M, dtype=torch.long, device=dev)
    alive = n >= min_obs
    logthr = math.log(alpha / K)
    ar = torch.arange(m, device=dev)
    mult = torch.arange(2, 9, device=dev, dtype=torch.float32)
    rows = torch.arange(M, device=dev)
    for _ in range(2 * K):
        tot = (Iw * rem).sum(1)
        pk, ik = torch.where(cand & rem, Iw, torch.zeros_like(Iw)).max(1)
        g = (pk / tot.clamp(min=1e-30)).clamp(max=1 - 1e-7)
        meff = (0.5 * n * rem.float().mean(1)).clamp(min=2)
        logp = torch.log(meff) + (meff - 1) * torch.log1p(-g)
        sig = alive & (pk > 0) & (logp < logthr)
        k = ik.float() + 1
        lo = torch.floor(L / (k + 1)).clamp(min=2)
        hi0 = torch.where(k > 1, torch.ceil(L / (k - 1).clamp(min=1)), torch.full_like(k, L))
        hi = torch.minimum(hi0, pmax)
        inr = (lag[None] >= lo[:, None]) & (lag[None] <= hi[:, None])
        aP, Pi = torch.where(inr, comb, torch.full_like(comb, -9.0)).max(1)
        P = Pi.float()
        edge = ((P >= 0.9 * hi) & (hi0 > pmax)) | (L / k > pmax)   # true period may lie past the min-cycle limit
        mP = P[:, None] * mult[None]
        okm = mP <= pmax[:, None]
        am = torch.where(okm, comb.gather(1, mP.long().clamp(max=m)), torch.full_like(mP, -9.0))
        abest, mi = am.max(1)
        up = (abest > aP + upgrade) & (abest > 2.5 * aP) & (aP >= 0.25 * coherence)
        P = torch.where(up, mP[rows, mi], P)
        aP = torch.where(up, abest, aP)
        coh = (aP >= coherence) & (P >= 2) & ~edge
        dup = ((periods - P[:, None]).abs() <= torch.clamp(0.05 * P[:, None], min=1.0)) & (periods > 0)
        ratio = periods / P[:, None]
        harm = (ratio >= 1.5) & ((ratio - torch.round(ratio)).abs() < 0.1) & (periods > 0)
        new = sig & (cnt < K) & coh & ~(dup | harm).any(1)
        slot = F.one_hot(cnt.clamp(max=K - 1), K).bool() & new[:, None]
        periods = torch.where(slot, P[:, None].expand(-1, K), periods)
        cnt = cnt + new.long()
        alive = sig & (cnt < K)
        rem = rem & ~(((ar[None] - ik[:, None]).abs() <= 1) & sig[:, None])
    return periods


def phase_features(periods, pos):
    """periods [..., K] (0 = empty), pos [...] start point index (broadcast) -> [..., K * PER_FEAT] features:
    per slot sin and cos of 2 pi (pos mod p) / p, active, active * log(p) / log(PER_LW); all zero when empty."""
    act = (periods > 0).float()
    p = periods.clamp(min=1)
    ph = 2 * math.pi * torch.remainder(pos[..., None].float(), p) / p
    f = torch.stack([torch.sin(ph) * act, torch.cos(ph) * act, act, act * torch.log(p) / math.log(PER_LW)], -1)
    return f.flatten(-2)


class Mixer(nn.Module):
    def __init__(self, D, E, P, kc):
        super().__init__()
        self.E, self.P, self.kc = E, P, kc
        self.norm = nn.LayerNorm(D)
        self.in_proj = nn.Linear(D, 2 * E, bias=False)
        self.conv_w = nn.Parameter(torch.randn(E, kc) / math.sqrt(kc))
        self.conv_b = nn.Parameter(torch.zeros(E))
        self.BC = nn.Linear(E, 3 * P)                                    # [dt | B re | B im], one matmul
        self.C = nn.Linear(2 * P, E, bias=False)
        with torch.no_grad():
            self.BC.weight[:P].normal_(std=0.02)
            self.BC.bias.zero_()
            self.BC.bias[:P] = math.log(math.e - 1)                     # softplus(dt) = 1 patch at init
        radius = torch.linspace(0.6, 0.995, P)[torch.randperm(P, generator=torch.Generator().manual_seed(1))]
        period = torch.exp(torch.linspace(math.log(2.0), math.log(64.0), P))
        self.a_log = nn.Parameter(torch.log(-torch.log(radius)))
        self.a_im = nn.Parameter(2 * math.pi / period)
        self.Dskip = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, D, bias=False)

    def pre(self, r):
        """Residual stream [..., D] -> pre-conv input x and gate, [..., E] each."""
        return self.in_proj(self.norm(r)).chunk(2, -1)

    def conv(self, win):
        """win [..., kc, E] (oldest first) -> causal conv output at the newest position, SiLU'd."""
        return F.silu((win * self.conv_w.t()).sum(-2) + self.conv_b)

    def coeffs(self, xc, mod=None):
        """Selective step coefficients for input xc [..., E]: lambda (re, im) and the input term (re, im), [..., P].
        mod: optional per-origin (decay rate, angular rate) [..., P] replacing (exp(a_log), a_im) (TSAR_WM_SSM_SPEC,
        from spectrum()), broadcast against xc's batch dims."""
        P = self.P
        g = self.BC(xc)
        dt = F.softplus(g[..., :P])
        if mod is None:
            mag = torch.exp(-dt * torch.exp(self.a_log))
            ang = dt * self.a_im
        else:
            mag = torch.exp(-dt * mod[0])
            ang = dt * mod[1]
        return mag * torch.cos(ang), mag * torch.sin(ang), dt * g[..., P:2 * P], dt * g[..., 2 * P:]

    def post(self, hr, hi, xc, gate):
        y = self.C(torch.cat([hr, hi], -1)) + self.Dskip * xc
        return self.out_proj(y * F.silu(gate))

    def step(self, r, hr, hi, buf, mod=None):
        """One step from state (hr, hi) and conv buffer buf [..., kc-1, E] on residual input r [..., D]."""
        x, gate = self.pre(r)
        win = torch.cat([buf, x.unsqueeze(-2)], -2) if self.kc > 1 else x.unsqueeze(-2)
        xc = self.conv(win)
        lr, li, vr, vi = self.coeffs(xc, mod)
        hr, hi = lr * hr - li * hi + vr, lr * hi + li * hr + vi
        return r + self.post(hr, hi, xc, gate), hr, hi, win[..., 1:, :]


class SSMFamily(WMFamily):
    name = "ssm"
    aux_weights = {}
    env_keys = ("TSAR_WM_SSM_EXPAND", "TSAR_WM_SSM_STATE", "TSAR_WM_SSM_CONV",
                "TSAR_WM_SSM_SPEC", "TSAR_WM_SSM_FB", "TSAR_WM_SSM_LOOP")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P, NQ, K = cfg.d_model, cfg.patch, cfg.nq, cfg.k_steps
        E = max(8, 8 * round(float(os.environ.get("TSAR_WM_SSM_EXPAND", "2.25")) * D / 8))
        S = int(os.environ.get("TSAR_WM_SSM_STATE", D))
        self.kc = int(os.environ.get("TSAR_WM_SSM_CONV", "4"))
        if self.kc < 1:
            raise SystemExit("TSAR_WM_SSM_CONV must be >= 1")
        self.spec = int(os.environ.get("TSAR_WM_SSM_SPEC", "0"))
        self.fb_on = int(os.environ.get("TSAR_WM_SSM_FB", "0"))
        self.loop = int(os.environ.get("TSAR_WM_SSM_LOOP", "1"))
        if self.spec not in (0, 1) or self.fb_on not in (0, 1) or self.loop not in (1, 2, 3):
            raise SystemExit("TSAR_WM_SSM_SPEC and TSAR_WM_SSM_FB must be 0|1, TSAR_WM_SSM_LOOP 1|2|3")
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.mixers = nn.ModuleList(Mixer(D, E, S, self.kc) for _ in range(cfg.n_layers))
        self.norm_f = nn.LayerNorm(D)
        self.step_emb = nn.Parameter(torch.zeros(K, D))
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.decode_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, P * NQ))
        self.ref_proj = nn.Linear(P, D)
        # round-2 options: built only when on, and after every module above (defaults stay bit-identical)
        self.S = S
        if self.spec:
            self.spec_head = nn.Sequential(nn.Linear(D, 16), nn.GELU(), nn.Linear(16, len(self.mixers) * 2 * S))
            nn.init.zeros_(self.spec_head[-1].weight); nn.init.zeros_(self.spec_head[-1].bias)
        if self.fb_on:
            self.fb = nn.Linear(P * NQ, D)
            nn.init.zeros_(self.fb.weight); nn.init.zeros_(self.fb.bias)
        if self.loop > 1:
            self.pass_emb = nn.Parameter(torch.zeros(self.loop - 1, D))
        # period lever (harness keys TSAR_PERIOD / TSAR_PERIOD_BIN, read and validated by train_wm.py; the features
        # arrive on the Context). Built last and zero-init, so the off path and every other init are unchanged.
        self.period = int(os.environ.get("TSAR_PERIOD", "0"))
        self.pbin = int(os.environ.get("TSAR_PERIOD_BIN", "0"))
        if self.period:
            self.ph_emb = nn.Linear(PER_K * PER_FEAT, D)                 # one clock for context and leads
            nn.init.zeros_(self.ph_emb.weight); nn.init.zeros_(self.ph_emb.bias)
        if self.pbin:
            self.tmpl_in = nn.Linear(D, D)                               # template -> rollout input
            self.tmpl_out = nn.Linear(D, D)                              # template -> decoder head input
            for lin in (self.tmpl_in, self.tmpl_out):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def param_groups(self):
        return {"ssm_shared_enc_and_rollout": self.mixers}

    def stack(self):
        """Effective layers: (mixer, pass embedding to add first or None). LOOP=1 -> the mixers once, as before."""
        L = len(self.mixers)
        return [(self.mixers[i % L], self.pass_emb[i // L - 1] if (i % L == 0 and i >= L) else None)
                for i in range(L * self.loop)]

    def spectrum_mods(self, o0):
        """TSAR_WM_SSM_SPEC: per-origin spectrum for every effective layer, computed once per origin (not per lead):
        decay rate exp(a_log + dlog) and angular rate a_im * exp(0.5 tanh(dang)), [N, P] each; None when off."""
        if not self.spec:
            return [None] * (len(self.mixers) * self.loop)
        d = self.spec_head(o0).view(o0.shape[0], len(self.mixers), 2, self.S)
        per_mixer = [(torch.exp(mx.a_log + d[:, i, 0]), mx.a_im * torch.exp(0.5 * torch.tanh(d[:, i, 1])))
                     for i, mx in enumerate(self.mixers)]
        return [per_mixer[i % len(self.mixers)] for i in range(len(self.mixers) * self.loop)]

    def template(self, o, ctx, b, j):
        """TSAR_PERIOD_BIN: per origin, the encoder outputs o_t of the last CONTEXT/PATCH patches t <= j (weighted by
        their observed share) averaged into PER_BINS phase bins of each detected period (phase of the patch start),
        then read by every lead patch at its own phase; summed over the active periods -> [N, K, D]. Causal: only
        o_t with t <= j and origin j's periods."""
        cfg = self.cfg
        Pt, K, NB, W = cfg.patch, cfg.k_steps, PER_BINS, cfg.context // cfg.patch
        N, dev = len(b), o.device
        t = j[:, None] - W + 1 + torch.arange(W, device=dev)             # [N, W]
        tc = t.clamp(min=0)
        wt = ctx.tok[b[:, None], tc, Pt:2 * Pt].mean(-1) * (t >= 0)      # observed share of patch t
        ow = o[b[:, None], tc]                                           # [N, W, D]
        per = ctx.periods[b, j]                                          # [N, Kp]
        Kp = per.shape[1]
        act = (per > 0).float()
        p = per.clamp(min=1)
        bt = torch.floor(NB * torch.remainder(Pt * tc[:, None, :].float(), p[:, :, None]) / p[:, :, None])
        A = F.one_hot(bt.long().clamp(0, NB - 1), NB).float() * wt[:, None, :, None]       # [N, Kp, W, NB]
        A = A.permute(0, 1, 3, 2).reshape(N, Kp * NB, W)
        T = torch.bmm(A, ow) / A.sum(-1, keepdim=True).clamp(min=1e-6)   # [N, Kp * NB, D], empty bin = 0
        s = Pt * (j[:, None] + 1 + torch.arange(K, device=dev))          # lead patch starts [N, K]
        bl = torch.floor(NB * torch.remainder(s[:, :, None].float(), p[:, None, :]) / p[:, None, :]).long()
        idx = bl.clamp(0, NB - 1) + NB * torch.arange(Kp, device=dev)    # [N, K, Kp]
        R = T.gather(1, idx.view(N, K * Kp, 1).expand(-1, -1, T.shape[-1])).view(N, K, Kp, -1)
        return (R * act[:, None, :, None]).sum(2)

    def encode(self, ctx):
        r = self.embed(ctx.tok)                                          # [B, n, D]
        if self.period:
            r = r + self.ph_emb(ctx.tok_phase)
        B_, n, _ = r.shape
        layers = []
        for mx, pe in self.stack():
            if pe is not None:
                r = r + pe
            x, gate = mx.pre(r)
            xp = F.pad(x, (0, 0, mx.kc - 1, 0))                         # causal left pad
            win = xp.unfold(1, mx.kc, 1).transpose(-1, -2)               # [B, n, kc, E]
            xc = mx.conv(win)
            lr, li, vr, vi = mx.coeffs(xc)                               # [B, n, P]
            hr = torch.zeros_like(vr[:, 0]); hi = torch.zeros_like(hr)
            Hr, Hi = [], []
            for t in range(n):                                           # sequential selective scan
                hr, hi = lr[:, t] * hr - li[:, t] * hi + vr[:, t], lr[:, t] * hi + li[:, t] * hr + vi[:, t]
                Hr.append(hr); Hi.append(hi)
            Hr, Hi = torch.stack(Hr, 1), torch.stack(Hi, 1)
            layers.append((Hr, Hi, win[:, :, 1:]))                       # state and conv buffer after patch t
            r = r + mx.post(Hr, Hi, xc, gate)
        return layers, self.norm_f(r)

    def decode(self, enc, ctx, b, j):
        cfg, abl = self.cfg, self.cfg.abl
        P, N, K = cfg.patch, len(b), cfg.k_steps
        layers, o = enc
        ref = ctx.ref[b, j].view(N, K, P)
        U = self.step_emb[None] + self.drive(ref)                        # [N, K, D]
        hx = None                                                        # extra decoder-head input (PERIOD_BIN)
        if self.period:                                                  # the rollout's clock (every ablation)
            U = U + self.ph_emb(ctx.fut_phase[b, j])
        if self.pbin:
            R = self.template(o, ctx, b, j)
            U = U + self.tmpl_in(R)
            hx = self.tmpl_out(R)
        stack = self.stack()
        mods = self.spectrum_mods(o[b, j]) if abl != "copy" else None
        qs = None
        if abl == "copy":
            O = o[b, j][:, None] + U
        elif abl == "direct":                                            # one step from the origin state per lead
            r = U
            for (mx, pe), (Hr, Hi, Bf), md in zip(stack, layers, mods):
                if pe is not None:
                    r = r + pe
                hr, hi, buf = Hr[b, j][:, None], Hi[b, j][:, None], Bf[b, j][:, None].expand(-1, K, -1, -1)
                md = None if md is None else (md[0][:, None], md[1][:, None])
                r, _, _, _ = mx.step(r, hr.expand(-1, K, -1), hi.expand(-1, K, -1), buf, md)
            O = self.norm_f(r)
        else:                                                            # none / zero: continue the same system
            st = [(Hr[b, j], Hi[b, j], Bf[b, j]) for Hr, Hi, Bf in layers]
            outs, qs, rp = [], [], self.ref_proj(ref) if self.fb_on else None
            if self.fb_on and hx is not None:
                rp = rp + hx
            for k in range(K):
                r = U[:, k]
                if self.fb_on and k > 0:                                 # closed loop: own forecast of lead k-1
                    r = r + self.fb(qs[-1].clamp(-20, 20))
                for li_, (mx, pe) in enumerate(stack):
                    if pe is not None:
                        r = r + pe
                    hr, hi, buf = st[li_]
                    r, hr, hi, buf = mx.step(r, hr, hi, buf, mods[li_])
                    st[li_] = (hr, hi, buf)
                outs.append(self.norm_f(r))
                if self.fb_on:
                    qs.append(self.decode_head(outs[-1] + rp[:, k]))
            O = torch.stack(outs, 1)
            if not self.fb_on:
                qs = None
        if abl == "zero":
            O = torch.zeros_like(O)
            qs = None
        if qs is not None:                                               # FB: head already applied per lead
            q = torch.stack(qs, 1).view(N, K, P, cfg.nq)
        else:
            hin = O + self.ref_proj(ref)
            if hx is not None:
                hin = hin + hx
            q = self.decode_head(hin).view(N, K, P, cfg.nq)
        return q, {}
