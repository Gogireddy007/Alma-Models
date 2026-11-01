"""
Family `vqtok`: a discrete-token world model (VQ-VAE tokenizer + autoregressive Transformer over codes, IRIS/Genie
style).

Design
  tokenizer  each patch token (values in its own frame, mask, frame features) -> MLP -> G groups of D/G dims, each
             snapped to the nearest of C codes (grouped VQ: G=4 x C=256 at default, straight-through gradients).
             Codebooks are seeded from the first training batch's encoder outputs, then updated by EMA (decay
             TSAR_WM_VQTOK_DECAY) with dead-code restarts (a code whose EMA
             usage falls below 0.1 is re-seeded from a random encoder output of the batch). A small decoder
             reconstructs the patch values (observed points) and frame features from the quantised vector
             (aux `rec`), and the encoder is pulled to its codes (aux `commit`). With TSAR_WM_VQTOK_EMA=0 the
             codebook is learned by gradient instead (aux `codebook`).
  world      a causal Transformer (N_LAYERS, FF ratio 5) reads ONLY the discrete token sequence (quantised code
             vectors + position): the series is a sentence of codes. Its head predicts the next patch's code
             distribution per group (distance logits against the codebook); teacher-forced cross-entropy on every
             position of the window (aux `ce`).
  imagine    from origin j the model rolls K patches forward in token space: the predicted code distribution is
             turned into the expected code embedding (TSAR_WM_VQTOK_FEED=soft; `hard` = argmax with straight-through),
             added to the action u_k = step_emb[k] + drive(ref_k), appended to the sequence, and run through the same
             Transformer with a cache of the context keys/values. The imagined token's hidden state S_k and its code
             embedding e_k are decoded to quantiles: decode_head(S_k + code_out(e_k) + ref_proj(ref_k)). Imagined
             steps are also trained to predict the true future codes (aux `ce_roll`, training windows only).
  monitor    aux `ppl` (weight 0): mean codebook perplexity per group (C = full use); `dead` (weight 0): codes
             restarted this step.

Leakage: future patches of a training window are tokenized for the code targets and the EMA codebook statistics
only (training signal, like the pinball targets). Forecasts at origin j use tokens <= j plus imagined tokens (wm_causal
checks this). At evaluation no codebook update happens (eval mode).

Ablations (TSAR_WM_ABL)
  none    autoregressive imagination: imagined token k attends to the context and to imagined tokens < k, and its
          input is the code predicted by the previous imagined token.
  direct  parallel (MaskGIT-like) decoding: every lead's query token gets the origin's predicted next code e_1 plus
          its own action u_k and position, and attends to the context only (no imagined token sees another). Same
          parameters as none; MACs equal up to the k extra attention scores.
  copy    no transition: S_k = h_j + u_k; the next-code head and code_out are not built, so there is no `ce`/`ce_roll`
          (the tokenizer still trains with rec/commit and the encoder through the pinball loss).
  zero    imagined states and code embeddings zeroed before the decoder (same params as none).

Sizes at width 1 (d=96, G=4, C=256): 368,061 params in total (gru -0.4%), of which 24,576 are the EMA codebook
(requires_grad False, updated by the EMA rule; also lerped by the harness's TSAR_EMA when set), so 343,485 trainable.
copy: 321,404. Width 1.6667 (d=160): 971,773. MACs per forecast: 12.80M none/zero, 12.79M direct, 9.86M copy
(gru 10.4M; 1.23x). Transformer, heads and tokenizer scale as d^2; the codebook as C*d.
Training cost: every origin of a window imagines K tokens through the Transformer (B*n*K = 41k imagined tokens per
batch of 128 vs 5k context tokens), so a training step is ~1.6 s and ~3.2 GB MPS at batch 128 (gru: ~0.35 s,
1.5 GB); expect ~70 min per 2500-step screen. MACs per forecast at evaluation are only 1.23x gru.

Env keys
  TSAR_WM_VQTOK_GROUPS  code groups G (default 4; must divide d_model)
  TSAR_WM_VQTOK_CODES   codes per group C (default 256)
  TSAR_WM_VQTOK_EMA     1 = EMA codebook + restarts (default), 0 = gradient codebook loss
  TSAR_WM_VQTOK_DECAY   EMA decay (default 0.99)
  TSAR_WM_VQTOK_FEED    soft (expected code embedding, default) | hard (argmax, straight-through)
  TSAR_WM_VQTOK_CE      weight of ce and ce_roll (default 0.1)
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily

_G = int(os.environ.get("TSAR_WM_VQTOK_GROUPS", 4))
_C = int(os.environ.get("TSAR_WM_VQTOK_CODES", 256))
_EMA = int(os.environ.get("TSAR_WM_VQTOK_EMA", 1))
_DECAY = float(os.environ.get("TSAR_WM_VQTOK_DECAY", 0.99))
_FEED = os.environ.get("TSAR_WM_VQTOK_FEED", "soft")
_CE = float(os.environ.get("TSAR_WM_VQTOK_CE", 0.1))
if _FEED not in ("soft", "hard"):
    raise SystemExit(f"TSAR_WM_VQTOK_FEED={_FEED!r} not in soft|hard")
_DEAD = 0.1


def _dot(x, cb):
    """x [..., G, dg], cb [G, C, dg] -> x . cb^T per group [..., G, C] (explicit bmm)."""
    sh = x.shape[:-2]
    xg = x.reshape(-1, x.shape[-2], x.shape[-1]).transpose(0, 1)          # [G, M, dg]
    return torch.bmm(xg, cb.transpose(1, 2)).transpose(0, 1).reshape(*sh, cb.shape[0], cb.shape[1])


def _mix(p, cb):
    """p [..., G, C] weights, cb [G, C, dg] -> [..., G, dg]."""
    sh = p.shape[:-2]
    pg = p.reshape(-1, p.shape[-2], p.shape[-1]).transpose(0, 1)          # [G, M, C]
    return torch.bmm(pg, cb).transpose(0, 1).reshape(*sh, cb.shape[0], cb.shape[2])


class _Block(nn.Module):
    """Pre-LN Transformer block with an explicit key/value interface (full causal pass, or one cached step)."""

    def __init__(self, D, H, ff):
        super().__init__()
        self.H, self.dh = H, D // H
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, ff * D), nn.GELU(), nn.Linear(ff * D, D))

    def _qkv(self, x):
        B, n, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v                                               # [B, H, n, dh]

    def full(self, x, mask):
        q, k, v = self._qkv(x)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.ln2(x)), k, v

    def step(self, x, Kp, Vp, okp, Ki, Vi):
        """One imagined token per origin. x [R, Q, D]: R rows, Q origins per row; prefix keys/values Kp, Vp
        [R, H, n, dh] shared by the row's origins, validity okp [R, Q, n] (key position <= origin); Ki/Vi lists of
        [R, H, Q, dh], earlier imagined tokens of the SAME origin. Returns new x and this token's k, v."""
        q, k, v = self._qkv(x)                                       # [R, H, Q, dh]
        Ks, Vs = Ki + [k], Vi + [v]
        sc = 1.0 / math.sqrt(self.dh)
        sp = (q @ Kp.transpose(-1, -2)) * sc                         # [R, H, Q, n]
        sp = sp.masked_fill(~okp[:, None], float("-inf"))
        si = torch.stack([(q * kj).sum(-1) for kj in Ks], -1) * sc   # [R, H, Q, m] (per-origin keys, no stack copy)
        p = torch.softmax(torch.cat([sp, si], -1), -1)
        n = Kp.shape[2]
        a = p[..., :n] @ Vp
        for i, vj in enumerate(Vs):
            a = a + p[..., n + i, None] * vj                         # [R, H, Q, dh]
        x = x + self.out(a.transpose(1, 2).reshape(x.shape))
        return x + self.mlp(self.ln2(x)), k, v


class VQTokFamily(WMFamily):
    name = "vqtok"
    aux_weights = {"rec": 1.0, "commit": 0.25, "ce": _CE, "ce_roll": _CE, "ppl": 0.0, "dead": 0.0}
    if not _EMA:
        aux_weights["codebook"] = 1.0
    env_keys = ("TSAR_WM_VQTOK_GROUPS", "TSAR_WM_VQTOK_CODES", "TSAR_WM_VQTOK_EMA", "TSAR_WM_VQTOK_DECAY",
                "TSAR_WM_VQTOK_FEED", "TSAR_WM_VQTOK_CE")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P, NQ = cfg.d_model, cfg.patch, cfg.nq
        if D % _G:
            raise SystemExit(f"vqtok: TSAR_WM_VQTOK_GROUPS={_G} must divide d_model={D}")
        self.dg = D // _G
        # tokenizer
        self.tok_enc = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.codebook = nn.Parameter(torch.randn(_G, _C, self.dg) * 0.5, requires_grad=not _EMA)
        self.register_buffer("cl_size", torch.ones(_G, _C))
        self.register_buffer("cl_sum", self.codebook.detach().clone())
        self.register_buffer("cb_init", torch.zeros(()))           # 1 once seeded from the first training batch
        self.tok_dec = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, P + 3))
        # world model over codes
        self.pos = nn.Parameter(torch.zeros(cfg.n_patch + cfg.k_steps, D))
        self.blocks = nn.ModuleList(_Block(D, cfg.n_heads, 5) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(D)
        has_head = cfg.abl != "copy"
        self.code_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D)) \
            if has_head else None
        self.log_temp = nn.Parameter(torch.zeros(())) if has_head else None
        self.code_out = nn.Linear(D, D) if has_head else None
        # actions and decoder (as gru)
        self.step_emb = nn.Parameter(torch.zeros(cfg.k_steps, D))
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.ref_proj = nn.Linear(P, D)
        self.decode_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, P * NQ))
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))

    def param_groups(self):
        g = {"tokenizer": nn.ModuleList([self.tok_enc, self.tok_dec]), "transformer": self.blocks}
        if self.code_head is not None:
            g["code_head"] = self.code_head
        return g

    def _train_aux(self):
        return self.training and torch.is_grad_enabled()

    # ---- tokenizer -------------------------------------------------------------------------------------------
    def _quantize(self, ze):
        """ze [..., D] -> (zq straight-through [..., D], codes [..., G], zg [..., G, dg], q [..., G, dg])."""
        zg = ze.view(*ze.shape[:-1], _G, self.dg)
        cb = self.codebook                                           # [G, C, dg]
        d = zg.pow(2).sum(-1, keepdim=True) - 2 * _dot(zg, cb) + cb.pow(2).sum(-1)                                    # [..., G, C]
        codes = d.argmin(-1)
        q = cb[torch.arange(_G, device=ze.device), codes]            # [..., G, dg]
        zq = zg + (q - zg).detach()
        return zq.reshape(ze.shape), codes, zg, q

    @torch.no_grad()
    def _seed_codebook(self, z):
        """Data-dependent init: every code = a random encoder output of the first training batch (+ small noise)."""
        if z.shape[0] == 0:
            return
        zg = z.view(-1, _G, self.dg)
        pick = torch.randint(zg.shape[0], (_G, _C), device=z.device)
        v = zg[pick, torch.arange(_G, device=z.device)[:, None]]                # [G, C, dg]
        v = v + 0.01 * zg.std() * torch.randn_like(v)
        self.codebook.data.copy_(v)
        self.cl_sum.copy_(v)
        self.cl_size.fill_(1.0)
        self.cb_init.fill_(1.0)

    @torch.no_grad()
    def _ema_codebook(self, zg, codes, aux):
        """zg [M, G, dg], codes [M, G] (valid positions only)."""
        oh = F.one_hot(codes, _C).float()                            # [M, G, C]
        cnt = oh.sum(0)                                              # [G, C]
        s = torch.bmm(oh.permute(1, 2, 0), zg.transpose(0, 1))     # [G, C, dg]
        self.cl_size.mul_(_DECAY).add_(cnt, alpha=1 - _DECAY)
        self.cl_sum.mul_(_DECAY).add_(s, alpha=1 - _DECAY)
        n = self.cl_size.sum(1, keepdim=True)
        size = (self.cl_size + 1e-5) / (n + _C * 1e-5) * n           # Laplace smoothing
        self.codebook.data.copy_(self.cl_sum / size[..., None])
        dead = self.cl_size < _DEAD                                  # [G, C]
        nd = int(dead.sum())
        if nd and zg.shape[0]:
            gi, ci = dead.nonzero(as_tuple=True)
            pick = torch.randint(zg.shape[0], (nd,), device=zg.device)
            v = zg[pick, gi]
            self.codebook.data[gi, ci] = v
            self.cl_sum[gi, ci] = v
            self.cl_size[gi, ci] = 1.0
        aux["dead"] = torch.tensor(float(nd), device=zg.device)

    # ---- world model -----------------------------------------------------------------------------------------
    def _logits(self, h):
        """Next-code logits per group, -||head(h)_g - c||^2 / T  -> [..., G, C]."""
        p = self.code_head(h).view(*h.shape[:-1], _G, self.dg)
        cb = self.codebook.detach() if _EMA else self.codebook
        d = p.pow(2).sum(-1, keepdim=True) - 2 * _dot(p, cb) + cb.pow(2).sum(-1)
        return -d * torch.exp(-self.log_temp)

    def _embed_codes(self, logits):
        """Code distribution -> code embedding [..., D] (expected, or argmax with straight-through)."""
        p = torch.softmax(logits, -1)
        if _FEED == "hard":
            p = F.one_hot(p.argmax(-1), _C).float() + p - p.detach()
        cb = self.codebook.detach() if _EMA else self.codebook
        e = _mix(p, cb)
        return e.reshape(*e.shape[:-2], _G * self.dg)

    def encode(self, ctx):
        B, n, _ = ctx.tok.shape
        aux = {}
        ze = self.tok_enc(ctx.tok)
        train = self._train_aux()
        if train and not bool(self.cb_init):
            self._seed_codebook(ze.detach()[ctx.cnt > 0])
        zq, codes, zg, q = self._quantize(ze)
        if train:
            valid = ctx.cnt > 0                                      # [B, n]
            P = self.cfg.patch
            rec = self.tok_dec(zq)
            obs = ctx.tok[..., P:2 * P]
            tv = torch.cat([ctx.tok[..., :P], ctx.tok[..., 2 * P:]], -1)
            w = torch.cat([obs, torch.ones_like(ctx.tok[..., 2 * P:])], -1) * valid[..., None].float()
            aux["rec"] = (F.smooth_l1_loss(rec, tv, reduction="none") * w).sum() / w.sum().clamp(min=1)
            vm = valid[..., None, None].float()
            msz = (vm.sum() * _G * self.dg).clamp(min=1)
            aux["commit"] = ((zg - q.detach()).pow(2) * vm).sum() / msz
            if not _EMA:
                aux["codebook"] = ((q - zg.detach()).pow(2) * vm).sum() / msz
            cv = codes[valid]                                        # [M, G]
            if _EMA:
                self._ema_codebook(zg[valid].detach(), cv, aux)
            with torch.no_grad():
                pr = F.one_hot(cv, _C).float().mean(0) if cv.shape[0] else torch.zeros(_G, _C, device=ze.device)
                aux["ppl"] = torch.exp(-(pr * torch.log(pr + 1e-10)).sum(-1)).mean()
        x = zq + self.pos[:n]
        mask = self.causal[:n, :n]
        kv = []
        for blk in self.blocks:
            x, k, v = blk.full(x, mask)
            kv.append((k, v))
        h = self.ln_f(x)                                             # [B, n, D]
        if train and self.code_head is not None and n > 1:
            lg = self._logits(h[:, :-1])                             # predicts codes of patch j+1
            okc = (ctx.cnt[:, :-1] > 0)
            aux["ce"] = F.cross_entropy(lg[okc].reshape(-1, _C), codes[:, 1:][okc].reshape(-1))
        return h, kv, codes, aux

    def _imagine(self, h, kv, ctx, b, j):
        """Roll K imagined tokens from each origin. Training passes (nearly) every origin of every row, so the
        rollout runs on the dense [B, n] origin grid (one batched matmul per row against its own keys) and the
        requested origins are gathered at the end; with about one origin per row (evaluation) it runs on the
        gathered origins. Both paths compute the same function. Returns S, E [N, K, D] and next-code logits
        L [N, K-1, G, C] (None outside training)."""
        cfg, abl = self.cfg, self.cfg.abl
        P, K, N = cfg.patch, cfg.k_steps, len(b)
        B, n = h.shape[:2]
        dense = N > B
        if dense:
            jj = torch.arange(n, device=j.device)[None].expand(B, n)            # [R=B, Q=n]
            h0, ref = h, ctx.ref
            KVp = kv
        else:
            jj = j[:, None]                                                   # [R=N, Q=1]
            h0, ref = h[b, j][:, None], ctx.ref[b, j][:, None]
            KVp = [(k[b], v[b]) for k, v in kv]
        R, Q = jj.shape
        U = self.step_emb + self.drive(ref.reshape(R, Q, K, P))              # [R, Q, K, D]
        okp = torch.arange(n, device=j.device)[None, None] <= jj[..., None]   # [R, Q, n]
        e1 = self._embed_codes(self._logits(h0))                              # [R, Q, D] expected code of j+1
        Ki = [[] for _ in self.blocks]
        Vi = [[] for _ in self.blocks]
        train = self._train_aux()
        e, Ss, Es, LG = e1, [], [], []
        for k in range(K):
            inp = e1 if abl == "direct" else e
            x = inp + self.pos[jj + 1 + k] + U[:, :, k]                       # [R, Q, D]
            for li, blk in enumerate(self.blocks):
                seen = (Ki[li], Vi[li]) if abl != "direct" else ([], [])
                x, kk, vv = blk.step(x, KVp[li][0], KVp[li][1], okp, *seen)
                if abl != "direct":
                    Ki[li].append(kk); Vi[li].append(vv)
            s_ = self.ln_f(x)
            Ss.append(s_); Es.append(inp)
            if k < K - 1:
                lg = self._logits(s_)                                         # code of patch j+2+k
                if train:
                    LG.append(lg)
                e = self._embed_codes(lg)
        S, E = torch.stack(Ss, 2), torch.stack(Es, 2)                         # [R, Q, K, D]
        L = torch.stack(LG, 2) if LG else None                                # [R, Q, K-1, G, C]
        if dense:
            S, E = S[b, j], E[b, j]
            L = L[b, j] if L is not None else None
        else:
            S, E = S[:, 0], E[:, 0]
            L = L[:, 0] if L is not None else None
        return S, E, L

    def decode(self, enc, ctx, b, j):
        h, kv, codes, aux = enc
        aux = dict(aux)
        cfg, abl = self.cfg, self.cfg.abl
        P, N, K = cfg.patch, len(b), cfg.k_steps
        n = h.shape[1]
        ref = ctx.ref[b, j]
        if abl == "copy":
            U = self.step_emb[None] + self.drive(ref.view(N, K, P))  # [N, K, D] actions
            S = h[b, j][:, None] + U
            E = None
        else:
            S, E, L = self._imagine(h, kv, ctx, b, j)
            if self._train_aux() and K > 1:                          # L [N, K-1, G, C] -> patches j+2 .. j+K
                pos = j[:, None] + 2 + torch.arange(K - 1, device=j.device)[None]
                ok = pos < n
                pc = pos.clamp(max=n - 1)
                ok = ok & (ctx.cnt[b[:, None], pc] > 0)
                if ok.any():
                    tgt = codes[b[:, None], pc]                      # [N, K-1, G]
                    aux["ce_roll"] = F.cross_entropy(L[ok].reshape(-1, _C), tgt[ok].reshape(-1))
        if abl == "zero":
            S = torch.zeros_like(S)
            E = torch.zeros_like(E)
        z = S + self.ref_proj(ref.view(N, K, P))
        if E is not None:
            z = z + self.code_out(E)
        q = self.decode_head(z).view(N, K, P, cfg.nq)
        return q, aux
