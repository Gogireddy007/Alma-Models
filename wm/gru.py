"""
Family `gru` (reference): the train_arch2.py design under the wave-4 recipe with TSAR_A2_DRIVE=1 (gc_roll).

  encode  causal Transformer over patch tokens (1 lower layer, a GRU carried patch to patch, N_LAYERS - 1 upper
          layers), z_j = tanh(to_state(h_j + state_j)): the latent world state after observing patch j.
  decode  a GRUCell rolls z forward K lead patches with input u_k = step_emb[k] + drive(ref_k) (the seasonal-naive
          patch as an exogenous "action"); each rolled state is decoded with decode(LN, Linear)(z_k + ref_proj(ref_k)).
          The harness adds the per-quantile seasonal-ref skip.

Modules are built in train_arch2's order (embed, pos, lower, state_rnn, upper, to_state, dynamics, step_emb, decode,
ref_proj, drive[, direct]) so that, at the same seed, initial weights match train_arch2 with A2_DRIVE=1.

Ablations (TSAR_WM_ABL), as train_arch2's W1/W2/W4:
  none    GRUCell rollout.
  direct  W1 (A2_DIRECT): z_k = z_0 + MLP(LN[z_0, u_k]), MLP 2d -> 2d -> d with zero-init last layer; no GRUCell.
          6d^2 + 7d params vs the GRUCell's 6d^2 + 6d; 6d^2 MACs per lead, as the GRUCell.
  copy    W2 (A2_STATE_OFF=copy): z_k = z_0 + u_k; no transition built.
  zero    W4: rolled states zeroed before the decoder (ref_proj, drive-free ref paths and the skip remain).
"""

import torch
import torch.nn as nn

from wm.base import WMFamily


class GRUFamily(WMFamily):
    name = "gru"

    def __init__(self, cfg):
        super().__init__(cfg)
        D, ZD, P, NQ = cfg.d_model, cfg.d_model, cfg.patch, cfg.nq
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.lower = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.state_rnn = nn.GRU(D, D, batch_first=True)
        self.upper = nn.TransformerEncoder(layer, cfg.n_layers - 1, enable_nested_tensor=False) if cfg.n_layers > 1 else None
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        self.to_state = nn.Linear(D, ZD)
        self.dynamics = nn.GRUCell(ZD, ZD) if cfg.abl in ("none", "zero") else None
        self.step_emb = nn.Parameter(torch.zeros(cfg.k_steps, ZD))
        self.decode_head = nn.Sequential(nn.LayerNorm(ZD), nn.Linear(ZD, P * NQ))
        self.ref_proj = nn.Linear(P, ZD)
        self.drive = nn.Linear(P, ZD)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        if cfg.abl == "direct":
            self.direct = nn.Sequential(nn.LayerNorm(2 * ZD), nn.Linear(2 * ZD, 2 * ZD), nn.GELU(), nn.Linear(2 * ZD, ZD))
            nn.init.zeros_(self.direct[-1].weight); nn.init.zeros_(self.direct[-1].bias)

    def param_groups(self):
        return {"transition": self.dynamics if self.dynamics is not None else getattr(self, "direct", nn.Identity())}

    def encode(self, ctx):
        n = ctx.tok.shape[1]
        e = self.embed(ctx.tok) + self.pos[:, :n]
        h = self.lower(e, mask=self.causal[:n, :n])
        state, _ = self.state_rnn(h)
        h = self.upper(h + state, mask=self.causal[:n, :n]) if self.upper is not None else h + state
        return torch.tanh(self.to_state(h + state))                     # [B, n, ZD]

    def decode(self, enc, ctx, b, j):
        cfg, abl = self.cfg, self.cfg.abl
        P, N = cfg.patch, len(b)
        z0 = z = enc[b, j]
        ref = ctx.ref[b, j]                                              # [N, MAX_H]
        zs = []
        for k in range(cfg.k_steps):
            u = self.step_emb[k].expand(N, -1) + self.drive(ref[:, k * P:(k + 1) * P])
            if abl == "direct":
                z = z0 + self.direct(torch.cat([z0, u], -1))
            elif abl == "copy":
                z = z0 + u
            else:
                z = self.dynamics(u, z)
            zs.append(z)
        S = torch.stack(zs, 1)                                           # [N, K, ZD]
        if abl == "zero":
            S = torch.zeros_like(S)
        outs = [self.decode_head(S[:, k] + self.ref_proj(ref[:, k * P:(k + 1) * P])).view(N, P, cfg.nq)
                for k in range(cfg.k_steps)]
        return torch.stack(outs, 1), {}
