"""Alma-Gas storage-balance world model and its matched direct twin (docs/gas_search.md section 1).

GasNet(rollout=True)  = WORLD MODEL: a latent daily state z_d (d dims) is rolled forward one DAY at a time by a
                        GRUCell driven by that day's weather (observed or CPC forecast), normals and calendar.
                        If assim: the 7 days of week W-1 are rolled first (observed weather), their summed regional
                        flows are compared with the released dS_{W-1} and the innovation is written into the state
                        (a learned Kalman-style update); then the 7 days of target week W are rolled.
GasNet(rollout=False) = DIRECT TWIN: identical modules and parameter count; every day's state is ONE step of the
                        same cell from the origin state (no state carried from day to day). The assimilation
                        innovation is computed the same way (from one-step flows) so both see identical inputs.

Readout (both): daily net injection of region r on day d
    inj_rd = a5_r/7 + sd_r/7 * base_r(z_d) - sum_b aH_rb(z_d) [phiH_b(HDD_d) - phiH_b(HDDn_d)]
                                         - sum_b aC_rb(z_d) [phiC_b(CDD_d) - phiC_b(CDDn_d)]
  with aH, aC >= 0 (softplus) and phi_b monotone non-decreasing => injection is monotone DEcreasing in HDD and CDD
  (demand monotone increasing), anchored at the 5-year-average flow under normal weather. Weekly dS_r = sum_d inj_rd;
  L48 = sum of the 5 regions (mass balance). Quantiles: median + sd_s * monotone offsets from the final state.
regime=True (family 2): base / aH / aC are mixtures over 3 learned regimes with a softmax gate from (z_d, calendar).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

QL = torch.tensor([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])
NQ, NR, NS = 13, 5, 6
HK = (0.0, 10.0, 20.0, 30.0)     # HDD hinge knots
CK = (0.0, 5.0, 10.0)            # CDD hinge knots


def phi(x, knots, s):
    """Monotone non-decreasing basis: softplus hinges (convex) + one saturating sigmoid; [.., B]."""
    out = [F.softplus((x - k) / 2.0) * 2.0 / s for k in knots]
    out.append(torch.sigmoid((x - knots[-1]) / 5.0) * 2.0)
    return torch.stack(out, -1)


class GasNet(nn.Module):
    def __init__(self, g_dim, u_dim, d=32, rollout=True, assim=True, regime=False, nreg=3, drop=0.0):
        super().__init__()
        self.rollout, self.assim, self.regime = rollout, assim, regime
        self.K = nreg if regime else 1
        self.BH, self.BC = len(HK) + 1, len(CK) + 1
        enc = [nn.Linear(g_dim, 64), nn.GELU(), nn.Linear(64, d), nn.Tanh()]
        if drop > 0:                        # only inserted when used (keeps drop=0 models bit-identical to R1)
            enc = [nn.Dropout(drop)] + enc[:2] + [nn.Dropout(drop)] + enc[2:]
        self.enc = nn.Sequential(*enc)
        self.inp = nn.Sequential(nn.Linear(u_dim, d), nn.GELU())
        self.cell = nn.GRUCell(d, d)
        self.innov = nn.Linear(NR, d)
        self.base = nn.Linear(d, self.K * NR)
        self.coef = nn.Linear(d, self.K * NR * (self.BH + self.BC))
        self.gate = nn.Linear(d + u_dim, self.K)
        self.spread = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, NS * (NQ - 1)))
        nn.init.zeros_(self.base.weight); nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.coef.weight, std=0.01); nn.init.constant_(self.coef.bias, -1.0)

    def readout(self, z, u, ph, a5, sd):
        """z [B,d], u [B,U], ph [B,4] (HDD, CDD, HDDn, CDDn) -> daily injection per region [B,5]."""
        B = z.shape[0]
        base = self.base(z).view(B, self.K, NR)
        co = F.softplus(self.coef(z)).view(B, self.K, NR, self.BH + self.BC)
        dH = phi(ph[:, 0], HK, 10.0) - phi(ph[:, 2], HK, 10.0)            # [B,BH]
        dC = phi(ph[:, 1], CK, 5.0) - phi(ph[:, 3], CK, 5.0)              # [B,BC]
        dem = (co[..., :self.BH] * dH[:, None, None]).sum(-1) + (co[..., self.BH:] * dC[:, None, None]).sum(-1)
        flow = base - dem                                                  # [B,K,5] in sd/7 units
        if self.K > 1:
            w = torch.softmax(self.gate(torch.cat([z, u], -1)), -1)[..., None]
            flow = (w * flow).sum(1)
        else:
            flow = flow[:, 0]
        return a5[:, 1:] / 7.0 + sd[1:] / 7.0 * flow

    def run_days(self, z0, U, PH, a5, sd, days):
        z, zs, flows = z0, [], []
        for t in days:
            z = self.cell(self.inp(U[:, t]), z if self.rollout else z0)
            zs.append(z)
            flows.append(self.readout(z, U[:, t], PH[:, t], a5, sd))
        return torch.stack(flows, 1), zs                                  # [B,7,5]

    def forward(self, b, sd, phys_shift=None):
        """b: dict of tensors g [B,G], u [B,14,U], phys [B,14,4], a5 [B,6], dsprev [B,6]; sd [6] dev scale.
        phys_shift [B,14,4] (scenario lever, added to phys AND to the scaled weather channels of u) or None."""
        U, PH = b["u"], b["phys"]
        if phys_shift is not None:
            PH = PH + phys_shift
            U = U.clone()
            U[..., 0] = U[..., 0] + phys_shift[..., 0] / 30; U[..., 1] = U[..., 1] + phys_shift[..., 1] / 15
            U[..., 4] = U[..., 4] + phys_shift[..., 0] / 15; U[..., 5] = U[..., 5] + phys_shift[..., 1] / 8
        z0 = self.enc(b["g"])
        out = {}
        if self.assim:
            fp, zp = self.run_days(z0, U, PH, b["a5"], sd, range(0, 7))
            prev = fp.sum(1)                                              # model's dS_{W-1} per region
            err = (b["dsprev"][:, 1:] - prev) / sd[1:]
            z0 = (zp[-1] if self.rollout else z0) + self.innov(err)
            out["prev"] = prev
        ft, zt = self.run_days(z0, U, PH, b["a5"], sd, range(7, 14))
        reg = ft.sum(1)                                                   # [B,5]
        med = torch.cat([reg.sum(1, keepdim=True), reg], 1)               # [B,6] (L48 = sum of regions)
        inc = F.softplus(self.spread(torch.cat([zt[-1], z0], -1))).view(-1, NS, NQ - 1) * 0.15
        cum = torch.cat([torch.zeros_like(inc[..., :1]), inc.cumsum(-1)], -1)   # [B,6,13]
        off = cum - cum[..., 6:7]
        q = med[..., None] + sd[None, :, None] * off
        out.update(q=q, daily=ft, med=med)
        return out


def pinball(q, y, sd, w):
    """q [B,6,13], y [B,6], sd [6], w [6] series weights; mean pinball in sd units."""
    e = (y[..., None] - q) / sd[None, :, None]
    l = torch.maximum(QL * e, (QL - 1) * e)
    return (l.mean(-1) * w).sum(-1).mean() / w.sum()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
