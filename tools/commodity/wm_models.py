"""Commodity world models and their matched direct twins (docs/commodity_search.md).

Every model has the same encoder: per-step features (wm_core.features) -> MLP embedding -> GRU state d_t over the
context (the observer: each observation corrects the state). They differ in the latent state they carry and in
how the horizon is produced.

  obs   deterministic observer-SSM. WM: d_k = GRU(step_emb[k], d_{k-1}) from the origin state, the SAME GRU that
        assimilated the context, run open-loop (no observations); quantile head on d_k (cumulative change to k).
  sv    stochastic-volatility WM. Explicit latent log-vol h_t (one per series) with learned mean-reverting dynamics
        and an observation correction (score-driven: |eta| and leverage eta terms), plus the GRU return state d_t.
        r_{t+1} = mu(d_t) + exp(h_t) eta, eta ~ Student-t(nu) (N = 1) or N(0, I + F F^T) (N > 1, joint latent over
        commodities). Imagination: sample eta, add a latent vol shock sigma_v(d) eps to h, feed the sampled
        returns back into the same GRU (closed loop) -> M paths -> quantiles. Trained by one-step NLL over the
        window (teacher forcing) + sample CRPS of the rolled-out cumulative change.
  rs    regime-switching WM (RSSM-style): K discrete regimes with a state-dependent transition matrix A(d_t),
        exact forward filtering of the regime belief over the context, per-regime Student-t emissions; imagination
        samples regimes and returns and feeds them back. Same losses as sv.

Direct twins (abl="direct"): identical modules and parameter count. Lead k is ONE step of the same machinery from
the origin state: d_k = GRU(step_emb[k], d_origin), and the heads decode the cumulative change to k directly
(obs: quantiles; sv: Student-t / Gaussian with scale exp(h_origin + m(d_k)) sqrt(k); rs: K-component mixture
with weights pi_origin A(d_k)). No state is carried from lead k to k+1 and nothing is sampled step by step.

Units: everything inside is in z units of the target (log return / window RMS of the target's log returns);
quantiles() multiplies back by that scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import wm_core as C

QL = torch.tensor(C.Q9)


def pinball_t(q, y):
    """q [..., 9], y [...] -> mean pinball."""
    e = y[..., None] - q
    return torch.maximum(QL * e, (QL - 1) * e).mean()


def crps_samples(x, y):
    """x [M, B, H] samples, y [B, H] -> mean CRPS (energy form, unbiased pair term)."""
    M = x.shape[0]
    t1 = (x - y[None]).abs().mean(0)
    xs, _ = torch.sort(x, 0)
    i = torch.arange(1, M + 1, dtype=x.dtype).view(M, 1, 1)
    t2 = ((2 * i - M - 1) * xs).sum(0) / (M * (M - 1))       # = 0.5 E|X - X'|
    return (t1 - t2).mean()


def sample_quantiles(x):
    """x [M, B, H] -> [B, 9, H]."""
    return torch.quantile(x, QL, dim=0).permute(1, 0, 2)


class Base(nn.Module):
    def __init__(self, N=1, d=64, d_in=32, abl="none", spreads=False, H=C.H_MAX):
        super().__init__()
        self.N, self.d, self.abl, self.spreads, self.Hmax = N, d, abl, spreads, H
        Fdim = C.n_features(N, spreads)
        self.embed = nn.Sequential(nn.Linear(Fdim, d_in), nn.GELU(), nn.Linear(d_in, d_in))
        self.gru = nn.GRU(d_in, d, batch_first=True)
        self.step_emb = nn.Parameter(torch.randn(H, d_in) * 0.1)
        assert abl in ("none", "direct")

    def encode(self, lp_ctx):
        x, s, ext = C.features(lp_ctx, self.spreads)
        ds, dL = self.gru(self.embed(x))
        return ds, dL[0], s, ext, x

    def gru_step(self, u, d):
        return self.gru(u[:, None], d[None])[1][0]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------------------------------------------ obs
class ObsWM(Base):
    def __init__(self, aux=0.0, **kw):
        super().__init__(**kw)
        self.aux = aux
        d = self.d
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 9))
        self.head1 = nn.Linear(d, 9)                            # one-step observation model (aux, both modes)

    def _q(self, o):
        raw = self.head(o)
        mid = raw[..., 4:5]
        inc = F.softplus(raw[..., 5:]) + 1e-3
        dec = F.softplus(raw[..., :4]) + 1e-3
        up = mid + torch.cumsum(inc, -1)
        dn = mid - torch.cumsum(dec.flip(-1), -1).flip(-1)
        return torch.cat([dn, mid, up], -1)

    def horizon(self, dL, H):
        out, d = [], dL
        B = dL.shape[0]
        for k in range(H):
            u = self.step_emb[k].expand(B, -1)
            d = self.gru_step(u, d if self.abl == "none" else dL)
            out.append(self._q(d))
        return torch.stack(out, 1)                              # [B, H, 9]

    def loss(self, w, L, H):
        ds, dL, s, _, x = self.encode(w[:, :L + 1])
        y = (w[:, L + 1:, 0] - w[:, L:L + 1, 0]) / s[:, :1]
        q = self.horizon(dL, H)
        lo = pinball_t(q, y)
        info = {"pin": lo.item()}
        if self.aux > 0:
            z1 = x[:, 1:, 0]
            a = pinball_t(self.head1(ds[:, :-1]), z1)
            lo = lo + self.aux * a
            info["aux"] = a.item()
        return lo, info

    @torch.no_grad()
    def quantiles(self, lp_ctx, H):
        _, dL, s, _, _ = self.encode(lp_ctx)
        q = self.horizon(dL, H) * s[:, :1, None]
        return q.permute(0, 2, 1)


# ------------------------------------------------------------------------------------------------------- sv
class SVWM(Base):
    def __init__(self, rank=2, lam=1.0, M_train=16, M_eval=512, burn=20, **kw):
        super().__init__(**kw)
        N, d = self.N, self.d
        self.lam, self.M_train, self.M_eval, self.burn, self.rank = lam, M_train, M_eval, burn, rank
        # per series: mu, kappa, m, beta, gamma, sigma_v  (+ low-rank factor when N > 1)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6 * N + (N * rank if N > 1 else 0)))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)
        self.nu_raw = nn.Parameter(torch.tensor(1.5))

    def nu(self):
        return 2.1 + F.softplus(self.nu_raw) * 4

    def params(self, d):
        N = self.N
        o = self.head(d)
        mu = 0.1 * o[..., :N]
        kappa = torch.sigmoid(o[..., N:2 * N] - 3.0)           # ~0.05 at init
        m = o[..., 2 * N:3 * N]
        beta = 0.1 + 0.1 * torch.tanh(o[..., 3 * N:4 * N])
        gamma = 0.1 * torch.tanh(o[..., 4 * N:5 * N])
        sv = 0.05 + F.softplus(o[..., 5 * N:6 * N] - 2.0) * 0.2
        Fm = o[..., 6 * N:].reshape(*o.shape[:-1], N, self.rank) * 0.3 if N > 1 else None
        return mu, kappa, m, beta, gamma, sv, Fm

    def h_update(self, h, eta, kappa, m, beta, gamma, sv=None, eps=None):
        g = eta.abs().clamp(max=6.0) - 0.8
        h = h + kappa * (m - h) + beta * g + gamma * eta
        if eps is not None:
            h = h + sv * eps
        return h.clamp(-4, 4)

    def logp(self, eta, h, Fm):
        """log density of z = mu + exp(h) eta."""
        if self.N == 1:
            nu = self.nu()
            lp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                  - (nu + 1) / 2 * torch.log1p(eta[..., 0] ** 2 / nu))
            return lp - h[..., 0]
        dist = torch.distributions.LowRankMultivariateNormal(torch.zeros_like(eta), Fm, torch.ones_like(eta))
        return dist.log_prob(eta) - h.sum(-1)

    def sample_eta(self, shape_like, Fm):
        if self.N == 1:
            nu = self.nu()
            g = torch.distributions.Gamma(nu / 2, nu / 2).rsample(shape_like[..., :1].shape)
            return torch.randn_like(shape_like) / g.sqrt()
        e = torch.randn_like(shape_like)
        f = torch.randn(*shape_like.shape[:-1], self.rank)
        return e + (Fm @ f[..., None])[..., 0]

    def filter(self, ds, z):
        """Teacher-forced pass. ds [B, T, d] states after obs t, z [B, T, N] obs. Predicts z_{t+1} from (d_t, h_t).
        Returns mean NLL over t >= burn and the final h."""
        B, T, N = z.shape
        h = torch.log(z[:, :self.burn].pow(2).mean(1).sqrt().clamp(min=0.05))
        P = self.params(ds)
        nll = []
        for t in range(self.burn, T):
            mu, kappa, m, beta, gamma, _, Fm = [p[:, t - 1] if p is not None else None for p in P]
            eta = (z[:, t] - mu) / torch.exp(h)
            nll.append(-self.logp(eta, h, Fm))
            h = self.h_update(h, eta.clamp(-20, 20), kappa, m, beta, gamma)
        return torch.stack(nll, 1), h

    def rollout(self, dL, hL, H, M, ext):
        """Closed-loop imagination. Returns cum target change [M, B, H] in z units of the target."""
        B = dL.shape[0]
        d = dL.repeat(M, 1)
        h = hL.repeat(M, 1)
        cum = torch.zeros(M * B, self.N)
        ls = ext["ls"].repeat(M, 1)
        out = []
        for k in range(H):
            mu, kappa, m, beta, gamma, sv, Fm = self.params(d)
            eta = self.sample_eta(mu, Fm)
            z = mu + torch.exp(h) * eta
            cum = cum + z
            out.append(cum[:, 0])
            h = self.h_update(h, eta.clamp(-20, 20), kappa, m, beta, gamma, sv, torch.randn_like(h))
            feat = [z.clamp(-10, 10), torch.ones_like(z), ls]
            if self.spreads and self.N > 1:
                lpch = cum * ext["s_rep"]                         # log-price change per series
                sp = ext["sp_last"] + lpch[:, 1:] - lpch[:, :1]
                feat.append(((sp - ext["sp_mu"]) / ext["sp_sd"]).clamp(-6, 6))
            u = self.embed(torch.cat(feat, -1)) + self.step_emb[k]
            d = self.gru_step(u, d)
        return torch.stack(out, -1).view(M, B, H)

    def direct(self, dL, hL, H, M):
        B = dL.shape[0]
        outs, ys = [], []
        for k in range(H):
            dk = self.gru_step(self.step_emb[k].expand(B, -1), dL)
            mu, _, m, _, _, _, Fm = self.params(dk)
            scale = torch.exp(hL + m) * math.sqrt(k + 1)
            outs.append((mu * math.sqrt(k + 1), scale, Fm))
        return outs

    def _ext(self, lp_ctx, ext, x):
        e = {"ls": x[:, -1, 2 * self.N:3 * self.N], "s_rep": None}
        if self.spreads and self.N > 1:
            sp_last = lp_ctx[:, -1, 1:] - lp_ctx[:, -1, :1]
            e.update(sp_mu=ext["sp_mu"][:, 0], sp_sd=ext["sp_sd"][:, 0], sp_last=sp_last)
        return e

    def _prep_rep(self, e, s, M):
        out = dict(e)
        out["s_rep"] = s.repeat(M, 1)
        for k in ("sp_mu", "sp_sd", "sp_last"):
            if k in e:
                out[k] = e[k].repeat(M, 1)
        return out

    def loss(self, w, L, H):
        # teacher-forced filter over context + future (future NLL is legitimate: training windows end before cut)
        ds_ctx, _, s, ext, x = self.encode(w[:, :L + 1])
        z_ctx = x[:, :, :self.N]
        nll, hL = self.filter(ds_ctx, z_ctx)
        lo = nll.mean()
        info = {"nll": lo.item()}
        dL = ds_ctx[:, -1]
        y = (w[:, L + 1:, 0] - w[:, L:L + 1, 0]) / s[:, :1]
        M = self.M_train
        if self.abl == "none":
            e = self._prep_rep(self._ext(w[:, :L + 1], ext, x), s, M)
            sam = self.rollout(dL, hL, H, M, e)
            cr = crps_samples(sam, y)
        else:
            cr = 0.0
            nl = 0.0
            for k, (mu, sc, Fm) in enumerate(self.direct(dL, hL, H, M)):
                eta = self.sample_eta(mu.expand(M, -1, -1), Fm)
                sam = mu[..., 0] + sc[..., 0] * eta[..., 0]
                cr = cr + crps_samples(sam[..., None], y[:, k:k + 1]) / H
                if self.N == 1:
                    et = (y[:, k:k + 1] - mu) / sc
                    nl = nl - (self.logp(et, torch.log(sc), Fm)).mean() / H
            if self.N == 1:
                lo = lo + 0.1 * nl
        lo = lo + self.lam * cr
        info["crps"] = float(cr)
        return lo, info

    @torch.no_grad()
    def samples(self, lp_ctx, H):
        ds, dL, s, ext, x = self.encode(lp_ctx)
        _, hL = self.filter(ds, x[:, :, :self.N])
        M = self.M_eval
        if self.abl == "none":
            e = self._prep_rep(self._ext(lp_ctx, ext, x), s, M)
            sam = self.rollout(dL, hL, H, M, e)
        else:
            cols = []
            for mu, sc, Fm in self.direct(dL, hL, H, M):
                eta = self.sample_eta(mu.expand(M, -1, -1), Fm)
                cols.append(mu[..., 0] + sc[..., 0] * eta[..., 0])
            sam = torch.stack(cols, -1)
        return sam * s[None, :, :1]

    @torch.no_grad()
    def quantiles(self, lp_ctx, H):
        return sample_quantiles(self.samples(lp_ctx, H))


# ------------------------------------------------------------------------------------------------------- rs
class RSWM(Base):
    def __init__(self, K=3, lam=1.0, M_train=16, M_eval=512, burn=20, **kw):
        super().__init__(**kw)
        assert self.N == 1
        d = self.d
        self.K, self.lam, self.M_train, self.M_eval, self.burn = K, lam, M_train, M_eval, burn
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * K + K * K))
        nn.init.zeros_(self.head[2].weight)
        with torch.no_grad():
            b = torch.zeros(2 * K + K * K)
            b[K:2 * K] = torch.linspace(-0.7, 0.7, K)          # calm / normal / turbulent at init
            b[2 * K:] = (torch.eye(K) * 4.0).reshape(-1)       # persistent regimes
            self.head[2].bias.copy_(b)
        self.nu_raw = nn.Parameter(torch.tensor(1.5))

    def nu(self):
        return 2.1 + F.softplus(self.nu_raw) * 4

    def params(self, d):
        K = self.K
        o = self.head(d)
        mu = 0.1 * o[..., :K]
        ls = o[..., K:2 * K].clamp(-4, 4)
        logA = F.log_softmax(o[..., 2 * K:].reshape(*o.shape[:-1], K, K), -1)
        return mu, ls, logA

    def tlogp(self, z, mu, ls):
        nu = self.nu()
        e = (z - mu) / torch.exp(ls)
        return (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi)
                - (nu + 1) / 2 * torch.log1p(e ** 2 / nu) - ls)

    def filter(self, ds, z):
        B, T = z.shape
        K = self.K
        logpi = torch.full((B, K), -math.log(K))
        mu, ls, logA = self.params(ds)
        nll = []
        for t in range(1, T):
            prior = torch.logsumexp(logpi[:, :, None] + logA[:, t - 1], 1)      # [B, K]
            lik = self.tlogp(z[:, t:t + 1], mu[:, t - 1], ls[:, t - 1])
            joint = prior + lik
            lse = torch.logsumexp(joint, 1)
            if t >= self.burn:
                nll.append(-lse)
            logpi = joint - lse[:, None]
        return torch.stack(nll, 1), logpi

    def rollout(self, dL, logpi, H, M, ls_feat):
        B = dL.shape[0]
        d = dL.repeat(M, 1)
        lp = logpi.repeat(M, 1)
        lsf = ls_feat.repeat(M, 1)
        cum = torch.zeros(M * B)
        out = []
        nu = self.nu()
        for k in range(H):
            mu, ls, logA = self.params(d)
            prior = torch.logsumexp(lp[:, :, None] + logA, 1)
            s = torch.distributions.Categorical(logits=prior).sample()
            oh = F.one_hot(s, self.K).float()
            g = torch.distributions.Gamma(nu / 2, nu / 2).rsample((M * B,))
            eta = torch.randn(M * B) / g.sqrt()
            z = (oh * mu).sum(-1) + torch.exp((oh * ls).sum(-1)) * eta
            cum = cum + z
            out.append(cum)
            lp = torch.log(oh.clamp(min=1e-6))
            u = self.embed(torch.stack([z.clamp(-10, 10), torch.ones_like(z), lsf[:, 0]], -1)) + self.step_emb[k]
            d = self.gru_step(u, d)
        return torch.stack(out, -1).view(M, B, H)

    def direct_samples(self, dL, logpi, H, M):
        B = dL.shape[0]
        nu = self.nu()
        cols, comps = [], []
        for k in range(H):
            dk = self.gru_step(self.step_emb[k].expand(B, -1), dL)
            mu, ls, logA = self.params(dk)
            logw = torch.logsumexp(logpi[:, :, None] + logA, 1)
            ls = ls + 0.5 * math.log(k + 1)
            comps.append((mu, ls, logw))
            s = torch.distributions.Categorical(logits=logw).sample((M,))      # [M, B]
            g = torch.distributions.Gamma(nu / 2, nu / 2).rsample((M, B))
            eta = torch.randn(M, B) / g.sqrt()
            cols.append(torch.gather(mu.expand(M, -1, -1), 2, s[..., None])[..., 0]
                        + torch.exp(torch.gather(ls.expand(M, -1, -1), 2, s[..., None])[..., 0]) * eta)
        return torch.stack(cols, -1), comps

    def loss(self, w, L, H):
        ds, dL, s, _, x = self.encode(w[:, :L + 1])
        nll, logpi = self.filter(ds, x[:, :, 0])
        lo = nll.mean()
        info = {"nll": lo.item()}
        y = (w[:, L + 1:, 0] - w[:, L:L + 1, 0]) / s[:, :1]
        if self.abl == "none":
            sam = self.rollout(dL, logpi, H, self.M_train, x[:, -1, 2:3])
            cr = crps_samples(sam, y)
        else:
            sam, comps = self.direct_samples(dL, logpi, H, self.M_train)
            cr = crps_samples(sam, y)
            nl = 0.0
            for k, (mu, ls, logw) in enumerate(comps):
                nl = nl - torch.logsumexp(logw + self.tlogp(y[:, k:k + 1], mu, ls), 1).mean() / H
            lo = lo + 0.1 * nl
        lo = lo + self.lam * cr
        info["crps"] = cr.item()
        return lo, info

    @torch.no_grad()
    def samples(self, lp_ctx, H):
        ds, dL, s, _, x = self.encode(lp_ctx)
        _, logpi = self.filter(ds, x[:, :, 0])
        if self.abl == "none":
            sam = self.rollout(dL, logpi, H, self.M_eval, x[:, -1, 2:3])
        else:
            sam, _ = self.direct_samples(dL, logpi, H, self.M_eval)
        return sam * s[None, :, :1]

    @torch.no_grad()
    def quantiles(self, lp_ctx, H):
        return sample_quantiles(self.samples(lp_ctx, H))


ARCHS = {"obs": ObsWM, "sv": SVWM, "rs": RSWM}
