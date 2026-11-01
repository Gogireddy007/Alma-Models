"""Auditor probe (audit r1, 2026-10-31). Read-only on model code: imports wm families, randomises ALL weights
(so zero-initialised transitions are live), and checks on CPU:
  causal  : outputs at origins j < c unchanged when every per-patch input at >= c is perturbed (dense AND gather);
  rowind  : row 0's outputs unchanged when row 1 is replaced (no cross-window/cross-series mixing in a batch);
  paths   : dense and gather decode agree (eval uses gather, the harness causality check uses dense).
Usage: uv run tools/smoke.py -- uv run python tools/audit_r1_probe.py"""
import importlib, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm.base import WMConfig, Context

P, CTXL, MAXH, NQ = 8, 256, 64, 9
K, NP = MAXH // P, (CTXL + MAXH) // P
cfg = lambda abl: WMConfig(patch=P, context=CTXL, max_h=MAXH, k_steps=K, n_patch=NP, nq=NQ, tok_dim=2 * P + 3,
                           d_model=96, n_layers=2, n_heads=4, width=1.0, abl=abl)


def mkctx(g, B):
    tok = torch.randn(B, NP, 2 * P + 3, generator=g)
    loc, scale = torch.randn(B, NP, generator=g), torch.rand(B, NP, generator=g) + 0.5
    cnt = torch.arange(1, NP + 1).float().repeat(B, 1) * P
    cnt[1, :5] = 0                                                  # left padding on row 1
    ref = torch.randn(B, NP, MAXH, generator=g)
    return Context(tok=tok, x=torch.randn(B, NP * P, generator=g), loc=loc, scale=scale, cnt=cnt, ref=ref)


def perturb_from(ctx, g, c, rows):
    t, r, x = ctx.tok.clone(), ctx.ref.clone(), ctx.x.clone()
    t[rows, c:] += 5 * torch.randn_like(t[rows, c:]); r[rows, c:] += 5 * torch.randn_like(r[rows, c:])
    x[rows, c * P:] += 5 * torch.randn_like(x[rows, c * P:])
    l, s = ctx.loc.clone(), ctx.scale.clone(); l[rows, c:] += 3; s[rows, c:] *= 2
    return Context(tok=t, x=x, loc=l, scale=s, cnt=ctx.cnt, ref=r)


def run(m, ctx, b, j):
    torch.manual_seed(0)
    return m.decode(m.encode(ctx), ctx, b, j)[0]


def probe(mod, cls, abl, env):
    os.environ.update(env)
    fam = getattr(importlib.reload(importlib.import_module(f"wm.{mod}")), cls)(cfg(abl)).eval()
    with torch.no_grad():
        for p in fam.parameters():
            p.normal_(0, 0.1)
    g = torch.Generator().manual_seed(7)
    ctx = mkctx(g, 3)
    c = NP // 2
    b = torch.arange(3).repeat_interleave(c); j = torch.arange(c).repeat(3)
    out = {}
    with torch.no_grad():
        q0 = run(fam, ctx, b, j)
        out["causal"] = (q0 - run(fam, perturb_from(ctx, g, c, slice(None)), b, j)).abs().max().item()
        ctx2 = mkctx(torch.Generator().manual_seed(99), 3)
        mixed = Context(**{k: torch.cat([getattr(ctx, k)[:1], getattr(ctx2, k)[1:]]) for k in
                           ("tok", "x", "loc", "scale", "cnt", "ref")})
        sel = b == 0
        out["rowind"] = (q0[sel] - run(fam, mixed, b, j)[sel]).abs().max().item()
        if mod == "tswm":                                            # gather path (eval) vs dense (check)
            os.environ["TSAR_WM_TSWM_PATH"] = "gather"; fam.path = "gather"
            qg = run(fam, ctx, b, j)
            out["gather_causal"] = (qg - run(fam, perturb_from(ctx, g, c, slice(None)), b, j)).abs().max().item()
            out["dense_vs_gather"] = (qg - q0).abs().max().item()
            fam.path = "auto"; os.environ.pop("TSAR_WM_TSWM_PATH")
    for k in env:
        os.environ.pop(k)
    return out


cases = [("tswm", "TSWMFamily", "none", {"TSAR_WM_TSWM_FRAME": "origin"}),
         ("tswm", "TSWMFamily", "direct", {"TSAR_WM_TSWM_FRAME": "origin"}),
         ("tswm", "TSWMFamily", "none", {"TSAR_WM_TSWM_FRAME": "loc"}),
         ("ssm", None, "none", {}), ("gru", None, "none", {})]
bad = 0
for mod, cls, abl, env in cases:
    if cls is None:
        M = importlib.import_module(f"wm.{mod}")
        cls = next(n for n, v in vars(M).items() if isinstance(v, type) and n.endswith("Family") and v.__module__ == M.__name__)
    r = probe(mod, cls, abl, env)
    ok = all(v < 1e-4 for v in r.values())
    bad += not ok
    print(f"{mod:5s} {abl:6s} {env} -> " + " ".join(f"{k}={v:.1e}" for k, v in r.items()) + ("  OK" if ok else "  FAIL"))
print("audit_r1_probe:", "PASS" if not bad else f"{bad} FAIL")
