"""Emulated ssm training step (train_wm.train_loss shapes) on MPS: activation memory and step time vs width, batch,
origins. Random tensors stand in for data (no data load). Run via tools/smoke.py:
    uv run tools/smoke.py --timeout 60 -- .venv/bin/python tools/memprobe_ssm.py WIDTH:LAYERS:BATCH:ORIGINS:MAX_H ..."""
import sys, time, types; sys.path.insert(0, ".")
import torch
from wm.base import WMConfig
from wm.ssm import SSMFamily
dev = "mps"
Q = torch.linspace(0.1, 0.9, 9, device=dev)
for spec in sys.argv[1:]:
    w, L, B, N, mh = spec.split(":"); w = float(w); L, B, N, mh = int(L), int(B), int(N), int(mh)
    D = max(8, 8 * round(96 * w / 8)); n = (256 + mh) // 8; K = mh // 8
    cfg = WMConfig(patch=8, context=256, max_h=mh, k_steps=K, n_patch=n, nq=9, tok_dim=19, d_model=D, n_layers=L,
                   n_heads=4, width=w, abl="none")
    torch.manual_seed(0)
    m = SSMFamily(cfg).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ctx = types.SimpleNamespace(tok=torch.randn(B, n, 19, device=dev), ref=torch.randn(B, n, mh, device=dev))
    N = min(N, B * (n - 1))
    idx = torch.randperm(B * (n - 1), device=dev)[:N]; b, j = idx // (n - 1), idx % (n - 1)
    tgt = torch.randn(N, K, 8, device=dev)
    torch.mps.synchronize(); torch.mps.empty_cache(); base = torch.mps.driver_allocated_memory()
    peak, ts = 0, []
    for it in range(4):
        t0 = time.time()
        q, _ = m.decode(m.encode(ctx), ctx, b, j)
        err = tgt.unsqueeze(-1) - q
        loss = torch.maximum(Q * err, (Q - 1) * err).mean()
        torch.mps.synchronize(); peak = max(peak, torch.mps.current_allocated_memory())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.mps.synchronize()
        peak = max(peak, torch.mps.current_allocated_memory()); ts.append(time.time() - t0)
    drv = torch.mps.driver_allocated_memory()
    print(f"w={w} L={L} d={D} B={B} origins={N} K={K} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M "
          f"alloc_peak={peak/2**20:.0f}MB driver={drv/2**20:.0f}MB step={min(ts[1:])*1000:.0f}ms", flush=True)
    del m, opt, ctx, q, err, loss; torch.mps.empty_cache()
