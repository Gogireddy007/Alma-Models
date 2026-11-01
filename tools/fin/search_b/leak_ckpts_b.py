"""Small (100-step) checkpoints, one per B family, for the perturbation leak test only (never scored)."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_b as LB, models_b as MB, feats_b as FB  # noqa
torch.set_num_threads(3)
B = {"steps": 100, "bs": 32, "lr": 2e-3, "d": 32, "enc_d": "mlp", "coin_drop": 0.5}
FAM = {"tf": {"enc_d": "tf"}, "gru": {"enc_d": "gru"}, "lru": {"enc_d": "lru", "enc_h": "lru"},
       "xattn": {"cross": "attn"}, "wm_st": {"wm": 0.3, "stoch": 4, "cross": "attn"}, "hartf": {"enc_d": "none", "enc_h": "tf"}}
D = LB.cache()
out = []
for k, v in FAM.items():
    m = LB.train_model(D, {**B, **v}, "2024-06-30", 0, log=lambda *a: None)
    p = os.path.join(FB.OUTB, "ckpt", f"leak_{k}.pt")
    MB.save(p, [m], {**B, **v}, {"purpose": "leak test only"})
    out.append(os.path.relpath(p, FB.ROOT))
print(",".join(out))
