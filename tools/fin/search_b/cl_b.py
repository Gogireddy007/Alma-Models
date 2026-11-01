"""Family 7 (continual / online meta-rule) on the DEV-early inner folds, from saved fold predictions (no refit).
Rule = wm_b_fin.BContinual: logit' = a*logit + b, one SGD step per revealed day (lr), median shift mu (lr_q).
The day-D update uses only y(D), applied before the decision for D+1 (same order as harness.run online).
S_C = P1(online) - P1(frozen). Logged as kind 'cl_b'.
  .venv/bin/python tools/fin/search_b/cl_b.py <pred-name> [lr,lr_q ...]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats_b as FB  # noqa
import loop_b as LB   # noqa


def simulate(D, idx, p, lr, lr_q, q=None):
    a, b, mu = 1.0, 0.0, 0.0
    out = np.full_like(p, np.nan)
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    for t, i in enumerate(idx):
        out[t] = 1 / (1 + np.exp(-(a * lg[t] + b)))
        y = D["y"][i]
        k = np.isfinite(y) & np.isfinite(lg[t])
        if k.any():
            g = out[t][k] - y[k]
            a, b = a - lr * float(np.mean(g * lg[t][k])), b - lr * float(np.mean(g))
    return out


def main():
    D = LB.cache()
    name = sys.argv[1]
    rules = [tuple(float(x) for x in r.split(",")) for r in sys.argv[2:]] or [(0.01, 0), (0.03, 0), (0.1, 0)]
    for lr, lrq in rules:
        dif, on, fr = [], [], []
        for fk in ("E1", "E2", "E3"):
            z = np.load(os.path.join(FB.OUTB, "preds", f"{name}_{fk}.npz"))
            p, q, idx = z["p"], z["q"], z["idx"]
            po = simulate(D, idx, p, lr, lrq)
            s0, s1 = LB.score(D, idx, p, q), LB.score(D, idx, po, q)
            dif.append(s1["all"] - s0["all"]); on.append(s1["all"]); fr.append(s0["all"])
            LB.log_row({"kind": "cl_b", "name": f"cl_{name}_lr{lr}", "config": {"base": name, "lr": lr, "lr_q": lrq},
                        "seed": 0, "fold": fk, "status": "ok", "P1_all": s1["all"], "P1_train": s1["train"],
                        "P1_heldout": s1["heldout"], "note": f"S_C {s1['all'] - s0['all']:+.5f} frozen {s0['all']:.5f}"})
        LB.log_row({"kind": "cl_b", "name": f"cl_{name}_lr{lr}", "config": {"base": name, "lr": lr, "lr_q": lrq},
                    "seed": 0, "fold": "EM", "status": "ok", "P1_all": float(np.mean(on)),
                    "note": f"S_C {np.mean(dif):+.5f} " + " ".join(f"{d:+.4f}" for d in dif)})
        print(f"cl {name} lr {lr}: S_C {np.mean(dif):+.5f} ({' '.join(f'{d:+.4f}' for d in dif)}) online {np.mean(on):.4f}")


if __name__ == "__main__":
    main()
