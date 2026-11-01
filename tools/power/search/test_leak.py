"""Information-set test for the architect's feature builder + models: forecasts for day D must not change when
anything the rules forbid is altered (prices of D and later, fundamentals of D-1 and later), and must change when
an allowed input (prices of D-1, fundamentals of D-2) is altered. Run: .venv/bin/python tools/power/search/test_leak.py"""
import copy, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wm_data as W, wm_models as M
D = W.PowerData("dev")
idx = W.valid_days(D, "2024-03-01", "2024-03-20", 28)
ok = True
for arch in ["cycle", "phys", "regime"]:
    for abl in ["none", "direct"]:
        torch.manual_seed(0); m = M.build(arch, abl, 32, 28).eval()
        def f(DD):
            b, mu, sd, _ = W.batch(DD, idx, 28, with_target=False)
            with torch.no_grad():
                return m(b)["q"].numpy() * sd[:, None, None] + mu[:, None, None]
        base = f(D)
        for what in ["P_D", "P_D+1", "F_D-1", "F_D", "P_D-1", "F_D-2"]:
            E = copy.copy(D); E.P = D.P.copy(); E.F = D.F.copy()
            if what == "P_D": E.P[idx] += 100
            if what == "P_D+1": E.P[idx + 1] += 100
            if what == "F_D-1": E.F[idx - 1] *= 2
            if what == "F_D": E.F[idx] *= 2
            if what == "P_D-1": E.P[idx - 1] += 100
            if what == "F_D-2": E.F[idx - 2] *= 2
            # perturbing day k for target i also perturbs other targets' allowed context; test each target alone
            changed = []
            for n, i in enumerate(idx):
                E2 = copy.copy(D); E2.P = D.P.copy(); E2.F = D.F.copy()
                off = {"P_D": ("P", 0), "P_D+1": ("P", 1), "F_D-1": ("F", -1), "F_D": ("F", 0), "P_D-1": ("P", -1), "F_D-2": ("F", -2)}[what]
                arr = E2.P if off[0] == "P" else E2.F
                if off[0] == "P": arr[i + off[1]] += 100
                else: arr[i + off[1]] *= 2
                b, mu, sd, _ = W.batch(E2, np.array([i]), 28, with_target=False)
                with torch.no_grad():
                    q = m(b)["q"].numpy() * sd[:, None, None] + mu[:, None, None]
                changed.append(not np.allclose(q[0], base[n], atol=1e-4))
            must_change = what in ("P_D-1", "F_D-2")
            good = all(changed) if must_change else not any(changed)
            ok &= good
            print(f"{arch:7s} {abl:6s} perturb {what:6s} changed={sum(changed)}/{len(changed)} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILURES")
