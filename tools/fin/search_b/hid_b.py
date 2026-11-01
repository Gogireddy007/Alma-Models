"""har_intra_dow reference (red-team's 13-parameter pooled OLS, re-implemented by B on the loop4 cache) on E1-E3:
log RV(D) ~ 1 + d1 + w5 + m22 + ref + 6 weekday dummies + US-federal-holiday(D) + last 6h block of D-1 + last hour,
fit on the 16 training coins (label day <= fit_end); up = pred > ref. Also its logistic twin on y_vol (p_up)."""
import os, sys
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_b as LB  # noqa

D = LB.cache()
tm, days = D["train_mask"], D["days"]
hol = set(USFederalHolidayCalendar().holidays("2019-01-01", "2024-12-31").to_numpy())


def feats(idx):
    H = D["har"][idx]; A = D["A"][idx]
    n, m = A.shape
    hd = np.array([d in hol for d in days[idx]], float)[:, None].repeat(m, 1)
    blk = D["X"][idx][:, :, -1, 12]                         # last 6h block of D-1 (relative to A)
    lh = D["hz"][idx][:, :, -1, 0]                          # last hour of D-1 (relative to A, daily units)
    f = [np.ones((n, m)), H[..., 0] + A, H[..., 1] + A, H[..., 2] + A, H[..., 3] + A] + \
        [H[..., 8 + k] for k in range(1, 7)] + [hd, blk, lh]
    return np.stack(f, -1)


out = {}
for name in ("har_intra_dow", "har_intra_dow_logit"):
    accs = []
    for fk, (fe, a, b) in LB.FOLDS.items():
        tr = np.nonzero((days <= np.datetime64(fe)) & (days >= np.datetime64("2020-06-01")))[0]
        te = LB.fold_idx(D, a, b)
        Xtr = feats(tr)[:, tm].reshape(-1, 14); oktr = D["ok"][tr][:, tm].reshape(-1)
        Xte = feats(te)
        if name == "har_intra_dow":
            y = D["lrv"][tr][:, tm].reshape(-1); k = oktr & np.isfinite(y) & np.isfinite(Xtr).all(1)
            beta = np.linalg.lstsq(Xtr[k], y[k], rcond=None)[0]
            p = np.where(Xte @ beta > D["ref"][te], 0.75, 0.25)
        else:
            from sklearn.linear_model import LogisticRegression
            y = D["y"][tr][:, tm].reshape(-1); k = oktr & np.isfinite(y)
            Z = Xtr[:, 1:].copy(); Z[:, :4] -= Xtr[:, 4:5]          # levels relative to ref
            clf = LogisticRegression(max_iter=3000).fit(Z[k], y[k].astype(int))
            Zt = Xte[..., 1:].copy(); Zt[..., :4] -= Xte[..., 4:5]
            p = clf.predict_proba(Zt.reshape(-1, 13))[:, 1].reshape(Xte.shape[:2])
        p = np.where(D["ok"][te], p, np.nan)
        r = LB.score(D, te, p, np.full(p.shape + (13,), np.nan))
        accs.append(r["all"])
        np.savez(os.path.join(LB.FB.OUTB, "preds", f"{name}_{fk}.npz"), p=p, idx=te)
        LB.log_row({"kind": "ref_b", "name": name, "config": "red-team har_intra_dow, B re-implementation", "seed": 0,
                    "fold": fk, "status": "ok", "P1_all": r["all"], "P1_train": r["train"], "P1_heldout": r["heldout"]})
    LB.log_row({"kind": "ref_b", "name": name, "config": "red-team har_intra_dow, B re-implementation", "seed": 0,
                "fold": "EM", "status": "ok", "P1_all": float(np.mean(accs)), "note": " ".join(f"{x:.4f}" for x in accs)})
    print(f"{name}: EM {np.mean(accs):.4f} " + " ".join(f"{x:.4f}" for x in accs))
