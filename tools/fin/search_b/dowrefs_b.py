"""Dow-aware reference models on B's inner folds E1-E3 (RT-15 / coordinator): fitted per fold on the 16 training
coins (label day <= fit_end), scored on all 24 coins, from the loop4 cache (nothing after 2024-06-30).
  har_dow_ols   : pooled OLS of log RV(D) on [1, d1, w5, m22 (absolute logs), weekday dummies]; up = pred > ref
  har_dow_logit : logistic regression of y_vol on [HAR block (8, relative to A), weekday one-hot]  (p_up head)
  lgbm_dow      : LightGBM classifier (lead lgbm params) on [HAR block, mom(5), funding, weekday one-hot]
Logged as kind 'ref_b'. The lead's registered har_dow / lgbm_dow (prereg amendment) supersede these.
"""
import os, sys
import numpy as np
import lightgbm  # noqa: F401  (import before torch: OpenMP clash segfaults otherwise)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_b as LB  # noqa

D = LB.cache()
tm = D["train_mask"]
days = D["days"]


def feats(idx, kind):
    H = D["har"][idx]                   # [n, m, 15]: 8 HAR + 7 dow
    A = D["A"][idx][..., None]
    if kind == "ols":
        ab = H[..., :3] + A
        return np.concatenate([np.ones_like(ab[..., :1]), ab, H[..., 9:]], -1)   # drop one dummy (Monday) for OLS
    x = [H]
    if kind == "lgbm":
        x += [np.tanh(D["mom"][idx] / 0.2), np.clip(D["fu_last"][idx][..., None] * 1e3, -5, 5)]
    return np.concatenate(x, -1)


for name in (sys.argv[1:] or ["har_dow_ols", "har_dow_logit", "lgbm_dow"]):
    accs = []
    for fk, (fe, a, b) in LB.FOLDS.items():
        tr = np.nonzero((days <= np.datetime64(fe)) & (days >= np.datetime64("2020-06-01")))[0]
        te = LB.fold_idx(D, a, b)
        kind = name.split("_")[-1]
        Xtr = feats(tr, kind)[:, tm].reshape(-1, feats(tr[:1], kind).shape[-1])
        oktr = D["ok"][tr][:, tm].reshape(-1)
        Xte = feats(te, kind)
        if kind == "ols":
            y = D["lrv"][tr][:, tm].reshape(-1)
            k = oktr & np.isfinite(y) & np.isfinite(Xtr).all(1)
            beta = np.linalg.lstsq(Xtr[k], y[k], rcond=None)[0]
            pred = Xte @ beta
            p = np.where(pred > D["ref"][te], 0.75, 0.25)
        else:
            y = D["y"][tr][:, tm].reshape(-1)
            k = oktr & np.isfinite(y)
            if kind == "logit":
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr[k], y[k].astype(int))
            else:
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                         verbose=-1, n_jobs=2, deterministic=True).fit(Xtr[k], y[k].astype(int))
            p = clf.predict_proba(Xte.reshape(-1, Xte.shape[-1]))[:, 1].reshape(Xte.shape[:2])
        p = np.where(D["ok"][te], p, np.nan)
        q = np.full(p.shape + (13,), np.nan)
        r = LB.score(D, te, p, q)
        accs.append(r["all"])
        LB.log_row({"kind": "ref_b", "name": name, "config": "dow-aware ref (B, in-cache)", "seed": 0, "fold": fk,
                    "status": "ok", "P1_all": r["all"], "P1_train": r["train"], "P1_heldout": r["heldout"]})
    LB.log_row({"kind": "ref_b", "name": name, "config": "dow-aware ref (B, in-cache)", "seed": 0, "fold": "EM",
                "status": "ok", "P1_all": float(np.mean(accs)), "note": " ".join(f"{x:.4f}" for x in accs)})
    print(f"{name}: EM {np.mean(accs):.4f} " + " ".join(f"{x:.4f}" for x in accs), flush=True)
