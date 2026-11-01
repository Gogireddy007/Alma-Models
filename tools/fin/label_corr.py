"""
Cross-coin dependence of the P1 labels (docs/fin_prereg.md section 6.4), measured on DEV (decision days
2022-07-01..2025-06-30) and VAL (2025-07-01..2025-10-31) only. No model is involved. TEST rows are not loaded.

    nice -n 10 .venv/bin/python tools/fin/label_corr.py   -> runs/fin/label_corr.json
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H     # noqa: E402

OUT = os.path.join(H.ROOT, "runs", "fin", "label_corr.json")


def mean_offdiag_corr(X):
    m = X.shape[1]
    cs = []
    for a in range(m):
        for b in range(a + 1, m):
            ok = np.isfinite(X[:, a]) & np.isfinite(X[:, b])
            if ok.sum() >= 30 and X[ok, a].std() > 0 and X[ok, b].std() > 0:
                cs.append(np.corrcoef(X[ok, a], X[ok, b])[0, 1])
    return float(np.mean(cs)) if cs else float("nan"), len(cs)


def main():
    out = {}
    for market in ("crypto", "etf"):
        P = H.Panel(market)
        groups = {"all": P.symbols} if market == "etf" else {
            "all": P.symbols, "train": [s for s in P.symbols if P.group[s] == "train"],
            "heldout": [s for s in P.symbols if P.group[s] == "heldout"]}
        for split in ("dev", "val"):
            days = P.split_days(split)
            Y = np.stack([P.labels(int(i))["y_vol"] for i in days])
            LR = np.stack([P.labels(int(i))["logrv"] - P.labels(int(i))["ref"] for i in days])
            for g, syms in groups.items():
                js = [P.symbols.index(s) for s in syms]
                rho, npair = mean_offdiag_corr(Y[:, js])
                rho_c, _ = mean_offdiag_corr(LR[:, js])
                m = len(js)
                n_cd = int(np.isfinite(Y[:, js]).sum())
                neff_day = m / (1 + (m - 1) * rho)
                out[f"{market}/{split}/{g}"] = {
                    "n_days": int(len(days)), "n_assets": m, "n_coin_days": n_cd, "n_pairs": npair,
                    "rho_labels": rho, "rho_logrv_minus_ref": rho_c, "n_eff_per_day": neff_day,
                    "base_rate": float(np.nanmean(Y[:, js]))}
                print(market, split, g, json.dumps(out[f"{market}/{split}/{g}"]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
