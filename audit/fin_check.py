"""Independent re-score of the frozen Alma-Fin forecasters on the crypto TEST year.

Forecasts come from the repo harness loop (tools/fin/harness.run: builds the information-set-safe context per day,
asserted), written to audit/fin_test/ (never the repo's runs/). Labels and metrics are recomputed here from the
sealed daily parquet: y_vol(D) = 1[log RV(D) > mean log RV(D-7..D-1)], P1 = accuracy of p_up > 0.5, QS = mean
pinball of the 13 log-RV quantiles at 0.1..0.9.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "fin"))
import run_eval as R   # noqa: E402

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["wm_fin", "wm_direct", "har_intra_dow", "har_dow",
                                                          "lgbm_dow", "har_roll", "lgbm_asset"]
R.RUN_DIR["test"] = os.path.join(HERE, "fin_test")
R.forecast("test", "crypto", MODELS)

QS = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])
I9 = slice(2, 11)
d = pd.concat([pd.read_parquet(os.path.join(REPO, "data/fin/crypto_daily.parquet")),
               pd.read_parquet(os.path.join(REPO, "data/fin/crypto_daily_SEALED.parquet"))])
print(d.columns.tolist()[:30])
uni = pd.read_json(os.path.join(REPO, "data/fin/universe.json"), typ="series")
held = set(uni["heldout"])
rows = []
for mn in MODELS:
    z = np.load(os.path.join(HERE, "fin_test", "crypto", f"{mn}.npz"), allow_pickle=True)
    rows.append((mn, {k: z[k].shape for k in z.files}))
for r in rows:
    print(r)
