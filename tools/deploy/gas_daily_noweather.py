"""Build data/gas/daily.csv WITHOUT the Open-Meteo weather files (deployment helper; see docs/deploy_2026_10_11.md).

The storage world model (wm_gas, wm_gas_ens) and its storage competitors read only the Henry Hub price history `hh`
(and its dates) from daily.csv, via tools/gas/storage.build_context. Panel rows are the EIA Henry Hub trading days
and `hh` is EIA's price; the Open-Meteo weather adds only the hdd/cdd columns used by the separate HH-PRICE task.
Open-Meteo rate-limits the shared egress IP of this environment, so the weather columns are left NaN here.
deploy_gas.py then reproduces the audited TEST forecasts (audit/gas_preds_test.npy) to prove the storage inputs are
unchanged. Do not use this file for the HH-price task.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "gas"))
import data as D  # noqa: E402

hh = D.parse_eia_daily(os.path.join(D.RAW, "rngwhhdd.htm"))
st = D.parse_storage()
st["avail"], st["avail_rule"] = D.release_dates(st["week"], D.parse_schedule())
days = pd.date_range("1997-01-01", hh["date"].max(), freq="D")
wx = pd.DataFrame({"hdd_pw": np.nan, "cdd_pw": np.nan}, index=days)
wx.index.name = "date"
p = D.make_panel(hh, st, wx)
p.to_csv(D.DAILY, index=False, date_format="%Y-%m-%d")
st.to_csv(os.path.join(D.DDIR, "storage.csv"), index=False, date_format="%Y-%m-%d")
open(os.path.join(D.DDIR, "DAILY_CSV_HAS_NO_WEATHER.txt"), "w").write(__doc__)
print(f"daily.csv: {len(p)} rows, {p.date.min().date()} .. {p.date.max().date()}, hh non-NaN {p.hh.notna().sum()}")
