"""Alma-Gas architect feature builder (docs/gas_search.md section 1).

Pure function of ONE harness storage context (storage.build_context): no file access, no global data. The same
function is used for training (contexts of storage.train_weeks()) and in the harness wrapper (tools/gas/wm_gas.py),
so training and inference inputs come from the identical code path.

Output of feats(ctx) (all float32):
  g      [G]        origin-level encoder vector (released weekly history, storage deficit, lagged weekly weather
                    anomalies, target-week calendar, HH price history <= origin, origin kind)
  u      [14, U]    per-day inputs: days 0..6 = week W-1 (Sat..Fri, all OBSERVED, from obs_daily),
                    days 7..13 = target week W (ctx['wx']: observed up to origin-1, CPC forecast after)
  phys   [14, 4]    raw HDD, CDD, causal normal HDD, causal normal CDD per day (for the monotone demand readout)
  a5     [6]        avg5 of the target week per series (deterministic from previous years)
  dsprev [6]        dS_{W-1} per series (the last released week)
Series order = storage.SERIES = [l48, east, midwest, mountain, pacific, south_central].
"""
import hashlib

import numpy as np
import pandas as pd

SERIES = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]
NLAG = 8
_HOL = {}
_NRM = {}


def _holidays(y0, y1):
    key = (y0, y1)
    if key not in _HOL:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        _HOL[key] = set(USFederalHolidayCalendar().holidays(f"{y0}-01-01", f"{y1}-12-31").date)
    return _HOL[key]


def _cal(dates):
    d = pd.to_datetime(dates)
    hol = _holidays(d.min().year - 1, d.max().year + 1)
    doy = 2 * np.pi * d.dayofyear.to_numpy() / 365.25
    dow = 2 * np.pi * d.dayofweek.to_numpy() / 7
    h = np.array([x.date() in hol for x in d], float)
    return np.stack([np.sin(dow), np.cos(dow), np.sin(doy), np.cos(doy), h], 1)


def _normals_from(obs, obs_dates, days, years=9):
    """Causal climatology from the context's own observed daily history: mean of the same month-day in the
    `years` previous years (only dates present in obs_daily, all <= origin-1)."""
    idx = {d: i for i, d in enumerate(obs_dates)}
    out = []
    for x in days:
        v = []
        for y in range(1, years + 1):
            try:
                t = x.replace(year=x.year - y)
            except ValueError:
                t = x.replace(year=x.year - y, day=28)
            j = idx.get(t.strftime("%Y-%m-%d"))
            if j is not None:
                v.append(obs[j])
        out.append(np.nanmean(v, 0) if v else [np.nan, np.nan])
    return np.array(out, float)


def feats(c, scales):
    """scales: dict with 'dev_sd' [6] and 'lvl_sd' [6] (training-set statistics stored in the checkpoint)."""
    dsd = np.asarray(scales["dev_sd"], float)
    lsd = np.asarray(scales["lvl_sd"], float)
    tw = pd.Timestamp(c["target_week"])
    # ------------------------------------------------ previous week W-1 (ended tw-7, all observed at either origin)
    pdays = [tw - pd.Timedelta(days=13 - j) for j in range(7)]
    od = np.asarray(c["obs_daily_dates"]).astype(str)
    obs = np.asarray(c["obs_daily"], float)
    pos = {d: i for i, d in enumerate(od)}
    pw = np.full((7, 2), np.nan)
    for j, x in enumerate(pdays):
        i = pos.get(x.strftime("%Y-%m-%d"))
        if i is not None:
            pw[j] = obs[i]
    key = (str(tw.date()), hashlib.md5(obs.tobytes()).hexdigest(), hashlib.md5(od.tobytes()).hexdigest())
    if key not in _NRM:                      # memo keyed by the exact input bytes (pure-function cache)
        if len(_NRM) > 5000:
            _NRM.clear()
        _NRM[key] = _normals_from(obs, od, pdays)
    pn = _NRM[key]
    # ------------------------------------------------ target week W
    tdays = [pd.Timestamp(x) for x in c["wx_days"]]
    tw_ = np.asarray(c["wx"], float)
    tn = np.asarray(c["wx_normal"], float)
    src = np.asarray(c["wx_src"], int)
    wx = np.concatenate([pw, tw_], 0)
    nm = np.concatenate([pn, tn], 0)
    nm = np.where(np.isfinite(nm), nm, np.nanmean(nm, 0, keepdims=True))
    nm = np.nan_to_num(nm, nan=0.0)
    miss = ~np.isfinite(wx)
    wx = np.where(miss, nm, wx)
    s = np.concatenate([np.zeros(7, int), src])
    obsf = (s == 0).astype(float)
    lead = np.where(s >= 1, s, 0) / 7.0
    cal = _cal(pdays + tdays)
    tgt = np.r_[np.zeros(7), np.ones(7)]
    u = np.concatenate([wx[:, :1] / 30, wx[:, 1:] / 15, nm[:, :1] / 30, nm[:, 1:] / 15,
                        (wx[:, :1] - nm[:, :1]) / 15, (wx[:, 1:] - nm[:, 1:]) / 8,
                        obsf[:, None], lead[:, None], miss.any(1, keepdims=True).astype(float), tgt[:, None], cal], 1)
    phys = np.concatenate([wx, nm], 1)
    # ------------------------------------------------ origin-level encoder vector
    g = []
    a5 = np.array([c["avg5"][k] for k in SERIES], float)
    dsprev = np.array([c["ds_hist"][k][-1] for k in SERIES], float)
    for j, k in enumerate(SERIES):
        dv = np.asarray(c["dev_hist"][k], float)[-NLAG:]
        g += list(np.nan_to_num(dv / dsd[j]))
    g += list(np.nan_to_num(a5 / dsd / 5))
    for j, k in enumerate(SERIES):
        lv = np.asarray(c["lvl_hist"][k], float)
        past = [lv[-1 - 52 * y] for y in range(1, 6) if len(lv) > 52 * y]
        g.append(np.nan_to_num((lv[-1] - np.nanmean(past)) / lsd[j]) if past else 0.0)
    hh_, hc_ = np.asarray(c["hist_hdd"], float), np.asarray(c["hist_cdd"], float)
    for arr, sc in ((hh_, 60.0), (hc_, 30.0)):
        for lag in range(1, 5):
            past = [arr[-lag - 52 * y] for y in range(1, 6) if len(arr) > lag + 52 * y]
            g.append(np.nan_to_num((arr[-lag] - np.nanmean(past)) / sc) if past else 0.0)
    doy = 2 * np.pi * tw.dayofyear / 365.25
    g += [np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy)]
    hh = np.asarray(c["hh"], float)
    hh = hh[np.isfinite(hh) & (hh > 0)]
    g += [np.log(max(hh[-1], 0.01)) - 1.0, np.log(hh[-1] / hh[-21]) if len(hh) > 21 else 0.0]
    g += [1.0 if c["kind"] == "thu" else 0.0]
    f32 = lambda a: np.nan_to_num(np.asarray(a, np.float32))
    return {"g": f32(g), "u": f32(u), "phys": f32(phys), "a5": f32(np.nan_to_num(a5)), "dsprev": f32(dsprev)}


def stack(fs):
    return {k: np.stack([f[k] for f in fs]) for k in fs[0]}
