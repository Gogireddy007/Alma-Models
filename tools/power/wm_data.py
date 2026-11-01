"""Data for the power world-model search (architect's own loader; reads the lead's raw Energy-Charts JSON).

Leakage guards (asserted):
  * only raw monthly files with month <= the split's last month are opened (DEV 2025-06, VAL 2025-10); nothing after
    2025-10 is ever read, and every timestamp after the split end is dropped and asserted absent;
  * batch(...) for target day d uses prices of days <= d-1 and fundamentals (load, wind, solar, residual load
    actuals) of days <= d-2 only; the calendar of day d is known in advance (weekday, German public holidays).
    The aux targets of the physical model (realised residual load of d-1 and d) live in a separate `tgt` dict and
    are used in the training loss only.

Daily matrices are local (Europe/Berlin) delivery days x 24 hours. DST: the 23-hour day gets the missing hour linearly
interpolated; the 25-hour day averages the doubled hour. From 2025-10-01 the price is 15-minute; hourly mean is used.
DE-AT-LU (to 2018-09-30) and DE-LU (from 2018-10-01) are concatenated (the model sees a z-normalised frame).
"""
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "power", "raw")
MYRAW = os.path.join(ROOT, "runs", "power", "search", "raw")


def raw_dirs(split):
    """Architect's split-limited fetch (tools/power/search/fetch_dev.py): DEV files were requested with end <= 2025-06-30;
    the VAL months live in a separate directory read only by split='val'."""
    return [os.path.join(MYRAW, "dev")] + ([os.path.join(MYRAW, "val")] if split == "val" else [])
SPLIT_END = {"dev": "2025-06-30", "val": "2025-10-31"}
HARD_END = "2025-10-31"
TZ = "Europe/Berlin"


def _months_ok(path, end):
    m = os.path.basename(path)[:7]
    return m <= end[:7]


def _read_series(name, end, field=None, dirs=None):
    out = []
    files = sorted(sum([glob.glob(os.path.join(r, name, "*.json")) for r in (dirs or [RAW])], []),
                   key=os.path.basename)
    for p in files:
        if not _months_ok(p, end):
            continue
        assert os.path.basename(p)[:7] <= HARD_END[:7]
        d = json.load(open(p))
        ts = pd.to_datetime(np.asarray(d["unix_seconds"], dtype=np.int64), unit="s", utc=True)
        if field is None:
            out.append(pd.Series(np.asarray(d["price"], dtype=float), index=ts))
        else:
            cols = {}
            for pt in d["production_types"]:
                if pt["name"] in field:
                    cols[pt["name"]] = np.asarray([np.nan if v is None else v for v in pt["data"]], dtype=float)
            out.append(pd.DataFrame(cols, index=ts))
    s = pd.concat(out)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def _daily(s, end):
    """UTC-indexed series -> [N, 24] local-day matrix + dates (datetime.date)."""
    s = s.tz_convert(TZ)
    s = s[s.index < pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)]
    h = s.groupby([s.index.date, s.index.hour]).mean()        # 25-h day: doubled hour averaged
    days = sorted(set(h.index.get_level_values(0)))
    M = np.full((len(days), 24), np.nan)
    di = {d: i for i, d in enumerate(days)}
    for (d, hr), v in h.items():
        M[di[d], hr] = v
    for i in range(len(days)):                                  # 23-h day / gaps: interpolate within the day
        r = M[i]
        if np.isnan(r).any() and (~np.isnan(r)).sum() >= 12:
            x = np.arange(24)
            ok = ~np.isnan(r)
            M[i] = np.interp(x, x[ok], r[ok])
    return M, days


def german_holidays(years):
    """National German public holidays (fixed + Easter-based) and the main regional ones that close much industry
    (Corpus Christi, Reformation Day, All Saints). Known years in advance."""
    from dateutil.easter import easter
    hol = set()
    for y in years:
        e = easter(y)
        for d in [dt.date(y, 1, 1), dt.date(y, 5, 1), dt.date(y, 10, 3), dt.date(y, 12, 25), dt.date(y, 12, 26),
                  e - dt.timedelta(days=2), e + dt.timedelta(days=1), e + dt.timedelta(days=39),
                  e + dt.timedelta(days=50), e + dt.timedelta(days=60), dt.date(y, 10, 31), dt.date(y, 11, 1),
                  dt.date(y, 1, 6), dt.date(y, 12, 24), dt.date(y, 12, 31)]:
            hol.add(d)
    return hol


def calendar_feats(days):
    hol = german_holidays(range(days[0].year, days[-1].year + 2))
    out = np.zeros((len(days), 12), dtype=np.float32)
    for i, d in enumerate(days):
        out[i, d.weekday()] = 1
        out[i, 7] = d in hol
        out[i, 8] = (d - dt.timedelta(days=1) in hol or d + dt.timedelta(days=1) in hol) and d.weekday() < 5
        doy = d.timetuple().tm_yday
        out[i, 9] = np.sin(2 * np.pi * doy / 365.25)
        out[i, 10] = np.cos(2 * np.pi * doy / 365.25)
        out[i, 11] = (d.year - 2020) / 5
    return out


FUND_SCALE = np.array([60000.0, 30000.0, 20000.0, 40000.0], dtype=np.float32)   # load, wind, solar, residual (MW)


def _smard(split, name, end):
    d = json.load(open(os.path.join(ROOT, "runs", "power", "search", "smard", split, name + ".json")))
    assert d["end_exclusive"] <= "2025-11-01"
    ts = pd.to_datetime(np.asarray(d["unix_ms"], dtype=np.int64), unit="ms", utc=True)
    s = pd.Series(np.asarray(d["value"], dtype=float), index=ts)
    assert s.index.max() < pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1), "data past split end"
    return s


class PowerData:
    """source='smard' (default; runs/power/search/smard/<split>/, truncated at fetch time) or 'ec' (lead raw)."""

    def __init__(self, split="dev", source="smard", start="2018-10-01"):
        end = SPLIT_END[split]
        assert end <= HARD_END
        self.split, self.end = split, end
        if source == "smard":
            price = _smard(split, "price", end)
            fund = pd.DataFrame({"load": _smard(split, "load", end),
                                 "wind": _smard(split, "wind_on", end) + _smard(split, "wind_off", end),
                                 "solar": _smard(split, "solar", end)})
        else:
            dirs = [RAW]
            p_old = _read_series("price_de_at_lu", end, dirs=dirs)
            p_new = _read_series("price_de_lu", end, dirs=dirs)
            p_old = p_old[p_old.index < pd.Timestamp("2018-09-30 22:00", tz="UTC")]
            price = pd.concat([p_old, p_new]).sort_index()
            price = price[~price.index.duplicated(keep="last")]
            pp = _read_series("public_power", end, field={"Load", "Wind onshore", "Wind offshore", "Solar"}, dirs=dirs)
            fund = pd.DataFrame({"load": pp["Load"],
                                 "wind": pp[["Wind onshore", "Wind offshore"]].sum(1, min_count=1),
                                 "solar": pp["Solar"].fillna(0.0)})
        P, days = _daily(price, end)
        keep = [i for i, d in enumerate(days) if d >= dt.date.fromisoformat(start)]
        P, days = P[keep], [days[i] for i in keep]
        Fm = []
        for c in ["load", "wind", "solar"]:
            M, fd = _daily(fund[c].dropna(), end)
            fi = {d: i for i, d in enumerate(fd)}
            A = np.full((len(days), 24), np.nan)
            for i, d in enumerate(days):
                if d in fi:
                    A[i] = M[fi[d]]
            Fm.append(A)
        Fm.append(Fm[0] - Fm[1] - Fm[2])
        self.P = P.astype(np.float32)
        self.F = np.stack(Fm, -1).astype(np.float32)               # [N,24,4] MW
        self.days = days
        assert days[-1] <= dt.date.fromisoformat(end), (days[-1], end)
        assert all((days[i + 1] - days[i]).days == 1 for i in range(len(days) - 1)), "non-contiguous days"
        self.cal = calendar_feats(days)
        self.di = {d: i for i, d in enumerate(days)}

    def idx(self, date):
        return self.di[dt.date.fromisoformat(date) if isinstance(date, str) else date]


def batch_from_arrays(P, F, cal, cal_d, L=14):
    """Pure feature builder used both by the search and by the harness wrapper.
    P   [B, L, 24]  prices of days d-L..d-1 (raw EUR/MWh)
    F   [B, L, 24, 4] fundamentals of days d-L-1..d-2 (raw MW), i.e. F[:, j] belongs to the day BEFORE price day j
    cal [B, L, 12]  calendar of the price days d-L..d-1; cal_d [B, 12] calendar of day d.
    Returns the model input dict (torch) and (mu, sd) of the z-frame."""
    import torch
    P = np.asarray(P, dtype=np.float32)
    last7 = P[:, -7:].reshape(P.shape[0], -1)
    mu = np.nanmean(last7, 1)
    sd = np.nanstd(last7, 1) + 5.0
    zp = (P - mu[:, None, None]) / sd[:, None, None]
    zp = np.nan_to_num(np.clip(zp, -15, 15))
    fu = np.nan_to_num(np.asarray(F, dtype=np.float32) / FUND_SCALE)
    b = {"zp": torch.tensor(zp), "fu": torch.tensor(fu), "cal": torch.tensor(np.asarray(cal, np.float32)),
         "cal_d": torch.tensor(np.asarray(cal_d, np.float32))}
    return b, mu, sd


def batch(D, idx, L=14, with_target=True):
    """Origins = target day indices idx (np array). Asserts the information set."""
    idx = np.asarray(idx)
    assert (idx >= L + 1).all()
    pj = idx[:, None] + np.arange(-L, 0)[None]                   # price days d-L..d-1
    fj = pj - 1                                                  # fundamentals days d-L-1..d-2
    assert (pj.max(1) == idx - 1).all() and (fj.max(1) == idx - 2).all()   # information-set assertion
    b, mu, sd = batch_from_arrays(D.P[pj], D.F[fj], D.cal[pj], D.cal[idx], L)
    if not with_target:
        return b, mu, sd, None
    import torch
    y = (D.P[idx] - mu[:, None]) / sd[:, None]
    tgt = {"y": torch.tensor(np.nan_to_num(np.clip(y, -15, 15))),
           "ymask": torch.tensor(~np.isnan(D.P[idx])),
           "rl": torch.tensor(np.nan_to_num(np.concatenate([D.F[idx - 1, :, 3], D.F[idx, :, 3]], 1) / FUND_SCALE[3]))}
    return b, mu, sd, tgt


def valid_days(D, lo, hi, L=14):
    """Indices of target days in [lo, hi] (dates) with complete target prices and enough history."""
    lo, hi = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
    out = []
    for i, d in enumerate(D.days):
        bad = np.isnan(D.P[i]).any() or (hasattr(D, "_nan_target") and D._nan_target[i])
        if lo <= d <= hi and i >= L + 1 and not bad:
            out.append(i)
    return np.array(out)


class LeadData(PowerData):
    """The lead's panel (tools/power/data.train_panel: Energy-Charts, delivery days <= 2025-06-30, raises otherwise),
    same layout as PowerData; used for the final refit. Spring-forward slot interpolated in INPUT prices only (the
    target of that day stays NaN, so valid_days() drops it from training)."""

    def __init__(self, start="2018-10-01"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("power_data_lead", os.path.join(ROOT, "tools", "power", "data.py"))
        LD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(LD)
        dates, P, C = LD.train_panel()
        assert str(dates.max()) <= SPLIT_END["dev"]
        keep = dates >= np.datetime64(start, "D")
        dates, P, C = dates[keep], P[keep], C[keep]
        self.split, self.end = "dev", SPLIT_END["dev"]
        self.Ptarget = P.astype(np.float32)
        Pin = np.array([LD.interp_dst_gap(r) for r in P], np.float32)
        F = np.stack([C[..., 0], C[..., 1] + C[..., 2], C[..., 3], C[..., 0] - C[..., 1] - C[..., 2] - C[..., 3]], -1)
        self.P = Pin
        self.F = F.astype(np.float32)
        self.days = [dt.date.fromisoformat(str(x)) for x in dates]
        assert all((self.days[i + 1] - self.days[i]).days == 1 for i in range(len(self.days) - 1))
        self.cal = calendar_feats(self.days)
        self.di = {d: i for i, d in enumerate(self.days)}
        self._nan_target = np.isnan(self.Ptarget).any(1)
