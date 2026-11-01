"""Red-team INDEPENDENT reference calendars for Alma-Live Gas (written without reading the data lead's code).

Every function returns tz-aware UTC timestamps (the time from which a value can be known by a live trader).
They are deliberately conservative; the store's published_at must be >= these (a store timestamp EARLIER than the
reference is a candidate leak, LATER is merely conservative).

  eia_release_utc(week_end)        WNGSR: Thu 10:30 ET after the week ending Fri; holiday overrides from the EIA
                                   schedule page (data/gas/raw/ngs_schedule.html) with their own times.
  cot_release_utc(asof_tuesday)    CFTC COT: Fri 15:30 ET for Tue positions; holiday week -> next business day.
  ng_last_trade(contract_month)    NYMEX NG: 3 business days before the first calendar day of the delivery month.
  ng_front(d)                      front contract at trade date d, CALENDAR rule only (no volume / OI).
  ng_settle_utc(d)                 NG daily settlement: 14:30 ET (settlement window 14:28-14:30) on trade day d.
  etf_close_utc(d)                 UNG/BOIL close 16:00 ET.
  era5_avail_utc(day)              Open-Meteo archive (ERA5/ERA5T) value for `day`: >= day+5 00:00 UTC (realized,
                                   never a forecast).
  hh_spot_eia_avail_utc(d)         EIA/FRED Henry Hub daily spot (RNGWHHD/DHHNGSP) is published WEEKLY (EIA NGW,
                                   Wednesdays/Thursdays); release Wednesday F+5 (16:00 ET assumed).
"""
import datetime as dt
import os
import re
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCHED = os.path.join(ROOT, "data/gas/raw/ngs_schedule.html")
EXPOSED = pd.Timestamp("2025-11-01", tz=UTC)


def _et(day, hh, mm=0):
    d = pd.Timestamp(day)
    return pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)).tz_convert(UTC)


def _fed_holidays():
    from pandas.tseries.holiday import USFederalHolidayCalendar
    h = set(USFederalHolidayCalendar().holidays("1990-01-01", "2030-12-31"))
    h |= {pd.Timestamp("2025-01-09"), pd.Timestamp("2018-12-05"), pd.Timestamp("2007-01-02")}  # days of mourning
    return h


def _nymex_holidays():
    """NYSE-style exchange holidays (no Columbus / Veterans; adds Good Friday, mourning days)."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                        USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
                                        nearest_workday)

    class C(AbstractHolidayCalendar):
        rules = [Holiday("NY", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay,
                 GoodFriday, USMemorialDay,
                 Holiday("J19", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
                 Holiday("J4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
                 Holiday("XM", month=12, day=25, observance=nearest_workday)]
    h = set(C().holidays("1990-01-01", "2030-12-31"))
    h |= {pd.Timestamp("2025-01-09"), pd.Timestamp("2018-12-05")}
    return h


FED = _fed_holidays()
NYMEX = _nymex_holidays()


def is_bday(d, hol=None):
    d = pd.Timestamp(d).normalize().tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d).normalize()
    return d.weekday() < 5 and d not in (NYMEX if hol is None else hol)


HOLIDAY_RULES = {  # name on the schedule page -> function(year) -> holiday date
    "New Year": lambda y: pd.Timestamp(f"{y}-01-01"),
    "Mourning": lambda y: pd.Timestamp("2025-01-09"),
    "Juneteenth": lambda y: pd.Timestamp(f"{y}-06-19"),
    "Independence Day": lambda y: pd.Timestamp(f"{y}-07-04"),
    "Veterans": lambda y: pd.Timestamp(f"{y}-11-11"),
    "Thanksgiving": lambda y: pd.Timestamp(f"{y}-11-01") + pd.offsets.WeekOfMonth(week=3, weekday=3),
    "Christmas": lambda y: pd.Timestamp(f"{y}-12-25"),
}


def schedule_overrides():
    """{standard Thursday it replaces: (alternate release date, hh, mm, holiday)} from the EIA schedule page.
    Pairing is by the holiday NAME (nearest instance), not by date proximity (Dec 29 2025 replaces Thu Dec 25)."""
    txt = re.sub(r"<[^>]+>", " ", open(SCHED, encoding="latin-1").read())
    txt = " ".join(txt.replace("&nbsp;", " ").split())
    out = {}
    pat = (r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), "
           r"(\d{4})(?: - \(Updated\))? (Monday|Tuesday|Wednesday|Thursday|Friday) (\d{1,2}):(\d{2}) ([ap])\.m\. "
           r"([A-Z][^0-9]*?)(?= (?:January|February|March|April|May|June|July|August|September|October|November|December) |EIA|$)")
    for m in re.finditer(pat, txt):
        d = pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}")
        hh = int(m.group(5)) % 12 + (12 if m.group(7) == "p" else 0)
        name = m.group(8).strip()
        key = next(k for k in HOLIDAY_RULES if k in name)
        hol = min((HOLIDAY_RULES[key](y) for y in (d.year - 1, d.year, d.year + 1)), key=lambda h: abs((h - d).days))
        std = hol + pd.Timedelta(days=3 - hol.weekday())  # Thursday of the holiday's week
        out[std] = (d, hh, int(m.group(6)), name)
    return out


_OVR = None


def eia_release_utc(week_end):
    """Release time of the WNGSR covering the week ending Friday `week_end`.

    Returns (utc timestamp, rule). rule in {standard, schedule, holiday_unverified}. holiday_unverified = a federal
    holiday falls Mon-Thu of the release week outside the schedule page's coverage: real release may be LATER
    (e.g. Friday) -> conservative answer Friday 12:00 ET (or next business day)."""
    global _OVR
    if _OVR is None:
        _OVR = schedule_overrides()
    F = pd.Timestamp(week_end).normalize()
    assert F.weekday() == 4, f"week_end {F.date()} is not a Friday"
    std = F + pd.Timedelta(days=6)
    if std in _OVR:
        a, hh, mm, _ = _OVR[std]
        return _et(a, hh, mm), "schedule"
    covered = False  # page lists only selected holidays; any uncovered holiday week is unverified
    wk = [F + pd.Timedelta(days=k) for k in range(4, 7)]  # Tue..Thu (Monday holidays historically do not shift)
    if any(d in FED for d in wk) and not covered:
        d = std + pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in FED:
            d += pd.Timedelta(days=1)
        return _et(d, 12, 0), "holiday_unverified"
    return _et(std, 10, 30), "standard"


def cot_release_utc(asof_tuesday):
    T = pd.Timestamp(asof_tuesday).normalize()
    fri = T + pd.Timedelta(days=3)
    if any((T + pd.Timedelta(days=k)) in FED for k in range(0, 4)):
        d = fri + pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in FED:
            d += pd.Timedelta(days=1)
        return _et(d, 15, 30)
    return _et(fri, 15, 30)


def ng_last_trade(contract_month):
    """contract_month: any date in the delivery month."""
    m = pd.Timestamp(contract_month).to_period("M").start_time
    d, n = m, 0
    while n < 3:
        d -= pd.Timedelta(days=1)
        if is_bday(d):
            n += 1
    return d


def ng_front(d, offset=0):
    """Delivery month (Period) of the front (offset=0) / next contract at trade date d: calendar rule only."""
    d = pd.Timestamp(d).normalize()
    p = (d + pd.offsets.MonthBegin(0)).to_period("M") if d.day == 1 else (d.to_period("M") + 1)
    p = d.to_period("M")
    while ng_last_trade(p.start_time) < d:
        p += 1
    return p + offset


def ng_settle_utc(d):
    return _et(d, 14, 30)


def etf_close_utc(d):
    return _et(d, 16, 0)


def era5_avail_utc(day):
    return pd.Timestamp(day).normalize().tz_localize(UTC) + pd.Timedelta(days=5)


def hh_spot_eia_avail_utc(d):
    """EIA posts RNGWHHD weekly with 'Release Date' on Wednesdays (time of day not stated): Wednesday after the
    trade date's week, 16:00 ET (Fri F -> F+5)."""
    d = pd.Timestamp(d).normalize()
    fri = d + pd.Timedelta(days=max(4 - d.weekday(), 0))
    return _et(fri + pd.Timedelta(days=5), 16, 0)


if __name__ == "__main__":
    print("schedule overrides:", {str(k.date()): (str(v[0].date()),) + v[1:] for k, v in sorted(schedule_overrides().items())})
    for F in ["2025-11-07", "2025-11-21", "2025-12-19", "2025-12-26", "2026-01-16", "2026-02-13", "2026-05-22",
              "2026-06-12", "2026-06-26", "2026-08-28", "2026-10-09", "2026-11-06", "2026-03-06", "2026-03-13"]:
        t, r = eia_release_utc(F)
        print("WNGSR week", F, "->", t.tz_convert(ET).strftime("%a %Y-%m-%d %H:%M %Z"), t.strftime("%H:%MZ"), r)
    for d in ["2025-11-25", "2026-01-26", "2026-04-27", "2026-10-28"]:
        print("front", d, ng_front(d), "LTD", ng_last_trade(ng_front(d).start_time).date())
