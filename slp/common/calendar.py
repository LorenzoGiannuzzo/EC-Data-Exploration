"""Reference calendar shared by the comparison and mapping stages (Sections 2.5-2.6).

Every object compared in Section 2.5 lives on a different native grid. The data-driven
profiles are defined over season x day type, the ARERA tables over month x day type, and
the GSE profiles over the calendar hour. The only grid all three can be projected onto
without inventing structure is the hourly calendar of a reference year, which is what this
module builds.

The season map and the day type convention are not hardcoded. They are recovered from
the cache written by preprocessing, so that the calendar used here cannot drift from the
one the profiles were estimated on.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DAYTYPES = ("weekday", "saturday", "sunday")


# --------------------------------------------------------------------------- holidays
def easter(year: int) -> dt.date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def italian_holidays(year: int) -> set[dt.date]:
    """National holidays. Patron saint days are local and deliberately excluded."""
    fixed = [(1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15),
             (11, 1), (12, 8), (12, 25), (12, 26)]
    days = {dt.date(year, m, d) for m, d in fixed}
    days.add(easter(year) + dt.timedelta(days=1))  # Easter Monday
    return days


def dst_days(year: int) -> pd.DatetimeIndex:
    """The two days of the year that do not have 24 hours.

    Preprocessing removes them rather than realigning them (Section 2.2), and the same
    exclusion is applied here so that the reference calendar is a rectangular 24-hour grid
    on which the three families of profiles can be compared without an alignment
    convention that would have to be justified separately.
    """
    march = pd.date_range(f"{year}-03-01", f"{year}-03-31", freq="D")
    october = pd.date_range(f"{year}-10-01", f"{year}-10-31", freq="D")
    last_sun = lambda idx: idx[idx.dayofweek == 6][-1]  # noqa: E731
    return pd.DatetimeIndex([last_sun(march), last_sun(october)])


# ----------------------------------------------------------------- recovered from cache
def season_map_from_days(days: pd.DataFrame) -> dict[int, str]:
    """month -> season, read off the preprocessing output.

    Raises if a month carries more than one season label, which would mean the cache was
    built by two different configurations.
    """
    g = days.assign(month=days["date"].dt.month).groupby("month")["season"]
    n = g.nunique()
    bad = n[n > 1]
    if len(bad):
        raise ValueError(f"months with inconsistent season labels: {list(bad.index)}")
    smap = g.agg(lambda s: s.iloc[0]).to_dict()
    missing = sorted(set(range(1, 13)) - set(smap))
    if missing:
        raise ValueError(
            f"months absent from the cache, season undetermined: {missing}. "
            "Extend the observation window or declare the mapping explicitly."
        )
    return {int(k): v for k, v in smap.items()}


def check_daytype_convention(days: pd.DataFrame) -> dict[str, object]:
    """Verify empirically how preprocessing labelled national holidays.

    Returns a small report rather than raising, because the answer belongs in the paper
    (Section 2.2) and not only in the code.
    """
    d = days[["date", "daytype"]].drop_duplicates()
    years = sorted(d["date"].dt.year.unique())
    hol = set()
    for y in years:
        hol |= italian_holidays(int(y))
    d = d.assign(dow=d["date"].dt.dayofweek,
                 is_holiday=d["date"].dt.date.isin(hol))
    weekday_hol = d[d["is_holiday"] & (d["dow"] < 5)]
    folded = int((weekday_hol["daytype"] == "sunday").sum())
    return {
        "weekday_holidays_observed": int(len(weekday_hol)),
        "labelled_sunday": folded,
        "holidays_folded_into_sunday": bool(len(weekday_hol) and folded == len(weekday_hol)),
    }


# ------------------------------------------------------------------------ the calendar
def build_calendar(year: int, season_map: dict[int, str]) -> pd.DataFrame:
    """One row per calendar day of `year`, carrying month, day type and season.

    The day type rule reproduces preprocessing: a national holiday falling on a working
    day is a Sunday, which is also the convention of the tariff bands (Resolution 181/06),
    so the same calendar serves the band split below.
    """
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    idx = idx[~idx.normalize().isin(dst_days(year))]   # see Section 2.2
    hol = italian_holidays(year)
    dow = idx.dayofweek
    daytype = np.where(idx.to_series().dt.date.isin(hol) | (dow == 6), "sunday",
                       np.where(dow == 5, "saturday", "weekday"))
    cal = pd.DataFrame({"date": idx, "month": idx.month, "daytype": daytype})
    cal["season"] = cal["month"].map(season_map)
    cal["cell"] = cal["season"] + "|" + cal["daytype"]
    return cal


def band_of(daytype: str, hour: int) -> int:
    """Tariff band F1/F2/F3 (Resolution 181/06), given the calendar day type above."""
    if daytype == "sunday":
        return 3
    if daytype == "saturday":
        return 2 if 7 <= hour <= 22 else 3
    if 8 <= hour <= 18:
        return 1
    if hour == 7 or 19 <= hour <= 22:
        return 2
    return 3


def hourly_index(cal: pd.DataFrame) -> pd.DataFrame:
    """Expand the day calendar to 8760 rows, one per hour, with band and day type."""
    h = np.tile(np.arange(24), len(cal))
    out = cal.loc[cal.index.repeat(24)].reset_index(drop=True)
    out["hour"] = h
    out["band"] = [band_of(t, x) for t, x in zip(out["daytype"], out["hour"])]
    return out
