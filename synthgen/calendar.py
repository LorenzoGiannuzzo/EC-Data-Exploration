"""Calendar.

Day types, seasons, Italian national holidays and the two civil days that do not
hold 96 quarter-hours. The rules are the ones the SLP pipeline already applies,
so that a day classified there and a day classified here cannot disagree.

The daylight saving days matter twice over. On the measured side they are removed,
since a 92 or 100 quarter day cannot be laid on a 96 slot grid. On the generated
side they must nevertheless be produced, because a synthetic year that skips them
is not a civil year: what the generator writes for those two days is described in
`generate.py`, and the timestamps in the output carry the truth.
"""
from __future__ import annotations

import pandas as pd

DAYTYPES = ["weekday", "saturday", "sunday"]


def easter_monday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm, then one day."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)


def italian_holidays(year: int) -> set[pd.Timestamp]:
    """Fixed national holidays plus Easter Monday."""
    fixed = [(1, 1), (1, 6), (4, 25), (5, 1), (6, 2),
             (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)]
    days = {pd.Timestamp(year=year, month=m, day=d) for m, d in fixed}
    days.add(easter_monday(year))
    return days


def dst_days(year: int) -> list[pd.Timestamp]:
    """Last Sunday of March, 92 quarter-hours, and last Sunday of October, 100."""
    out = []
    for month in (3, 10):
        d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        while d.dayofweek != 6:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return out


def day_type(dates: pd.Series, holidays: set[pd.Timestamp]) -> pd.Series:
    """weekday, saturday, sunday, with holidays counted as Sunday."""
    dow = dates.dt.dayofweek
    out = pd.Series("weekday", index=dates.index, dtype=object)
    out[dow == 5] = "saturday"
    out[dow == 6] = "sunday"
    out[dates.isin(holidays)] = "sunday"
    return out


def season_map(seasons: dict) -> dict[int, str]:
    return {m: label for label, months in seasons.items() for m in months}


def season(dates: pd.Series, mapping: dict[int, str]) -> pd.Series:
    return dates.dt.month.map(mapping)


def annotate(days: pd.DataFrame, seasons: dict,
             date_col: str = "date") -> pd.DataFrame:
    """Add daytype, season and month to a frame that carries a date column."""
    years = sorted(days[date_col].dt.year.unique().tolist())
    hol: set[pd.Timestamp] = set()
    for y in years:
        hol |= italian_holidays(int(y))
    out = days.copy()
    out["daytype"] = day_type(out[date_col], hol)
    out["season"] = season(out[date_col], season_map(seasons))
    out["month"] = out[date_col].dt.month
    return out