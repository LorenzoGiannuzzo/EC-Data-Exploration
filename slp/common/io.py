"""Reading the monthly folders.

The source layout is one folder per month, named <3-letter Italian month><2-digit
year> in mixed case (ago24, Ago25, Apr24, ...), each holding:

    Metadati POD <mesYY>.xlsx     POD, CCATETE, D_49DES, FDESC, TATE3DES
    misure_<anything>.csv         POD, DataMisura, Q1..Q96, PotenzaContrattuale

Nothing about the encoding, the field separator or the decimal mark is assumed:
the files come from a distributor's export and all three have been seen to vary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

Q_COLS = [f"Q{i}" for i in range(1, 97)]

MONTHS_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}
DIR_RE = re.compile(r"^([a-zA-Z]{3})(\d{2})$")


# ── folder scan ──────────────────────────────────────────────────────────────
def scan_data_dir(root: Path) -> pd.DataFrame:
    """Return one row per monthly folder found, with its parsed month and year."""
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = DIR_RE.match(d.name)
        if not m:
            continue                              # _cache/ and anything else
        mon = m.group(1).lower()
        if mon not in MONTHS_IT:
            continue
        meta = _find_one(d, ["Metadati*.xlsx", "metadati*.xlsx", "*.xlsx"])
        meas = _find_one(d, ["misure*.csv", "Misure*.csv", "*.csv"])
        rows.append({
            "folder": d.name,
            "path": d,
            "month": MONTHS_IT[mon],
            "year": 2000 + int(m.group(2)),
            "meta_file": meta,
            "meas_file": meas,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No monthly folder matching <mesYY> under {root}")

    # Two folders for the same month (say Feb24 and feb24) would both be read and
    # every day would be counted twice.
    dup = df.groupby(["year", "month"]).size()
    dup = dup[dup > 1]
    if len(dup):
        for (y, m), n in dup.items():
            names = df.loc[(df["year"] == y) & (df["month"] == m), "folder"].tolist()
            print(f"  WARNING: {n} folders for {m:02d}/{y}: {names} — all will be read")

    return df.sort_values(["year", "month"]).reset_index(drop=True)


def _find_one(folder: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


# ── robust readers ───────────────────────────────────────────────────────────
def _sniff_csv(path: Path) -> tuple[str, str, str]:
    """Guess (encoding, separator, decimal) from the first two lines."""
    head = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                head = [f.readline() for _ in range(2)]
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if head is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"cannot decode {path}")

    header = head[0]
    sep = max([";", ",", "\t", "|"], key=header.count)
    body = head[1] if len(head) > 1 and head[1] else ""
    # If the separator is ';' a comma can only be the decimal mark.
    decimal = "," if (sep == ";" and "," in body) else "."
    return encoding, sep, decimal


def to_number(s: pd.Series) -> pd.Series:
    """Numeric cast that survives a comma decimal mark.

    The export is not internally consistent: the Consumo columns use a dot while
    PotenzaContrattuale uses a comma, in the same file. pd.to_numeric('4,5')
    returns NaN, so every POD on a fractional contracted power would silently
    lose its threshold while the integer ones kept theirs.
    """
    if s.dtype.kind in "if":
        return s
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce")


def read_measurements(path: Path, date_col: str = "DataMisura",
                      date_format: str | None = "%d/%m/%Y",
                      kind_col: str | None = "Tipologia",
                      kind_keep: str = "AP",
                      k_col: str | None = "K") -> pd.DataFrame:
    """Read one monthly measurement CSV, keeping one row per POD-day.

    The file carries several quantities per POD-day, distinguished by
    `Tipologia`: AP is the active energy withdrawn, RLP the inductive reactive,
    AN the active energy injected, RLN and RCN the reactive injected. Only AP is
    a consumption curve; the others are different physical quantities and would
    be read as extra days if left in.

    The readings are meter counts, not energy, and `K` is the constant that turns
    the one into the other. It is 1 for most points and 20, 25 or 40 for those
    metered through a current transformer, which are the largest of the archive:
    on the February 2024 file their peak reads 2 to 4 per cent of their own
    contractual power without the constant and 48 to 72 per cent with it, against
    a median of 14 per cent over the whole population. Leaving it out understates
    those points by that same factor, and since annual energy is a scale feature
    of the second clustering stage it also places them in the wrong profile. It is
    read per row, because a POD can change it inside a single month, and taken in
    absolute value, since on the injected rows the sign carries the direction
    rather than a scale.
    """
    encoding, sep, decimal = _sniff_csv(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding,
                     low_memory=False)
    df = _normalise_columns(df)

    qs = _q_columns(df)
    if len(qs) < 96:
        raise ValueError(f"{path.name}: found {len(qs)} quarter-hour columns, expected at least 96")

    df[qs] = df[qs].apply(to_number).astype("float32")

    kc = _match(df, k_col) if k_col else None
    k_applied = None
    if kc is not None:
        k = to_number(df[kc]).abs()
        k = k.where(np.isfinite(k) & (k > 0), 1.0).astype("float32")
        df[qs] = df[qs].to_numpy() * k.to_numpy()[:, None]
        k_applied = k

    if len(qs) > 96:
        # Columns Q97..Q100 exist because the day on which DST ends has 25 hours.
        # On every other day they are present and filled with zeros rather than
        # left empty, so a null test counts them as used on every row: what marks
        # real use is a non-zero value. The 25-hour day is excluded in Section 2.2
        # anyway, so the surplus is dropped here.
        extra = qs[96:]
        used = (df[extra].fillna(0) != 0).any(axis=1)
        df.attrs["dst_surplus_rows"] = int(used.sum())
        df = df.drop(columns=extra)
        qs = qs[:96]

    dcol = _match(df, date_col)
    if dcol is None:
        raise ValueError(f"{path.name}: no date column matching '{date_col}'")
    df[dcol] = pd.to_datetime(df[dcol], format=date_format, errors="coerce")
    if df[dcol].isna().all():                     # the declared format did not apply
        df[dcol] = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce")
    df = df.rename(columns={dcol: "date"})

    pcol = _match(df, "POD")
    if pcol is None:
        raise ValueError(f"{path.name}: no POD column")
    df = df.rename(columns={pcol: "pod"})
    df["pod"] = df["pod"].astype(str).str.strip()

    # ── one quantity per POD-day ─────────────────────────────────────────────
    kcol = _match(df, kind_col) if kind_col else None
    if kcol is not None:
        kinds = df[kcol].astype(str).str.strip().str.upper()
        # PODs that inject: they carry AN rows. This is the only way to identify
        # a prosumer here, since the readings are withdrawals and never negative.
        df.attrs["prosumer_pods"] = set(df.loc[kinds.str.startswith("AN"), "pod"])
        df.attrs["kind_counts"] = kinds.value_counts().to_dict()
        df = df[kinds == kind_keep.upper()].copy()
        if df.empty:
            raise ValueError(
                f"{path.name}: no row with {kind_col}='{kind_keep}'. "
                f"Present: {sorted(df.attrs['kind_counts'])}")
    else:
        df.attrs["prosumer_pods"] = set()
        df.attrs["kind_counts"] = {}

    # counted after the filter, so the figure reported is the one that reaches the
    # pipeline rather than the one in the file
    if k_applied is not None:
        kk = k_applied.reindex(df.index)
        df.attrs["meter_constant_rows"] = int((kk > 1).sum())
        df.attrs["meter_constant_values"] = {
            float(v): int(n) for v, n in kk[kk > 1].value_counts().items()}
    else:
        df.attrs["meter_constant_rows"] = 0
        df.attrs["meter_constant_values"] = {}

    return df


def read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = _normalise_columns(df)
    pcol = _match(df, "POD")
    if pcol is None:
        raise ValueError(f"{path.name}: no POD column")
    df = df.rename(columns={pcol: "pod"})
    df["pod"] = df["pod"].astype(str).str.strip()
    return df


# ── column helpers ───────────────────────────────────────────────────────────
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _match(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column lookup."""
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def _q_columns(df: pd.DataFrame) -> list[str]:
    """The quarter-hour columns, in numeric order, however they are spelled.

    The export carries one column per quarter-hour of the longest civil day, so
    Q1..Q100 rather than Q1..Q96: the day on which DST ends has 25 hours. On
    every other day the last four are empty. Three digits are therefore matched,
    not two, or Q100 would be missed and the count would come out at 99.
    """
    found = {}
    for c in df.columns:
        m = re.fullmatch(r"[Qq]\s*(\d{1,3})", str(c).strip())
        if m:
            found[int(m.group(1))] = c
    return [found[i] for i in sorted(found)]


# ── ATECO hierarchy ──────────────────────────────────────────────────────────
def split_ateco(code: str | float) -> tuple[str | None, str | None, str | None]:
    """Split CCATETE into its three levels.

    Beware of the naming: these follow the source column, not NACE.
        l1 = 2 chars  -> NACE Division   ('47', 'DO')
        l2 = 4 chars  -> NACE Class      ('47.11', 'DO.01')
        l3 = 6 chars  -> NACE Subcategory('47.11.10')
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None, None, None
    parts = str(code).strip().split(".")
    l1 = parts[0] if parts and parts[0] else None
    l2 = ".".join(parts[:2]) if len(parts) >= 2 else None
    l3 = ".".join(parts[:3]) if len(parts) >= 3 else None
    return l1, l2, l3


# ── year assembly ────────────────────────────────────────────────────────────
def load_year(root: Path, year: int, meas_date_col: str = "DataMisura",
              date_format: str | None = "%d/%m/%Y",
              kind_col: str | None = "Tipologia", kind_keep: str = "AP",
              months: list[tuple[int, int]] | None = None,
              verbose: bool = True,
              k_col: str | None = "K") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the monthly folders into one measurement frame plus one
    metadata frame (last observation per POD wins).

    `months`, when given, is an explicit list of (year, month) and overrides
    `year`: it is what allows a rolling window across two calendar years.
    """
    idx = scan_data_dir(root)
    if months:
        want = set(months)
        idx = idx[[(y, m) in want for y, m in zip(idx["year"], idx["month"])]]
        label = f"{len(idx)} months"
    elif year is not None:
        idx = idx[idx["year"] == year]
        label = str(year)
    else:
        label = "every month available"      # no truncation
    if idx.empty:
        available = sorted(scan_data_dir(root)["year"].unique())
        raise ValueError(f"No folder for {label}. Available years: {available}")

    meas_parts, meta_parts = [], []
    surplus_rows = 0
    k_rows = 0
    k_values: dict[float, int] = {}
    prosumers: set[str] = set()
    kinds: dict[str, int] = {}
    for _, r in idx.iterrows():
        if r["meas_file"] is not None:
            m = read_measurements(r["meas_file"], meas_date_col, date_format,
                                  kind_col, kind_keep, k_col)
            surplus_rows += m.attrs.get("dst_surplus_rows", 0)
            k_rows += m.attrs.get("meter_constant_rows", 0)
            for v, n in m.attrs.get("meter_constant_values", {}).items():
                k_values[v] = k_values.get(v, 0) + int(n)
            prosumers |= m.attrs.get("prosumer_pods", set())
            for k, v in m.attrs.get("kind_counts", {}).items():
                kinds[k] = kinds.get(k, 0) + int(v)
            meas_parts.append(m)
            if verbose:
                note = ""
                if m.attrs.get("dst_surplus_rows"):
                    note = f"  ({m.attrs['dst_surplus_rows']} rows past Q96, DST)"
                if m.attrs.get("meter_constant_rows"):
                    note += f"  [{m.attrs['meter_constant_rows']:,} rows with K > 1]"
                print(f"    {r['folder']:8s} {len(m):>8,} rows   {m['pod'].nunique():>6,} PODs{note}")
        if r["meta_file"] is not None:
            meta_parts.append(read_metadata(r["meta_file"]))

    meas = pd.concat(meas_parts, ignore_index=True)
    meas.attrs["dst_surplus_rows"] = surplus_rows
    meas.attrs["meter_constant_rows"] = k_rows
    meas.attrs["meter_constant_values"] = k_values
    meas.attrs["prosumer_pods"] = prosumers
    meas.attrs["kind_counts"] = kinds
    meta = pd.concat(meta_parts, ignore_index=True).drop_duplicates("pod", keep="last")

    if months:
        want = {y * 100 + m for y, m in months}
        code = meas["date"].dt.year * 100 + meas["date"].dt.month
        meas = meas[code.isin(want)].reset_index(drop=True)
    elif year is not None:
        meas = meas[meas["date"].dt.year == year].reset_index(drop=True)
    return meas.reset_index(drop=True), meta