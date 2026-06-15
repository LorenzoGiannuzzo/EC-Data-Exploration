"""One-off ingestion: read the legacy `data/` folder (monthly subdirs with
   one Metadati Excel + one CSV per month) and bulk-load it into PostgreSQL.

   This is the bridge between the file-based dashboard and the SQL-based one.
   Idempotent — running it twice on the same data is safe (ON CONFLICT DO NOTHING).

   Invoked via the CLI:
       data-explorer ingest --data-dir /data/raw
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

from data_explorer.db.session import engine

console = Console()

# ── Constants matching the legacy dashboard ──────────────────────────────────
Q_COLS_CSV = [f"Q{i}" for i in range(1, 97)]               # in the CSV
Q_COLS_DB  = [f"q{i}" for i in range(1, 97)]               # in the DB

MONTH_REGEX = re.compile(r"^(?P<mon>[a-zA-Z]{3})(?P<yr>\d{2})$")
ITAL_MONTHS = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
    "jan": 1, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "dec": 12,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_dir_name(name: str) -> tuple[int, int] | None:
    """`ago24` -> (8, 2024).  Returns None on bad names."""
    m = MONTH_REGEX.match(name.lower())
    if not m:
        return None
    mon = ITAL_MONTHS.get(m["mon"])
    if mon is None:
        return None
    return mon, 2000 + int(m["yr"])


def normalise_pod(value) -> str | None:
    """Cope with POD values arriving as floats/scientific notation in Excel.

    Real PODs look like ``99999E00040748`` — alphanumeric. CSV/Excel
    occasionally store them as ``9.999900e+57`` (scientific) or ``inf``
    when the spreadsheet auto-converted. We strip and uppercase whatever
    comes in; non-recoverable rows return None.
    """
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("", "NAN", "INF", "NONE"):
        return None
    # Scientific-notation floats: try to recover the original string
    if re.fullmatch(r"\d+(\.\d+)?E[+-]?\d+", s):
        # The Excel coercion is lossy — best-effort: keep the literal
        return s.replace("E+", "E").replace(".", "")
    return s


def split_ateco(code: str | None) -> tuple[str | None, str | None, str | None]:
    """Extract L1/L2/L3 from a CCATETE value such as ``47.78.99`` or ``DO.R``."""
    if not code or str(code).strip().upper() in ("", "N/A", "NAN"):
        return None, None, None
    code = str(code).strip()
    parts = code.split(".")
    if len(parts) >= 3:
        return parts[0], ".".join(parts[:2]), ".".join(parts[:3])
    if len(parts) == 2:
        return parts[0], code, None
    return code, None, None


# ── Metadata ingestion ───────────────────────────────────────────────────────
def ingest_metadata_file(session: Session, xlsx_path: Path) -> int:
    """Upsert one ``Metadati POD ...xlsx`` file. Returns row count inserted/updated."""
    df = pd.read_excel(xlsx_path, dtype=str)        # everything as str — safer
    df.columns = [c.strip() for c in df.columns]
    # Normalise POD column
    df["POD"] = df["POD"].apply(normalise_pod)
    df = df.dropna(subset=["POD"]).drop_duplicates(subset=["POD"], keep="last")

    # Map Excel -> DB column names (lowercase, prefix-preserved)
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Derive ATECO hierarchy from `ccatete`
    if "ccatete" in df.columns:
        ateco = df["ccatete"].apply(split_ateco)
        df["ateco_l1"] = ateco.apply(lambda x: x[0])
        df["ateco_l2"] = ateco.apply(lambda x: x[1])
        df["ateco_l3"] = ateco.apply(lambda x: x[2])

    # Convert date columns
    for col in ("d_dtall", "d_dtsmo"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Numeric columns
    for col in ("d_flg1", "d_pot1", "d_potma", "d_potc",
                "d_8potim", "d_tfor", "mkost"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Booleans
    if "d_flatt" in df.columns:
        df["d_flatt"] = df["d_flatt"].map(
            {"True": True, "TRUE": True, "1": True,
             "False": False, "FALSE": False, "0": False}
        )

    # Replace NaN with None for psycopg
    df = df.where(pd.notna(df), None)

    cols = [c for c in df.columns if c in {
        "pod", "d_dtall", "d_dtsmo", "d_flg1", "d_pot1", "d_potma", "d_potc",
        "d_flatt", "d_8potim", "d_tipta", "d_49des", "d_viafo", "d_locfo",
        "d_frazfo", "d_tfor", "mkost", "ctar1", "fdesc", "ccatete", "tate3des",
        "ateco_l1", "ateco_l2", "ateco_l3",
    }]
    placeholders = ", ".join(f":{c}" for c in cols)
    updates      = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "pod")
    sql = text(
        f"INSERT INTO pod_metadata ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (pod) DO UPDATE SET {updates};"
    )
    session.execute(sql, df[cols].to_dict("records"))
    session.commit()
    return len(df)


# ── Measurements ingestion ───────────────────────────────────────────────────
def ingest_measurements_file(session: Session, csv_path: Path) -> int:
    """Bulk-insert one monthly CSV. Skips rows already present (ON CONFLICT)."""
    df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["POD"] = df["POD"].apply(normalise_pod)
    df = df.dropna(subset=["POD", "DataMisura"])

    # Date parsing — CSV uses DD/MM/YYYY
    df["data_misura"] = pd.to_datetime(
        df["DataMisura"], format="%d/%m/%Y", errors="coerce"
    ).dt.date
    df = df.dropna(subset=["data_misura"])

    # Pick & rename columns we keep
    rename_map = {
        "POD":                            "pod",
        "Tipologia":                      "tipologia",
        "Matricola":                      "matricola",
        "TipoRilevatore":                 "tipo_rilevatore",
        "CodiceContrattoDispacciamento":  "codice_dispacciamento",
        "Trattamento":                    "trattamento",
        "Tensione":                       "tensione",
        "PotenzaContrattuale":            "potenza_contrattuale",
        "K":                              "k_factor",
        "Effettiva":                      "effettiva",
        "Validata":                       "validata",
        "TotalizzatoriPresenti":          "totalizzatori_presenti",
        "IntervalliCompleti":             "intervalli_completi",
        "Ricostruita":                    "ricostruita",
        "TotalizzatoreF1":                "totalizzatore_f1",
        "TotalizzatoreF2":                "totalizzatore_f2",
        "TotalizzatoreF3":                "totalizzatore_f3",
        "ConsumoF1":                      "consumo_f1",
        "ConsumoF2":                      "consumo_f2",
        "ConsumoF3":                      "consumo_f3",
    }
    df = df.rename(columns={**rename_map, **{q: q.lower() for q in Q_COLS_CSV}})

    # Type coercion
    for col in ("tensione",) + tuple(f"totalizzatore_f{i}" for i in (1, 2, 3)) \
            + tuple(f"consumo_f{i}" for i in (1, 2, 3)):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for q in Q_COLS_DB:
        if q in df.columns:
            df[q] = pd.to_numeric(df[q], errors="coerce")
    for col in ("effettiva", "validata", "totalizzatori_presenti",
                "intervalli_completi", "ricostruita"):
        if col in df.columns:
            df[col] = df[col].map(
                {"True": True, "TRUE": True, "1": True,
                 "False": False, "FALSE": False, "0": False}
            )

    df["source_file"] = csv_path.name

    keep_cols = [
        "pod", "data_misura", "tipologia", "matricola", "tipo_rilevatore",
        "codice_dispacciamento", "trattamento", "tensione",
        "potenza_contrattuale", "k_factor", "effettiva", "validata",
        "totalizzatori_presenti", "intervalli_completi", "ricostruita",
        "totalizzatore_f1", "totalizzatore_f2", "totalizzatore_f3",
        "consumo_f1", "consumo_f2", "consumo_f3", *Q_COLS_DB, "source_file",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].where(pd.notna(df[keep_cols]), None)

    # Deduplicate within the file (PK is pod+data_misura)
    df = df.drop_duplicates(subset=["pod", "data_misura"], keep="last")

    # Bulk insert — chunked to avoid huge parameter lists
    chunk_size = 1000
    total = 0
    placeholders = ", ".join(f":{c}" for c in keep_cols)
    sql = text(
        f"INSERT INTO measurements ({', '.join(keep_cols)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (pod, data_misura) DO NOTHING;"
    )
    records = df.to_dict("records")
    for i in range(0, len(records), chunk_size):
        session.execute(sql, records[i:i + chunk_size])
        total += min(chunk_size, len(records) - i)
    session.commit()
    return total


# ── ATECO lookup ingestion ────────────────────────────────────────────────────
def ingest_ateco_lookup(session: Session, xlsx_path: Path) -> int:
    """Load the official ATECO 2025 description Excel from ISTAT.

    The file has two sheets:
      - "Legenda" → column legend (skipped)
      - "Note esplicative ATECO 2025" → the actual table with columns
        CODICE_ATECO_2025, TITOLO_ITALIANO_ATECO_2025, GERARCHIA_ATECO_2025, …

    We pick the data sheet by looking at its **column names** (not cell values),
    so the legend sheet — which only mentions those names as descriptions —
    is properly skipped.
    """
    all_sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)

    data_df = None
    for sheet_name, df in all_sheets.items():
        cols_upper = [str(c).strip().upper() for c in df.columns]
        if any(c.startswith("CODICE_ATECO") for c in cols_upper):
            df.columns = cols_upper
            data_df = df
            console.print(f"[blue]Using sheet:[/blue] {sheet_name} "
                          f"({len(df):,} rows)")
            break

    if data_df is None:
        raise ValueError(
            f"Could not find an ATECO data sheet in {xlsx_path.name}. "
            f"Expected a sheet with a column named CODICE_ATECO_*."
        )

    code_col = next(
        (c for c in data_df.columns if c.startswith("CODICE_ATECO")), None
    )
    desc_col = next(
        (c for c in data_df.columns if "TITOLO_ITALIANO" in c),
        next((c for c in data_df.columns if "TITOLO" in c), None),
    )
    level_col = next(
        (c for c in data_df.columns if "GERARCHIA_ATECO" in c and "PADRE" not in c),
        None,
    )

    keep = {"code": code_col}
    if desc_col is not None:
        keep["description"] = desc_col

    df = data_df.rename(columns={v: k for k, v in keep.items()})
    df = df[list(keep.keys())].copy()
    df["code"] = df["code"].astype(str).str.strip()
    df = df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")

    # Filter to plausible ATECO codes: alphanumeric / dot-separated, ≤ 20 chars
    valid = df["code"].str.match(r"^[A-Za-z0-9]+(\.[A-Za-z0-9]+){0,3}$", na=False) \
            & (df["code"].str.len() <= 20)
    df = df[valid].reset_index(drop=True)

    # Level: prefer the explicit GERARCHIA column if it exists & is numeric,
    # otherwise count dot-separated segments.
    if level_col is not None and level_col in data_df.columns:
        level_map = (
            data_df[[code_col, level_col]]
            .dropna(subset=[code_col])
            .drop_duplicates(subset=[code_col])
            .set_index(code_col)[level_col]
        )
        df["level"] = df["code"].map(level_map).pipe(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(
                df["code"].apply(lambda c: len(c.split(".")) if "." in c else 1)
            ).astype(int)
        )
    else:
        df["level"] = df["code"].apply(
            lambda c: len(c.split(".")) if "." in c else 1
        )

    if "description" in df.columns:
        df["description"] = df["description"].astype(str).str.strip()
    else:
        df["description"] = None
    df = df.where(pd.notna(df), None)

    if df.empty:
        console.print(f"[yellow]No valid ATECO codes found in[/yellow] {xlsx_path.name}")
        return 0

    sql = text(
        "INSERT INTO ateco_lookup (code, description, level) "
        "VALUES (:code, :description, :level) "
        "ON CONFLICT (code) DO UPDATE SET "
        "  description = EXCLUDED.description, level = EXCLUDED.level;"
    )
    session.execute(sql, df.to_dict("records"))
    session.commit()
    return len(df)


# ── Schema migration for reference tables ────────────────────────────────────
def migrate_reference_tables(session: Session) -> None:
    """Drop and re-create the reference_profiles_* tables so the latest schema
    is applied. Safe to run only when those tables are empty (they hold
    re-ingestable data, not source-of-truth records)."""
    session.execute(text("DROP TABLE IF EXISTS reference_profiles_gse"))
    session.execute(text("DROP TABLE IF EXISTS reference_profiles_arera"))
    session.execute(text("""
        CREATE TABLE reference_profiles_gse (
            profile_code  VARCHAR(10)  NOT NULL,
            month_idx     SMALLINT     NOT NULL,
            hour_idx      SMALLINT     NOT NULL,
            value         REAL         NOT NULL,
            PRIMARY KEY (profile_code, month_idx, hour_idx)
        )
    """))
    session.execute(text("""
        CREATE TABLE reference_profiles_arera (
            power_class   VARCHAR(40)  NOT NULL,
            market        VARCHAR(100) NOT NULL,
            residenza     VARCHAR(50)  NOT NULL,
            province      VARCHAR(60)  NOT NULL,
            day_type      VARCHAR(20)  NOT NULL,
            month_idx     SMALLINT     NOT NULL,
            hour_idx      SMALLINT     NOT NULL,
            value         REAL         NOT NULL,
            PRIMARY KEY (power_class, market, residenza, province, day_type, month_idx, hour_idx)
        )
    """))
    session.commit()


# ── GSE reference profile ingestion ───────────────────────────────────────────
GSE_PROFILE_CODES = ["PDMM", "PDMF", "PAUM", "PAUF",
                     "PIRM", "PIRF", "PACM", "PACF",
                     "MDMM", "MDMF", "MAUM", "MAUF"]


def ingest_gse_profiles(session: Session, xlsx_path: Path) -> int:
    """Load `profili GSE_prelievo_2025.xlsx` into `reference_profiles_gse`.

    The file is wide (one column per profile code) with rows = (Anno, Mese,
    Giorno, Ora). We pivot to long form, aggregate by (profile_code, Mese, Ora)
    via mean, and store the result. Percentage strings ("0,10%") are parsed to
    floats; raw decimals (0.001) are multiplied by 100 to be on the same
    scale as the percentage strings.
    """
    df = pd.read_excel(xlsx_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for c in ("Mese", "Ora"):
        if c not in df.columns:
            raise ValueError(f"GSE file missing required column {c!r}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    profile_cols = [c for c in df.columns if c in GSE_PROFILE_CODES]
    if not profile_cols:
        raise ValueError(
            f"No recognised profile columns in {xlsx_path.name}. "
            f"Expected at least one of {GSE_PROFILE_CODES}."
        )

    for col in profile_cols:
        raw = df[col].astype(str).str.strip()
        has_pct = raw.str.contains("%", na=False).any()
        numeric = (
            raw.str.replace("%", "", regex=False)
               .str.replace(",", ".", regex=False)
               .pipe(pd.to_numeric, errors="coerce")
        )
        df[col] = numeric if has_pct else numeric * 100

    df = df.dropna(subset=["Mese", "Ora"])

    long = df.melt(
        id_vars=["Mese", "Ora"], value_vars=profile_cols,
        var_name="profile_code", value_name="value",
    ).dropna(subset=["value"])

    # Aggregate across days (Anno, Giorno) by (profile_code, Mese, Ora)
    agg = (
        long.groupby(["profile_code", "Mese", "Ora"], observed=True)["value"]
        .mean().reset_index()
        .rename(columns={"Mese": "month_idx", "Ora": "hour_idx"})
    )
    agg["month_idx"] = agg["month_idx"].astype(int)
    agg["hour_idx"]  = agg["hour_idx"].astype(int)

    if agg.empty:
        console.print(f"[yellow]GSE file produced no rows: {xlsx_path.name}[/yellow]")
        return 0

    sql = text(
        "INSERT INTO reference_profiles_gse "
        "(profile_code, month_idx, hour_idx, value) "
        "VALUES (:profile_code, :month_idx, :hour_idx, :value) "
        "ON CONFLICT (profile_code, month_idx, hour_idx) "
        "DO UPDATE SET value = EXCLUDED.value"
    )
    session.execute(sql, agg.to_dict("records"))
    session.commit()
    return len(agg)


# ── ARERA reference profile ingestion ─────────────────────────────────────────
# Re-declared here to avoid an import cycle with config.py
ARERA_FILE_TO_POWER_CLASS = {
    "Copia di dati prelievo orario per provincia0_a_1_5 anno 2024mkt.xlsx": "≤ 1.5 kW",
    "Copia di dati prelievo orario per provincia1_5_a_3 anno 2024TOT.xlsx": "1.5–3 kW",
    "Copia di dati prelievo orario per provincia3_a_4_5 anno 2024mkt.xlsx": "3–4.5 kW",
    "Copia di dati prelievo orario per provincia4_5_a_6 anno 2024TOT.xlsx": "4.5–6 kW",
    "Copia di dati prelievo orario per provincia potenza6 anno 2024-mkt.xlsx": "> 6 kW",
}
ARERA_DAYTYPE_MAP = {
    "Giorno feriale": "Weekday",
    "Sabato":         "Saturday",
    "Domenica":       "Sunday",
}


def _parse_arera_month(v) -> int:
    """`Anno Mese` is sometimes a datetime, sometimes an int year — 0 = annual."""
    if isinstance(v, (int, np.integer)):
        return 0
    if hasattr(v, "month"):
        return int(v.month)
    return 0


def ingest_arera_file(
    session: Session,
    xlsx_path: Path,
    power_class: str,
    province: str = "Trento",
) -> int:
    """Load one ARERA Excel file for a given power class and insert the rows
    matching the requested province into `reference_profiles_arera`."""
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Residenza " in df.columns:                       # tolerated trailing space
        df = df.rename(columns={"Residenza ": "Residenza"})

    required = {"Anno Mese", "Provincia", "Tipo mercato", "Residenza",
                "Working Day", "Orario", "Prelievo medio Orario Provinciale (kWh)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{xlsx_path.name}: missing required columns {sorted(missing)}"
        )

    df = df[df["Provincia"] == province].copy()
    if df.empty:
        console.print(
            f"[yellow]{xlsx_path.name}: no rows for province {province!r}[/yellow]"
        )
        return 0

    df["month_idx"] = df["Anno Mese"].apply(_parse_arera_month)
    df["hour_idx"]  = df["Orario"].astype(str).str.replace("Ora", "", regex=False) \
                                  .astype(int) - 1
    df["day_type"]  = df["Working Day"].map(ARERA_DAYTYPE_MAP).fillna(df["Working Day"])

    out = pd.DataFrame({
        "power_class": power_class,
        "market":      df["Tipo mercato"].astype(str).str.strip(),
        "residenza":   df["Residenza"].astype(str).str.strip(),
        "province":    province,
        "day_type":    df["day_type"].astype(str),
        "month_idx":   df["month_idx"].astype(int),
        "hour_idx":    df["hour_idx"].astype(int),
        "value":       pd.to_numeric(
            df["Prelievo medio Orario Provinciale (kWh)"], errors="coerce"
        ),
    }).dropna(subset=["value"])

    # Deduplicate within file
    out = out.drop_duplicates(subset=[
        "power_class", "market", "residenza", "province",
        "day_type", "month_idx", "hour_idx",
    ])

    if out.empty:
        return 0

    sql = text(
        "INSERT INTO reference_profiles_arera "
        "(power_class, market, residenza, province, day_type, month_idx, hour_idx, value) "
        "VALUES (:power_class, :market, :residenza, :province, :day_type, "
        "        :month_idx, :hour_idx, :value) "
        "ON CONFLICT (power_class, market, residenza, province, day_type, "
        "             month_idx, hour_idx) "
        "DO UPDATE SET value = EXCLUDED.value"
    )
    # Chunk to avoid huge parameter lists
    chunk = 1000
    records = out.to_dict("records")
    for i in range(0, len(records), chunk):
        session.execute(sql, records[i:i + chunk])
    session.commit()
    return len(records)


def ingest_arera_directory(
    session: Session,
    data_dir: Path,
    province: str = "Trento",
) -> dict[str, int]:
    """Find every known ARERA file in `data_dir` and load it."""
    stats: dict[str, int] = {}
    for fname, pc in ARERA_FILE_TO_POWER_CLASS.items():
        path = data_dir / fname
        if path.exists():
            console.print(f"Loading [cyan]{fname}[/cyan] → power class {pc!r}")
            stats[pc] = ingest_arera_file(session, path, pc, province=province)
        else:
            console.print(f"[yellow]Not found:[/yellow] {fname}")
            stats[pc] = 0
    return stats


# ── Materialized view: per-POD average daily profile ─────────────────────────
def create_aggregated_profiles_view(session: Session) -> int:
    """Create (or recreate) the pod_avg_profile materialized view.

    The view contains one row per (POD, tipologia) with 96 pre-computed
    AVG(qN) values. Clustering queries become an instant lookup on a 12k-row
    table instead of a `GROUP BY` over 4.3M measurement rows.

    Returns the number of rows in the resulting view.
    """
    q_avgs = ", ".join(f"AVG({q}) AS {q}" for q in (f"q{i}" for i in range(1, 97)))
    session.execute(text("DROP MATERIALIZED VIEW IF EXISTS pod_avg_profile"))
    session.execute(text(f"""
        CREATE MATERIALIZED VIEW pod_avg_profile AS
        SELECT pod, tipologia, {q_avgs}
        FROM measurements
        WHERE tipologia IS NOT NULL
        GROUP BY pod, tipologia
        WITH DATA
    """))
    session.execute(text(
        "CREATE INDEX idx_pod_avg_profile_pod ON pod_avg_profile (pod)"
    ))
    session.execute(text(
        "CREATE INDEX idx_pod_avg_profile_tipologia ON pod_avg_profile (tipologia)"
    ))
    session.commit()
    n = session.execute(text("SELECT count(*) FROM pod_avg_profile")).scalar_one()
    return int(n)


def refresh_aggregated_profiles_view(session: Session) -> int:
    """Refresh the materialized view in-place (faster than recreating)."""
    try:
        session.execute(text("REFRESH MATERIALIZED VIEW pod_avg_profile"))
        session.commit()
        n = session.execute(text("SELECT count(*) FROM pod_avg_profile")).scalar_one()
        return int(n)
    except Exception:
        session.rollback()
        # If the view doesn't exist yet, create it
        return create_aggregated_profiles_view(session)


# ── Orchestration ────────────────────────────────────────────────────────────
def ingest_data_directory(data_dir: Path) -> dict[str, int]:
    """Walk a legacy `data/` folder and ingest every monthly sub-directory.

    Expected layout:
        data/
          ago24/
            Metadati POD ago24.xlsx
            misure_<timestamp>.csv
          set24/
            …
    """
    stats = {"meta_files": 0, "meas_files": 0, "meta_rows": 0, "meas_rows": 0}

    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        return stats

    monthly_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and parse_dir_name(d.name) is not None
    )
    if not monthly_dirs:
        console.print(f"[yellow]No monthly sub-directories found in[/yellow] {data_dir}")
        return stats

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting monthly folders", total=len(monthly_dirs))
        with engine.begin() as conn:
            session = Session(bind=conn)
            for d in monthly_dirs:
                progress.update(task, description=f"Ingesting {d.name}")
                # Metadata
                for xlsx in d.glob("Metadati*.xlsx"):
                    n = ingest_metadata_file(session, xlsx)
                    stats["meta_files"] += 1
                    stats["meta_rows"]  += n
                # Measurements
                for csv in d.glob("misure*.csv"):
                    n = ingest_measurements_file(session, csv)
                    stats["meas_files"] += 1
                    stats["meas_rows"]  += n
                progress.advance(task)
            session.close()

    return stats
