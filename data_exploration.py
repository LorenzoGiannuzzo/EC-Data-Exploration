"""
================================================================================
DATA EXPLORATION & ANALYSIS - Dati di Consumo Elettrico
================================================================================
Autore: Lorenzo - Energy Center Lab, DENERG, Politecnico di Torino

Flusso di analisi (ad imbuto):
  1. Identificazione di tutti i POD unici e delle tipologie presenti
  2. Filtro: POD con almeno 12+ mesi di rilevazioni
  3. Per i POD filtrati: completezza, consumo mensile/annuale, potenza
     tutto disaggregato per tipologia di utenza
================================================================================
"""

import re
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ==============================================================================
# Lorenzo Giannuzzo: STILE GRAFICI
# ==============================================================================

plt.rcParams.update({
    "figure.figsize": (12, 7), "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 10,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
})

COLORS = [
    "#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#8BC34A", "#FFC107",
    "#3F51B5", "#795548", "#009688", "#CDDC39", "#F44336",
]

MESI_NOMI = {
    1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mag", 6: "Giu",
    7: "Lug", 8: "Ago", 9: "Set", 10: "Ott", 11: "Nov", 12: "Dic",
}


# ==============================================================================
# Lorenzo Giannuzzo: DUAL OUTPUT
# ==============================================================================

class TeeWriter:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

MESI_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}

DIR_PATTERN = re.compile(r"^([a-zA-Z]{3})(\d{2})$")
Q_COLS = [f"Q{i}" for i in range(1, 97)]

META_TARGET_COLS = {
    "POD": ["POD", "pod", "Pod", "Codice POD"],
    "D_49DES": ["D_49DES", "d_49des", "D_49_DES", "D49DES"],
    "FDESC": ["FDESC", "fdesc", "F_DESC"],
    "TATE3DES": ["TATE3DES", "tate3des", "TATE3_DES"],
}

SEP = "=" * 80


# ==============================================================================
# Lorenzo Giannuzzo: UTILITY
# ==============================================================================

def setup_results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tabelle").mkdir(exist_ok=True)
    (RESULTS_DIR / "grafici").mkdir(exist_ok=True)


def parse_directory_name(dirname: str):
    match = DIR_PATTERN.match(dirname)
    if not match:
        return None
    mese_str = match.group(1).lower()
    anno_short = int(match.group(2))
    if mese_str not in MESI_IT:
        return None
    return mese_str, MESI_IT[mese_str], 2000 + anno_short


def find_metadata_file(directory: Path, dirname: str):
    exact = directory / f"Metadati POD {dirname}.xlsx"
    if exact.exists():
        return exact
    for f in directory.iterdir():
        if f.suffix.lower() == ".xlsx" and "metadati" in f.name.lower():
            return f
    return None


def find_measures_file(directory: Path):
    for f in directory.iterdir():
        if f.name.lower().startswith("misure_") and f.suffix.lower() == ".csv":
            return f
    for f in directory.iterdir():
        if f.suffix.lower() == ".csv" and "misure" in f.name.lower():
            return f
    return None


def find_column(df_columns, target_name, variants):
    for v in variants:
        if v in df_columns:
            return v
    col_map = {c.strip().lower(): c for c in df_columns}
    for v in variants:
        if v.strip().lower() in col_map:
            return col_map[v.strip().lower()]
    return None


def normalize_pod(series):
    return series.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)


def pr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def pr_sub(title):
    print(f"\n  --- {title} ---")


def save_table(df, name, index=False):
    df.to_csv(RESULTS_DIR / "tabelle" / f"{name}.csv", index=index, encoding="utf-8-sig", sep=";")
    df.to_excel(RESULTS_DIR / "tabelle" / f"{name}.xlsx", index=index)
    print(f"  >> tabelle/{name}")


def save_fig(fig, name):
    fig.savefig(RESULTS_DIR / "grafici" / f"{name}.png")
    plt.close(fig)
    print(f"  >> grafici/{name}.png")


def add_bar_labels(ax, fmt="{:.0f}", fontsize=9):
    for bar in ax.patches:
        val = bar.get_height()
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val, fmt.format(val),
                    ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def periodo_to_label(p):
    parts = p.split("-")
    return f"{MESI_NOMI.get(int(parts[1]), '?')} {parts[0]}"


def periodo_to_int(p):
    """Converte '2024-02' in 202402 per ordinamento/confronto."""
    parts = p.split("-")
    return int(parts[0]) * 100 + int(parts[1])


def build_tipo_label(df, gcols):
    if len(gcols) == 3:
        return (df[gcols[0]].str[:20] + " | " + df[gcols[1]].str[:20] + " | " + df[gcols[2]].str[:35])
    elif len(gcols) == 2:
        return df[gcols[0]].str[:25] + " | " + df[gcols[1]].str[:30]
    return df[gcols[0]].str[:50]


def has_12_consecutive_months(periodi_list):
    """Verifica se una lista di periodi contiene almeno 12+ mesi."""
    if len(periodi_list) < 12:
        return False
    ints = sorted(set(periodo_to_int(p) for p in periodi_list))
    consecutive = 1
    for i in range(1, len(ints)):
        prev_y, prev_m = divmod(ints[i - 1], 100)
        curr_y, curr_m = divmod(ints[i], 100)
        # Mese successivo?
        if (curr_y == prev_y and curr_m == prev_m + 1) or \
           (curr_y == prev_y + 1 and prev_m == 12 and curr_m == 1):
            consecutive += 1
            if consecutive >= 12:
                return True
        else:
            consecutive = 1
    return False


def get_longest_consecutive_run(periodi_list):
    """Restituisce la lunghezza della sequenza consecutiva più lunga."""
    if not periodi_list:
        return 0
    ints = sorted(set(periodo_to_int(p) for p in periodi_list))
    best = 1
    current = 1
    for i in range(1, len(ints)):
        prev_y, prev_m = divmod(ints[i - 1], 100)
        curr_y, curr_m = divmod(ints[i], 100)
        if (curr_y == prev_y and curr_m == prev_m + 1) or \
           (curr_y == prev_y + 1 and prev_m == 12 and curr_m == 1):
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


# ==============================================================================
# Lorenzo Giannuzzo: CARICAMENTO DATI
# ==============================================================================

def load_all_data():
    pr("CARICAMENTO DATI")

    if not DATA_DIR.exists():
        print(f"  [ERRORE] '{DATA_DIR}' non esiste."); sys.exit(1)

    all_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    valid_dirs = []
    issues = []

    for d in all_dirs:
        parsed = parse_directory_name(d.name)
        if parsed:
            valid_dirs.append((d, parsed))

    print(f"  Directory valide: {len(valid_dirs)}")
    valid_dirs.sort(key=lambda x: (x[1][2], x[1][1]))

    meta_frames, meas_frames = [], []
    first_debug = True

    for d, (mese_str, mese_num, anno) in valid_dirs:
        dirname = d.name
        periodo = f"{anno}-{mese_num:02d}"
        print(f"\n  [{dirname}] -> {periodo}")

        # Metadati
        meta_path = find_metadata_file(d, dirname)
        if meta_path:
            try:
                df_m = pd.read_excel(meta_path, dtype=str)
                if first_debug:
                    print(f"    [DEBUG] Colonne: {list(df_m.columns)}")
                    first_debug = False

                col_map = {}
                for target, variants in META_TARGET_COLS.items():
                    found = find_column(df_m.columns, target, variants)
                    if found:
                        col_map[found] = target

                if "POD" not in col_map.values():
                    issues.append(f"POD non trovato in {dirname}")
                    continue

                df_m = df_m.rename(columns=col_map)
                meta_cols = [c for c in ["POD", "D_49DES", "FDESC", "TATE3DES"] if c in df_m.columns]
                df_m = df_m[meta_cols].copy()
                df_m["POD"] = normalize_pod(df_m["POD"])
                df_m = df_m[df_m["POD"].notna() & (df_m["POD"] != "") & (df_m["POD"] != "NAN")]
                df_m["Periodo"] = periodo
                meta_frames.append(df_m)
                print(f"    Meta: {df_m['POD'].nunique()} POD unici")
            except Exception as e:
                issues.append(f"Errore meta {dirname}: {e}")

        # Misure
        meas_path = find_measures_file(d)
        if meas_path:
            try:
                df_ms = pd.read_csv(meas_path, sep=None, engine="python", dtype=str)
                available_q = [c for c in Q_COLS if c in df_ms.columns]
                if "DataMisura" not in df_ms.columns or "POD" not in df_ms.columns:
                    issues.append(f"Colonne mancanti in {dirname}")
                    continue

                cols = ["DataMisura", "POD"]
                for opt in ["Tensione", "PotenzaContrattuale"]:
                    if opt in df_ms.columns:
                        cols.append(opt)
                cols += available_q
                df_ms = df_ms[cols].copy()
                df_ms["POD"] = normalize_pod(df_ms["POD"])

                num_cols = available_q + [c for c in ["Tensione", "PotenzaContrattuale"] if c in df_ms.columns]
                for col in num_cols:
                    df_ms[col] = pd.to_numeric(
                        df_ms[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
                        errors="coerce")

                df_ms["DataMisura"] = pd.to_datetime(df_ms["DataMisura"], dayfirst=True, errors="coerce")
                df_ms = df_ms[df_ms["POD"].notna() & (df_ms["POD"] != "") & (df_ms["POD"] != "NAN")]
                df_ms["Periodo"] = periodo
                meas_frames.append(df_ms)
                print(f"    Misure: {len(df_ms)} righe, {df_ms['POD'].nunique()} POD")
            except Exception as e:
                issues.append(f"Errore misure {dirname}: {e}")

    df_meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    df_meas = pd.concat(meas_frames, ignore_index=True) if meas_frames else pd.DataFrame()

    print(f"\n  TOTALE: {len(df_meta)} righe meta, {len(df_meas)} righe misure")
    return df_meta, df_meas, issues


# ==============================================================================
# Lorenzo Giannuzzo: STEP 1: PANORAMICA COMPLETA - Tutti i POD
# ==============================================================================

def step1_panoramica(df_meta, df_meas):
    pr("STEP 1: PANORAMICA - TUTTI I POD")

    # Metadati unici
    df_unique = df_meta.sort_values("Periodo").drop_duplicates(subset=["POD"], keep="last")
    n_total = df_unique["POD"].nunique()
    n_meas = df_meas["POD"].nunique()
    group_cols = [c for c in ["D_49DES", "FDESC", "TATE3DES"] if c in df_unique.columns]

    print(f"  POD unici nei metadati:  {n_total:,}")
    print(f"  POD unici nelle misure:  {n_meas:,}")
    print(f"  Periodi nel dataset:     {df_meas['Periodo'].nunique()}")
    print(f"  Range: {df_meas['Periodo'].min()} -> {df_meas['Periodo'].max()}")

    # --- Tipologie ---
    pr_sub("Tipologie utenti identificate")
    combo = (
        df_unique.groupby(group_cols)["POD"].nunique().reset_index()
        .rename(columns={"POD": "N_Utenti"})
        .sort_values("N_Utenti", ascending=False).reset_index(drop=True)
    )
    combo["Percentuale [%]"] = (combo["N_Utenti"] / n_total * 100).round(2)
    combo["Cumulativa [%]"] = combo["Percentuale [%]"].cumsum().round(2)
    print(f"  Combinazioni uniche: {len(combo)}")
    save_table(combo, "step1_tipologie_complete")

    # Grafico top 15
    top = combo.head(15).copy()
    top["Label"] = build_tipo_label(top, group_cols)

    fig, ax = plt.subplots(figsize=(18, max(6, len(top) * 0.5)))
    ax.barh(range(len(top)), top["N_Utenti"], color=COLORS[:len(top)])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Numero Utenti")
    ax.set_title(f"Top 15 Tipologie Utenti\nTotale: {n_total:,} POD unici")
    for i, (val, pct) in enumerate(zip(top["N_Utenti"], top["Percentuale [%]"])):
        ax.text(val + n_total * 0.003, i, f"{val:,}  ({pct}%)", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "step1_top_tipologie")

    # Grafico D_49DES aggregato
    if "D_49DES" in df_unique.columns:
        by_d49 = (
            df_unique.groupby("D_49DES")["POD"].nunique().reset_index()
            .rename(columns={"POD": "N_Utenti"})
            .sort_values("N_Utenti", ascending=False).reset_index(drop=True)
        )
        by_d49["Percentuale [%]"] = (by_d49["N_Utenti"] / n_total * 100).round(2)
        save_table(by_d49, "step1_utenti_per_D49DES")

        fig, ax = plt.subplots(figsize=(max(10, len(by_d49) * 1.5), 6))
        ax.bar(range(len(by_d49)), by_d49["N_Utenti"], color=COLORS[:len(by_d49)], edgecolor="white")
        ax.set_xticks(range(len(by_d49)))
        ax.set_xticklabels(by_d49["D_49DES"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Numero Utenti")
        ax.set_title(f"Distribuzione per Macro-Tipologia (D_49DES)\nTotale: {n_total:,} POD")
        add_bar_labels(ax)
        fig.tight_layout()
        save_fig(fig, "step1_distribuzione_D49DES")

    # POD per mese
    pr_sub("POD presenti per mese")
    pod_per_mese = (
        df_meas.groupby("Periodo")["POD"].nunique().reset_index()
        .rename(columns={"POD": "N_POD"}).sort_values("Periodo")
    )
    pod_per_mese["Label"] = pod_per_mese["Periodo"].apply(periodo_to_label)
    save_table(pod_per_mese, "step1_POD_per_mese")

    fig, ax = plt.subplots(figsize=(max(14, len(pod_per_mese) * 0.8), 7))
    ax.bar(range(len(pod_per_mese)), pod_per_mese["N_POD"], color=COLORS[3], edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(pod_per_mese)))
    ax.set_xticklabels(pod_per_mese["Label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Numero POD"); ax.set_title("POD Presenti per Mese")
    add_bar_labels(ax)
    fig.tight_layout()
    save_fig(fig, "step1_POD_per_mese")

    return df_unique, group_cols


# ==============================================================================
# Lorenzo Giannuzzo: STEP 2: FILTRO - POD con almeno 12+ mesi
# ==============================================================================

def step2_filtro_12_mesi(df_meas, df_unique, group_cols):
    pr("STEP 2: FILTRO - POD CON 12+ MESI DI RILEVAZIONI")

    # Calcola mesi unici per ogni POD
    pr_sub("Calcolo copertura temporale per POD")
    pod_mesi = df_meas.groupby("POD")["Periodo"].nunique().reset_index().rename(columns={"Periodo": "N_Mesi"})

    n_tot = len(pod_mesi)
    n_ok = (pod_mesi["N_Mesi"] >= 12).sum()

    print(f"  POD totali:                    {n_tot:,}")
    print(f"  POD con 12+ mesi di dati:      {n_ok:,}  ({n_ok / n_tot * 100:.1f}%)")
    print(f"  POD con meno di 12 mesi:       {n_tot - n_ok:,}  ({(n_tot - n_ok) / n_tot * 100:.1f}%)")

    # Lorenzo Giannuzzo: Distribuzione mesi disponibili
    bins_m = [0, 1, 3, 6, 9, 12, 15, 19, 100]
    labels_m = ["<1", "1-2", "3-5", "6-8", "9-11", "12-14", "15-18", "19+"]
    pod_mesi["Fascia"] = pd.cut(pod_mesi["N_Mesi"], bins=bins_m, labels=labels_m, right=False)
    dist = (
        pod_mesi.groupby("Fascia", observed=True)["POD"]
        .count().reset_index().rename(columns={"POD": "N_POD"})
    )
    dist["Percentuale [%]"] = (dist["N_POD"] / n_tot * 100).round(2)
    save_table(dist, "step2_distribuzione_mesi_disponibili")

    # Lorenzo Giannuzzo: Grafico imbuto + distribuzione
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    cats = ["Tutti i POD", "12+ mesi"]
    vals = [n_tot, n_ok]
    bars = ax.bar(cats, vals, color=[COLORS[0], COLORS[2]], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:,}\n({val / n_tot * 100:.1f}%)", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("Numero POD")
    ax.set_title("Imbuto: Tutti i POD vs Filtro 12+ Mesi")

    ax = axes[1]
    bar_colors = [COLORS[3] if not l.startswith("12") and not l.startswith("15") and not l.startswith("19") else COLORS[2]
                  for l in dist["Fascia"]]
    ax.bar(range(len(dist)), dist["N_POD"], color=bar_colors, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(dist["Fascia"])
    ax.set_ylabel("Numero POD")
    ax.set_title("Distribuzione Mesi di Dati per POD")
    add_bar_labels(ax)
    fig.tight_layout()
    save_fig(fig, "step2_imbuto_e_distribuzione")

    # Filtra POD ok
    pods_ok = set(pod_mesi.loc[pod_mesi["N_Mesi"] >= 12, "POD"])

    # Lorenzo Giannuzzo: Tipologie dei POD filtrati
    if group_cols and pods_ok:
        pr_sub("Composizione tipologie - POD con 12+ mesi")
        df_filt = df_unique[df_unique["POD"].isin(pods_ok)]
        tipo_filt = (
            df_filt.groupby(group_cols)["POD"].nunique().reset_index()
            .rename(columns={"POD": "N_Utenti"})
            .sort_values("N_Utenti", ascending=False).reset_index(drop=True)
        )
        tipo_filt["Percentuale [%]"] = (tipo_filt["N_Utenti"] / n_ok * 100).round(2)
        tipo_filt["Cumulativa [%]"] = tipo_filt["Percentuale [%]"].cumsum().round(2)
        save_table(tipo_filt, "step2_tipologie_filtrate")

        top_f = tipo_filt.head(15).copy()
        top_f["Label"] = build_tipo_label(top_f, group_cols)

        fig, ax = plt.subplots(figsize=(18, max(6, len(top_f) * 0.5)))
        ax.barh(range(len(top_f)), top_f["N_Utenti"], color=COLORS[:len(top_f)])
        ax.set_yticks(range(len(top_f)))
        ax.set_yticklabels(top_f["Label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Numero Utenti")
        ax.set_title(f"Top 15 Tipologie - POD con 12+ mesi ({n_ok:,} POD)")
        for i, (val, pct) in enumerate(zip(top_f["N_Utenti"], top_f["Percentuale [%]"])):
            ax.text(val + n_ok * 0.003, i, f"{val:,}  ({pct}%)", va="center", fontsize=8)
        fig.tight_layout()
        save_fig(fig, "step2_tipologie_filtrate")

    return pods_ok


# ==============================================================================
# Lorenzo Giannuzzo: STEP 3: COMPLETEZZA DATI (POD filtrati)
# ==============================================================================

def step3_completezza(df_meas, pods_ok):
    pr("STEP 3: COMPLETEZZA DATI (POD con 12+ mesi)")

    if not pods_ok:
        print("  Nessun POD filtrato."); return

    df_f = df_meas[df_meas["POD"].isin(pods_ok)].copy()
    available_q = [c for c in Q_COLS if c in df_f.columns]
    if not available_q:
        print("  Nessuna colonna Q."); return

    n_q = len(available_q)
    df_f["N_Q_NaN"] = df_f[available_q].isna().sum(axis=1)

    comp = (
        df_f.groupby("POD").agg(
            N_Righe=("DataMisura", "count"), N_Mesi=("Periodo", "nunique"),
            Tot_NaN_Q=("N_Q_NaN", "sum")).reset_index()
    )
    comp["Tot_Celle"] = comp["N_Righe"] * n_q
    comp["Completezza [%]"] = ((comp["Tot_Celle"] - comp["Tot_NaN_Q"]) / comp["Tot_Celle"] * 100).round(2)

    overall = (1 - comp["Tot_NaN_Q"].sum() / comp["Tot_Celle"].sum()) * 100
    print(f"  Completezza globale: {overall:.2f}%")
    print(f"  POD analizzati: {len(comp):,}")
    save_table(comp, "step3_completezza_per_POD")

    # Distribuzione
    bins = [0, 50, 80, 90, 95, 99, 99.9, 100.01]
    labels = ["<50%", "50-80%", "80-90%", "90-95%", "95-99%", "99-99.9%", ">=99.9%"]
    comp["Fascia"] = pd.cut(comp["Completezza [%]"], bins=bins, labels=labels, right=True)
    dist = comp.groupby("Fascia", observed=True)["POD"].count().reset_index().rename(columns={"POD": "N_POD"})
    dist["Percentuale [%]"] = (dist["N_POD"] / len(comp) * 100).round(2)
    save_table(dist, "step3_distribuzione_completezza")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.bar(range(len(dist)), dist["N_POD"], color=COLORS[4], edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(dist["Fascia"], rotation=30, ha="right")
    ax.set_ylabel("Numero POD")
    ax.set_title(f"Completezza Dati ({len(comp):,} POD filtrati)")
    add_bar_labels(ax)

    ax = axes[1]
    ax.hist(comp["Completezza [%]"], bins=50, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(100, color="red", ls="--", lw=1.5, label="100%")
    ax.set_xlabel("Completezza [%]"); ax.set_ylabel("Frequenza")
    ax.set_title("Istogramma Completezza"); ax.legend()
    fig.tight_layout()
    save_fig(fig, "step3_completezza")


# ==============================================================================
# Lorenzo Giannuzzo: STEP 4: CONSUMO MENSILE PER TIPOLOGIA (POD filtrati)
# ==============================================================================

def step4_consumo_mensile(df_meas, pods_ok, df_unique, group_cols):
    pr("STEP 4: CONSUMO MENSILE PER TIPOLOGIA (POD filtrati)")

    if not pods_ok:
        print("  Nessun POD filtrato."); return

    df_f = df_meas[df_meas["POD"].isin(pods_ok)].copy()
    available_q = [c for c in Q_COLS if c in df_f.columns]
    if not available_q:
        return

    df_f["ConsumoGiornaliero_kWh"] = df_f[available_q].sum(axis=1)

    monthly = (
        df_f.groupby(["POD", "Periodo"])["ConsumoGiornaliero_kWh"]
        .sum().reset_index().rename(columns={"ConsumoGiornaliero_kWh": "ConsumoMensile_kWh"})
    )

    # Lorenzo Giannuzzo: Merge tipologia
    meta = df_unique[["POD"] + group_cols].copy()
    meta["Tipologia"] = build_tipo_label(meta, group_cols)
    monthly = monthly.merge(meta[["POD", "Tipologia"]], on="POD", how="left")
    monthly["Tipologia"] = monthly["Tipologia"].fillna("Sconosciuto")

    # Lorenzo Giannuzzo: Statistiche per tipologia
    pr_sub("Consumo medio mensile per tipologia")
    stats = (
        monthly.groupby("Tipologia")["ConsumoMensile_kWh"]
        .agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    )
    stats.columns = ["Tipologia", "N_Osservazioni", "Media_kWh", "Mediana_kWh",
                      "StdDev_kWh", "Min_kWh", "Max_kWh"]
    npod = monthly.groupby("Tipologia")["POD"].nunique().reset_index().rename(columns={"POD": "N_POD"})
    stats = stats.merge(npod, on="Tipologia").sort_values("N_POD", ascending=False).reset_index(drop=True)
    for col in stats.select_dtypes(include=[np.number]).columns:
        stats[col] = stats[col].round(1)
    save_table(stats, "step4_consumo_mensile_per_tipologia")

    # Lorenzo Giannuzzo: Box plot top 15
    top = stats.nlargest(15, "N_POD")["Tipologia"].tolist()
    df_top = monthly[monthly["Tipologia"].isin(top)]

    if top and not df_top.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        box_data = [df_top.loc[df_top["Tipologia"] == t, "ConsumoMensile_kWh"].dropna().values for t in top]
        # Filtra tipologie senza dati
        valid = [(d, t) for d, t in zip(box_data, top) if len(d) > 0]
        if valid:
            box_data_clean, top_clean = zip(*valid)
            bp = ax.boxplot(list(box_data_clean), tick_labels=[t[:40] for t in top_clean],
                            patch_artist=True, showfliers=False, vert=False)
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(COLORS[j % len(COLORS)]); patch.set_alpha(0.7)
            ax.set_xlabel("Consumo Mensile [kWh]")
            ax.set_title(f"Consumo Mensile - Top 15 Tipologie ({len(pods_ok):,} POD filtrati)")
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            save_fig(fig, "step4_boxplot_consumo_mensile")
        else:
            plt.close(fig)

    # Lorenzo Giannuzzo: Heatmap
    periodi_ord = sorted(monthly["Periodo"].unique())
    pivot = monthly.pivot_table(index="Tipologia", columns="Periodo",
                                 values="ConsumoMensile_kWh", aggfunc="mean").round(1)
    pivot["Media"] = pivot.mean(axis=1).round(1)
    pivot = pivot.sort_values("Media", ascending=False)
    save_table(pivot, "step4_heatmap_consumo_tipologia_periodo", index=True)

    top15 = pivot.head(15).drop(columns=["Media"])
    cols = [c for c in periodi_ord if c in top15.columns]
    hm = top15[cols].values

    fig, ax = plt.subplots(figsize=(16, max(5, len(top15) * 0.5)))
    im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([periodo_to_label(c) for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([t[:45] for t in top15.index], fontsize=7)
    ax.set_title("Heatmap Consumo Medio Mensile [kWh] - Top 15 Tipologie")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            val = hm[i, j]
            if not np.isnan(val):
                clr = "white" if val > np.nanpercentile(hm, 70) else "black"
                ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=5, color=clr)
    fig.colorbar(im, ax=ax, label="kWh", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "step4_heatmap_consumo")

    # Lorenzo Giannuzzo: Andamento temporale
    pr_sub("Andamento temporale per macro-tipologia (D_49DES)")
    if "D_49DES" in df_unique.columns:
        m2 = monthly.merge(df_unique[["POD", "D_49DES"]], on="POD", how="left")
        m2["D_49DES"] = m2["D_49DES"].fillna("Sconosciuto")
        tipologie = sorted(m2["D_49DES"].unique())

        fig, ax = plt.subplots(figsize=(max(14, len(periodi_ord) * 0.9), 8))
        for idx, tipo in enumerate(tipologie):
            s = m2[m2["D_49DES"] == tipo].groupby("Periodo")["ConsumoMensile_kWh"].agg(["mean", "std"]).reindex(periodi_ord)
            color = COLORS[idx % len(COLORS)]
            ax.plot(range(len(periodi_ord)), s["mean"], "o-", color=color, lw=2,
                    label=tipo[:40], markersize=5)
            ax.fill_between(range(len(periodi_ord)),
                            (s["mean"] - s["std"]).clip(lower=0), s["mean"] + s["std"],
                            alpha=0.1, color=color)
        ax.set_xticks(range(len(periodi_ord)))
        ax.set_xticklabels([periodo_to_label(p) for p in periodi_ord], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Consumo Mensile [kWh]")
        ax.set_title(f"Andamento Consumo Mensile per D_49DES ({len(pods_ok):,} POD)")
        ax.legend(title="Tipologia", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        save_fig(fig, "step4_andamento_temporale")


# ==============================================================================
# Lorenzo Giannuzzo: STEP 5: CONSUMO ANNUALE PER TIPOLOGIA (POD filtrati)
# ==============================================================================

def step5_consumo_annuale(df_meas, pods_ok, df_unique, group_cols):
    pr("STEP 5: CONSUMO ANNUALE PER TIPOLOGIA (POD filtrati)")

    if not pods_ok:
        print("  Nessun POD filtrato."); return

    df_f = df_meas[df_meas["POD"].isin(pods_ok)].copy()
    available_q = [c for c in Q_COLS if c in df_f.columns]

    if "ConsumoGiornaliero_kWh" not in df_f.columns:
        df_f["ConsumoGiornaliero_kWh"] = df_f[available_q].sum(axis=1)

    df_f["Anno"] = df_f["Periodo"].str[:4].astype(int)

    # Solo POD con 12 mesi nello stesso anno
    pod_anno = df_f.groupby(["POD", "Anno"])["Periodo"].nunique().reset_index().rename(columns={"Periodo": "N_Mesi"})
    pod_anno_12 = pod_anno[pod_anno["N_Mesi"] >= 12]
    n_annual = pod_anno_12["POD"].nunique()

    print(f"  POD con 12 mesi in uno stesso anno: {n_annual:,}")

    if n_annual == 0:
        print("  Nessun POD con anno completo. Salto.")
        return

    annual = (
        df_f.merge(pod_anno_12[["POD", "Anno"]], on=["POD", "Anno"], how="inner")
        .groupby(["POD", "Anno"])["ConsumoGiornaliero_kWh"]
        .sum().reset_index().rename(columns={"ConsumoGiornaliero_kWh": "ConsumoAnnuale_kWh"})
    )

    # Lorenzo Giannuzzo: Merge tipologia
    meta = df_unique[["POD"] + group_cols].copy()
    meta["Tipologia"] = build_tipo_label(meta, group_cols)
    annual = annual.merge(meta[["POD", "Tipologia"]], on="POD", how="left")
    annual["Tipologia"] = annual["Tipologia"].fillna("Sconosciuto")

    # Lorenzo Giannuzzo: Statistiche
    pr_sub("Consumo annuale per tipologia")
    stats_a = (
        annual.groupby("Tipologia")["ConsumoAnnuale_kWh"]
        .agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    )
    stats_a.columns = ["Tipologia", "N", "Media_kWh", "Mediana_kWh", "StdDev_kWh", "Min_kWh", "Max_kWh"]
    npod_a = annual.groupby("Tipologia")["POD"].nunique().reset_index().rename(columns={"POD": "N_POD"})
    stats_a = stats_a.merge(npod_a, on="Tipologia").sort_values("N_POD", ascending=False).reset_index(drop=True)
    for col in stats_a.select_dtypes(include=[np.number]).columns:
        stats_a[col] = stats_a[col].round(1)
    save_table(stats_a, "step5_consumo_annuale_per_tipologia")
    save_table(annual, "step5_consumo_annuale_per_POD")

    # Fasce consumo
    bins_a = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, np.inf]
    labels_a = ["0-500", "500-1k", "1k-1.5k", "1.5k-2k", "2k-3k",
                "3k-5k", "5k-10k", "10k-20k", "20k-50k", "50k-100k", ">100k"]
    annual["Fascia"] = pd.cut(annual["ConsumoAnnuale_kWh"], bins=bins_a, labels=labels_a, right=False)

    fascia = annual.groupby("Fascia", observed=True).agg(
        N_Utenti=("POD", "nunique"), Media=("ConsumoAnnuale_kWh", "mean")).reset_index()
    fascia["Percentuale [%]"] = (fascia["N_Utenti"] / n_annual * 100).round(2)
    save_table(fascia, "step5_fasce_consumo_annuale")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(range(len(fascia)), fascia["N_Utenti"], color=COLORS[0], edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(fascia)))
    ax.set_xticklabels(fascia["Fascia"], rotation=30, ha="right")
    ax.set_ylabel("Numero Utenti")
    ax.set_title(f"Distribuzione Consumo Annuale [kWh] ({n_annual:,} POD)")
    add_bar_labels(ax)
    fig.tight_layout()
    save_fig(fig, "step5_distribuzione_consumo_annuale")

    # Lorenzo Giannuzzo: Istogramma + log
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.hist(annual["ConsumoAnnuale_kWh"], bins=50, color=COLORS[0], edgecolor="white", alpha=0.85)
    ax.axvline(annual["ConsumoAnnuale_kWh"].median(), color="red", ls="--", lw=2,
               label=f"Mediana: {annual['ConsumoAnnuale_kWh'].median():,.0f}")
    ax.axvline(annual["ConsumoAnnuale_kWh"].mean(), color="orange", ls="--", lw=2,
               label=f"Media: {annual['ConsumoAnnuale_kWh'].mean():,.0f}")
    ax.set_xlabel("kWh"); ax.set_ylabel("Frequenza"); ax.set_title("Consumo Annuale"); ax.legend()

    ax = axes[1]
    pos = annual.loc[annual["ConsumoAnnuale_kWh"] > 0, "ConsumoAnnuale_kWh"]
    if len(pos) > 0:
        ax.hist(np.log10(pos), bins=50, color=COLORS[2], edgecolor="white", alpha=0.85)
        ax.set_xlabel("log10(kWh)"); ax.set_ylabel("Frequenza"); ax.set_title("Log-Scale")
    fig.tight_layout()
    save_fig(fig, "step5_istogramma_consumo_annuale")

    # Lorenzo Giannuzzo: Stacked per D_49DES
    if "D_49DES" in df_unique.columns:
        annual2 = annual.merge(df_unique[["POD", "D_49DES"]], on="POD", how="left")
        annual2["D_49DES"] = annual2["D_49DES"].fillna("Sconosciuto")
        cross = annual2.groupby(["Fascia", "D_49DES"], observed=True)["POD"].nunique().reset_index().rename(columns={"POD": "N"})
        piv = cross.pivot_table(index="Fascia", columns="D_49DES", values="N", fill_value=0, observed=True)
        save_table(piv, "step5_consumo_annuale_per_D49DES", index=True)

        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(piv))
        for idx, col in enumerate(piv.columns):
            ax.bar(range(len(piv)), piv[col].values, bottom=bottom, color=COLORS[idx % len(COLORS)],
                   edgecolor="white", linewidth=0.5, label=col[:35])
            bottom += piv[col].values
        ax.set_xticks(range(len(piv)))
        ax.set_xticklabels(piv.index, rotation=30, ha="right")
        ax.set_ylabel("Numero Utenti")
        ax.set_title(f"Consumo Annuale per Fascia e Tipologia ({n_annual:,} POD)")
        ax.legend(title="D_49DES", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
        fig.tight_layout()
        save_fig(fig, "step5_consumo_annuale_stacked")


# ==============================================================================
# Lorenzo Giannuzzo: STEP 6: POTENZA PER TIPOLOGIA (POD filtrati)
# ==============================================================================

def step6_potenza(df_meas, pods_ok, df_unique, group_cols):
    pr("STEP 6: POTENZA CONTRATTUALE PER TIPOLOGIA (POD filtrati)")

    if "PotenzaContrattuale" not in df_meas.columns or not pods_ok:
        print("  Dati insufficienti."); return

    df_f = df_meas[df_meas["POD"].isin(pods_ok)].copy()
    pod_power = (
        df_f.dropna(subset=["PotenzaContrattuale"])
        .groupby("POD")["PotenzaContrattuale"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )

    meta = df_unique[["POD"] + group_cols].copy()
    meta["Tipologia"] = build_tipo_label(meta, group_cols)
    merged = pod_power.merge(meta[["POD", "Tipologia"]], on="POD", how="inner")

    # Statistiche potenza per tipologia
    pr_sub("Potenza media per tipologia")
    pstats = merged.groupby("Tipologia")["PotenzaContrattuale"].agg(
        ["count", "mean", "median", "min", "max"]).reset_index()
    pstats.columns = ["Tipologia", "N_POD", "PotMedia_kW", "PotMediana_kW", "PotMin_kW", "PotMax_kW"]
    pstats = pstats.sort_values("N_POD", ascending=False).reset_index(drop=True)
    for col in pstats.select_dtypes(include=[np.number]).columns:
        pstats[col] = pstats[col].round(2)
    save_table(pstats, "step6_potenza_per_tipologia")

    # Lorenzo Giannuzzo: Torta globale
    n_pods = len(pod_power)
    power_dist = pod_power.groupby("PotenzaContrattuale")["POD"].nunique().reset_index().rename(columns={"POD": "N"})
    save_table(power_dist, "step6_distribuzione_potenza")

    fig, ax = plt.subplots(figsize=(16, 11))
    if len(power_dist) > 10:
        top_n = power_dist.nlargest(10, "N")
        others = power_dist[~power_dist.index.isin(top_n.index)]["N"].sum()
        pie_data = pd.concat([top_n, pd.DataFrame([{"PotenzaContrattuale": -1, "N": others}])], ignore_index=True)
    else:
        pie_data = power_dist
    pie_labels = ["Altro" if r["PotenzaContrattuale"] == -1 else f"{r['PotenzaContrattuale']:.1f} kW"
                  for _, r in pie_data.iterrows()]
    explode = [0.03] * len(pie_data); explode[pie_data["N"].idxmax()] = 0.06
    wedges, _, autotexts = ax.pie(
        pie_data["N"], labels=None, autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
        colors=COLORS[:len(pie_data)], startangle=90, explode=explode, pctdistance=0.8)
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold")
    legend_labels = [f"{lbl}  ({r['N']:,}, {r['N']/n_pods*100:.1f}%)"
                     for lbl, (_, r) in zip(pie_labels, pie_data.iterrows())]
    ax.legend(wedges, legend_labels, title="Potenza Contrattuale",
              loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=12, title_fontsize=13)
    ax.set_title(f"Distribuzione Potenza Contrattuale ({n_pods:,} POD filtrati)")
    fig.tight_layout()
    save_fig(fig, "step6_distribuzione_potenza")

    # Lorenzo Giannuzzo: Stacked top 10 tipologie × potenza
    cross = merged.groupby(["Tipologia", "PotenzaContrattuale"])["POD"].nunique().reset_index().rename(columns={"POD": "N"})
    pivot = cross.pivot_table(index="Tipologia", columns="PotenzaContrattuale", values="N", fill_value=0)
    pivot["TOT"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOT", ascending=False)
    save_table(pivot, "step6_potenza_per_tipologia_pivot", index=True)

    top10 = pivot.nlargest(10, "TOT").drop(columns=["TOT"])
    fig, ax = plt.subplots(figsize=(16, 8))
    top10.plot(kind="barh", stacked=True, ax=ax, colormap="tab20", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Numero Utenti"); ax.set_ylabel("")
    ax.set_title("Top 10 Tipologie - Distribuzione Potenza")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(title="kW", loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=min(8, len(top10.columns)), fontsize=7, title_fontsize=9)
    fig.tight_layout()
    save_fig(fig, "step6_top10_tipologie_potenza")


# ==============================================================================
# Lorenzo Giannuzzo: STEP 7: RIEPILOGO FINALE EXCEL
# ==============================================================================

def step7_riepilogo(df_meta, df_meas, df_unique, group_cols, pods_ok):
    pr("STEP 7: GENERAZIONE RIEPILOGO EXCEL")

    xlsx_path = RESULTS_DIR / "RIEPILOGO_GENERALE.xlsx"
    n_totali = df_meas["POD"].nunique()
    n_filtrati = len(pods_ok)

    # Lorenzo Giannuzzo: Sheet 1: Numeri chiave
    general = pd.DataFrame([
        {"Metrica": "POD unici totali (misure)", "Valore": f"{n_totali:,}"},
        {"Metrica": "POD con 12+ mesi di dati", "Valore": f"{n_filtrati:,}"},
        {"Metrica": "% POD filtrati", "Valore": f"{n_filtrati / n_totali * 100:.1f}%"},
        {"Metrica": "Mesi nel dataset", "Valore": str(df_meas["Periodo"].nunique())},
        {"Metrica": "Range temporale", "Valore": f"{df_meas['Periodo'].min()} -> {df_meas['Periodo'].max()}"},
    ])

    # Lorenzo Giannuzzo: Sheet 2: Tipologie tutti i POD
    tipo_all = (
        df_unique.groupby(group_cols)["POD"].nunique().reset_index()
        .rename(columns={"POD": "N_Utenti"})
        .sort_values("N_Utenti", ascending=False).reset_index(drop=True)
    )
    tipo_all["Percentuale [%]"] = (tipo_all["N_Utenti"] / df_unique["POD"].nunique() * 100).round(2)
    tipo_all["Cumulativa [%]"] = tipo_all["Percentuale [%]"].cumsum().round(2)

    # Lorenzo Giannuzzo: Sheet 3: Tipologie POD filtrati
    df_filt = df_unique[df_unique["POD"].isin(pods_ok)]
    tipo_filt = (
        df_filt.groupby(group_cols)["POD"].nunique().reset_index()
        .rename(columns={"POD": "N_Utenti"})
        .sort_values("N_Utenti", ascending=False).reset_index(drop=True)
    )
    tipo_filt["Percentuale [%]"] = (tipo_filt["N_Utenti"] / n_filtrati * 100).round(2)
    tipo_filt["Cumulativa [%]"] = tipo_filt["Percentuale [%]"].cumsum().round(2)

    # Lorenzo Giannuzzo: Sheet 4: Riepilogo per tipologia (consumo + potenza)
    available_q = [c for c in Q_COLS if c in df_meas.columns]
    df_f = df_meas[df_meas["POD"].isin(pods_ok)].copy()
    if "ConsumoGiornaliero_kWh" not in df_f.columns:
        df_f["ConsumoGiornaliero_kWh"] = df_f[available_q].sum(axis=1)

    meta = df_unique[["POD"] + group_cols].copy()
    meta["Tipologia"] = build_tipo_label(meta, group_cols)

    monthly = (
        df_f.groupby(["POD", "Periodo"])["ConsumoGiornaliero_kWh"]
        .sum().reset_index().rename(columns={"ConsumoGiornaliero_kWh": "ConsumoMensile_kWh"})
        .merge(meta[["POD", "Tipologia"]], on="POD", how="left")
    )

    riepilogo = monthly.groupby("Tipologia").agg(
        N_POD=("POD", "nunique"),
        Consumo_Mensile_Medio_kWh=("ConsumoMensile_kWh", "mean"),
        Consumo_Mensile_Mediano_kWh=("ConsumoMensile_kWh", "median"),
    ).reset_index()

    # Lorenzo Giannuzzo: Aggiungi consumo annuale (stima: media mensile × 12)
    riepilogo["Consumo_Annuale_Stimato_kWh"] = (riepilogo["Consumo_Mensile_Medio_kWh"] * 12).round(0)

    # Lorenzo Giannuzzo: Aggiungi potenza media
    if "PotenzaContrattuale" in df_f.columns:
        pod_pow = (
            df_f.dropna(subset=["PotenzaContrattuale"])
            .groupby("POD")["PotenzaContrattuale"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index().merge(meta[["POD", "Tipologia"]], on="POD", how="left")
        )
        pot_stats = pod_pow.groupby("Tipologia")["PotenzaContrattuale"].agg(
            ["mean", "median"]).reset_index()
        pot_stats.columns = ["Tipologia", "Potenza_Media_kW", "Potenza_Mediana_kW"]
        riepilogo = riepilogo.merge(pot_stats, on="Tipologia", how="left")

    riepilogo = riepilogo.sort_values("N_POD", ascending=False).reset_index(drop=True)
    for col in riepilogo.select_dtypes(include=[np.number]).columns:
        riepilogo[col] = riepilogo[col].round(2)

    # Lorenzo Giannuzzo: Scrivi Excel
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        general.to_excel(writer, sheet_name="Numeri_Chiave", index=False)
        tipo_all.to_excel(writer, sheet_name="Tipologie_Tutti_POD", index=False)
        tipo_filt.to_excel(writer, sheet_name="Tipologie_12mesi", index=False)
        riepilogo.to_excel(writer, sheet_name="Riepilogo_per_Tipologia", index=False)

    print(f"  >> {xlsx_path.name}")
    print(f"\n  === RIEPILOGO ===")
    print(f"  POD totali:      {n_totali:,}")
    print(f"  POD filtrati:    {n_filtrati:,} ({n_filtrati / n_totali * 100:.1f}%)")
    print(f"  Tipologie:       {len(tipo_filt)}")
    print(f"\n  Top 5 tipologie (POD filtrati):")
    for _, r in tipo_filt.head(5).iterrows():
        lbl = " | ".join(str(r[c]) for c in group_cols)
        print(f"    {lbl}: {r['N_Utenti']:,} ({r['Percentuale [%]']}%)")


# ==============================================================================
# Lorenzo Giannuzzo: MAIN
# ==============================================================================

def main():
    setup_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"log_esplorazione_{timestamp}.txt"
    tee = TeeWriter(log_path)
    sys.stdout = tee

    print(SEP)
    print("  DATA EXPLORATION - Dati di Consumo Elettrico")
    print(f"  {DATA_DIR.resolve()}")
    print(f"  {timestamp}")
    print(SEP)

    try:
        df_meta, df_meas, issues = load_all_data()

        if df_meta.empty and df_meas.empty:
            print("\n  [ERRORE] Nessun dato."); sys.exit(1)

        # Lorenzo Giannuzzo: STEP 1: Panoramica tutti i POD
        df_unique, group_cols = step1_panoramica(df_meta, df_meas)

        # Lorenzo Giannuzzo: STEP 2: Filtro 12+ mesi
        pods_ok = step2_filtro_12_mesi(df_meas, df_unique, group_cols)

        # Lorenzo Giannuzzo: STEP 3-6: Analisi sui POD filtrati
        step3_completezza(df_meas, pods_ok)
        step4_consumo_mensile(df_meas, pods_ok, df_unique, group_cols)
        step5_consumo_annuale(df_meas, pods_ok, df_unique, group_cols)
        step6_potenza(df_meas, pods_ok, df_unique, group_cols)

        # Lorenzo Giannuzzo: STEP 7: Riepilogo Excel
        step7_riepilogo(df_meta, df_meas, df_unique, group_cols, pods_ok)

        if issues:
            print(f"\n  Warnings: {len(issues)}")
            for iss in issues[:20]:
                print(f"    {iss}")

        print(f"\n{SEP}\n  Analisi completata!\n  {RESULTS_DIR.resolve()}\n{SEP}")

    except Exception as e:
        print(f"\n  [ERRORE FATALE] {e}")
        import traceback; traceback.print_exc()
    finally:
        tee.close()

if __name__ == "__main__":
    main()