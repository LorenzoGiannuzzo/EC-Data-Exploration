# EC-Data-Exploration

**Electrical Consumption Data Exploration** — Energy Center Lab, DENERG, Politecnico di Torino

Tools for exploring and clustering electrical consumption data from POD (Point of Delivery) measurements, with ATECO-based hierarchical classification of user typologies.

---

## Repository Structure

```
EC-Data-Exploration/
├── data_exploration.py            # 7-step funnel analysis (CLI)
├── dashboard.py                   # Interactive Streamlit dashboard
├── requirements_dashboard.txt     # Python dependencies
├── .gitignore
└── data/                          # ⚠ NOT tracked in git
    ├── <mesYY>/                   # Monthly folders (e.g. gen25, Feb24, Ago25)
    │   ├── Metadati POD xxx.xlsx  # Metadata: POD, D_49DES, CCATETE, ...
    │   └── misure_xxx.csv         # Measurements: POD, DataMisura, Q1-Q96
    └── Note-esplicative-ATECO-2025-italiano-inglese.xlsx  # ATECO 2025 classification
```

The `data/` folder is excluded from version control (contains sensitive POD data). Each monthly subfolder follows the naming convention `<3-letter Italian month><2-digit year>` (e.g., `gen25` = January 2025, `Ago25` = August 2025).

---

## data_exploration.py

Standalone CLI script performing a 7-step funnel analysis on the full dataset.

### Steps

1. **Census** — Counts unique PODs per month, classifies by user typology (D_49DES)
2. **Temporal filter** — Selects PODs with 12+ unique months of data
3. **Completeness analysis** — Evaluates quarter-hourly data completeness (Q1-Q96) per POD
4. **Consumption profiles** — Computes monthly consumption totals per typology, generates heatmaps and boxplots
5. **Outlier detection** — Identifies anomalous consumption patterns
6. **Power analysis** — Analyzes contractual power distribution per typology
7. **Summary report** — Generates comprehensive log and output tables/charts

### Usage

```bash
python data_exploration.py
```

Outputs are saved to `results/` (tables in `.xlsx`/`.csv`, charts in `.png`, logs in `.txt`).

---

## dashboard.py (Streamlit Dashboard)

Interactive web dashboard for ATECO-based hierarchical exploration and load profile clustering.

### Features

- **Tab 1 — POD Counts**: Horizontal bar charts showing POD distribution across 3 ATECO levels (Section, Division, Class) with scrollable containers and data tables
- **Tab 2 — Load Profile Clustering**: Hierarchical clustering (Ward linkage) on daily load profiles (Q1-Q96, 00:00-23:45 in 15-min intervals)
  - Min-max normalization per POD per month (focuses on pattern shape)
  - Optimal k (min 3) via majority vote among 5 methods (Silhouette, Calinski-Harabasz, Davies-Bouldin, Elbow, Gap Statistic) with Elbow as tie-breaker
  - Cluster profiles with ±1σ bands, monthly breakdown, summary table, typology composition
  - Dendrogram visualization
- **Tab 3 — ATECO Legend**: Searchable reference table with all ATECO 2025 codes found in the dataset (loaded from the official ISTAT classification Excel file)
- **Sidebar**: Global statistics, data coverage filter (All / 12+ months), cascading ATECO level filters (L1 → L2 → L3)

### ATECO Classification

The dashboard loads the official **ATECO 2025** classification from `data/Note-esplicative-ATECO-2025-italiano-inglese.xlsx` (3,257 codes). Non-ATECO distributor codes are handled separately:

| Code | Description |
|------|-------------|
| DO.01 | Domestic - Resident (primary residence) |
| DO.02 | Domestic - Non-Resident (secondary/vacation home) |
| CO.01 | Condominium services - Resident |
| CO.02 | Condominium services - Non-Resident |
| IL.01 | Public lighting |

### Installation & Launch

```bash
pip install -r requirements_dashboard.txt
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501`.

### Dependencies

- Python 3.10+
- streamlit, pandas, numpy, plotly, scipy, scikit-learn, matplotlib, openpyxl

---

## Data Format

### Metadata (Excel)

Each monthly folder contains a `Metadati POD xxx.xlsx` file with columns:

| Column | Description |
|--------|-------------|
| POD | Unique Point of Delivery identifier |
| D_49DES | User typology description (tariff-based) |
| FDESC | Supply phase description |
| TATE3DES | Tariff type description |
| CCATETE | ATECO code (e.g., `47.11.10`, `DO.01`) |

### Measurements (CSV)

Each monthly folder contains a `misure_xxx.csv` file with columns:

| Column | Description |
|--------|-------------|
| POD | Point of Delivery identifier |
| DataMisura | Measurement date (DD/MM/YYYY) |
| Q1-Q96 | Quarter-hourly consumption values (00:00-23:45) |

---

## Author

**Lorenzo Giannuzzo** — Research Engineer, Energy Center Lab, DENERG, Politecnico di Torino
