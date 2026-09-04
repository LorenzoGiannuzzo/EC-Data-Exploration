# synthgen — Input and output specification

Companion document to `README.md`. It specifies every file the pipeline reads
and every file it writes, stage by stage, with the schema, the units and the
provenance of each.

| | |
|---|---|
| **Software** | synthgen, version 1.0 |
| **Author** | Lorenzo Giannuzzo, Politecnico di Torino, DENERG, Energy Center Lab |
| **Contact** | lorenzo.giannuzzo@polito.it |
| **Date** | September 2026 |

---

## 1. Overview

The pipeline consists of four stages. Each writes artefacts that the following
one reads, so no stage recomputes what a previous one has already produced.

```
data/                      ──►  Stage 1  ──►  cache/synthgen/
                                                    │
                                                    ├──►  Stage 2  ──►  results/eligibility/
                                                    │
                                                    ├──►  Stage 3  ──►  models/
                                                    │                       │
                                                    └──►  Stage 4  ◄────────┘
                                                             │
                                                             └──►  <outdir>/
```

Artefacts fall into three classes, which determine how they should be handled:

| Class | Directories | Rebuildable |
|---|---|---|
| **Input** | `data/` | No. Supplied externally |
| **Intermediate** | `cache/synthgen/` | Yes, by re-running Stage 1 |
| **Models** | `models/` | Yes, by re-running Stage 3, at significant computational cost |
| **Output** | `results/`, and the directory given to `--outdir` | Yes, by re-running the stage that produced it |

---

## 2. Input: the measurement archive

### 2.1 Location and structure

Resolved from `data.root` in `config_synthgen.yaml`, relative to the parent
directory of the package. One directory per month, in any capitalisation. Each
must contain one file matching `data.meas_glob` (`misure*.csv`) and one matching
`data.meta_glob` (`Metadati*.xls*`). Directories missing either are skipped.

```
data/
├── ago24/
│   ├── misure_ago24.csv
│   └── Metadati POD ago24.xlsx
└── Set24/
    ├── misure_set24.csv
    └── Metadati POD set24.xlsx
```

### 2.2 Measurement file

| Property | Value |
|---|---|
| Format | CSV |
| Field separator | `;` (`data.sep`) |
| Encoding | `utf-8-sig` (`data.encoding`); a byte order mark precedes the first header |
| Decimal mark | Comma |
| Date format | `%d/%m/%Y` (`data.date_format`) |
| Unit of readings | Wh (`data.reading_unit`); converted to kWh on read |

Columns read:

| Column | Type | Unit | Description |
|---|---|---|---|
| `Id` | integer | | Record identifier, used only to break ties between duplicates |
| `POD` | text | | Point of delivery identifier. Must be read as text: a code such as `99999E00000053` is valid scientific notation and any reader left to infer types converts it to a float |
| `DataMisura` | date | | Civil date of the measurement |
| `Tipologia` | text | | Quantity recorded. `AP` active withdrawn, `AN` active injected, `RLP` / `RLN` / `RCN` reactive |
| `PotenzaContrattuale` | decimal | kW | Contractual power |
| `K` | integer | | Meter constant |
| `Q1` … `Q96` | decimal | Wh | The ninety-six quarter-hours of the day |

Only rows with `Tipologia == "AP"` (`data.keep_kind`) carry the consumption
curve and are retained. Points appearing with `Tipologia == "AN"`
(`data.inject_kind`) are flagged as prosumers.

Three properties of this format receive specific treatment:

**Meter constant.** `K` equals 1 for the majority of points and 20, 25 or 40 for
those metered through a current transformer, whose `Q` values are raw meter
counts rather than energy. It is applied per row rather than per point, since a
point may change it within a single month. Governed by `data.apply_k`.

**Padding.** Columns `Q97` to `Q100` exist so that every row has uniform width.
They are zero padding and do not represent the two additional quarter-hours of
the October daylight saving day. Only `Q1` to `Q96` are read
(`data.n_quarters`).

**Duplicates.** A limited number of POD-day pairs appear more than once.
Resolution is deterministic, so that two runs of the same code cannot disagree:
the row with the greatest number of valued quarter-hours is retained, with ties
broken on `Id`. The count of duplicates resolved is reported in the funnel.

### 2.3 Metadata workbook

| Property | Value |
|---|---|
| Format | Excel `.xlsx`, read with openpyxl |
| Decimal mark | Point, unlike the measurement file |

Columns read:

| Column | Configuration key | Description |
|---|---|---|
| `POD` | `data.meta_cols.pod` | Point identifier, joined to the measurement file |
| `CCATETE` | `data.meta_cols.ateco` | Activity code, right-padded. `47.11.10`, or `DO.01` / `DO.02` for domestic |
| `D_POTC` | `data.meta_cols.power` | Contractual power, kW |
| `D_TIPTA` | `data.meta_cols.tariff` | Tariff class. `TD` domestic, `BTA1` … `BTA6` non-domestic |
| `D_49DES` | `data.meta_cols.tariff_desc` | Tariff description |
| `D_DTSMO` | `data.meta_cols.deactivated` | Non-empty when the point was decommissioned |

Contractual power is declared in both files and the two do not always agree. The
larger of the two is retained, since it serves as a censoring threshold and must
not be too tight.

---

## 3. Stage 1: pre-processing

**Command:** `python -m synthgen.preprocessing`

**Reads:** the archive described in Section 2.

**Writes:**

### 3.1 `cache/synthgen/curves.npy`

NumPy array, `float32`, shape `(n_days, 96)`. One row per valid POD-day, in the
same order as `days.parquet`. Values are the energy drawn in each quarter-hour,
in kWh, after application of the meter constant and conversion from Wh.

Row `i` of this array and row `i` of `days.parquet` describe the same POD-day.
The two files must be kept together.

### 3.2 `cache/synthgen/days.parquet`

One row per POD-day.

| Column | Type | Description |
|---|---|---|
| `pod` | text | Point identifier |
| `date` | date | Civil date |
| `power_kW` | float | Contractual power at that date |
| `k` | float | Meter constant applied |
| `prosumer` | bool | Whether the point also injects |
| `daytype` | text | `weekday`, `saturday` or `sunday`; national holidays are assigned to `sunday` |
| `season` | text | `winter`, `mid` or `summer`, per `preprocessing.seasons` |
| `month` | int | Calendar month |
| `energy` | float | Daily energy, kWh, the row sum of `curves.npy` |
| `valid` | bool | True when the day carries no missing quarter-hour after gap filling |
| `is_zero_day` | bool | True when the day drew nothing. A zero day is an observation, not a hole |

### 3.3 `cache/synthgen/users.parquet`

One row per point.

| Column | Type | Description |
|---|---|---|
| `pod` | text | Point identifier |
| `power_kW` | float | Contractual power, the larger of the two declared values |
| `prosumer` | bool | Whether the point ever injects |
| `n_days_read` | int | Days read from the archive |
| `n_days_valid` | int | Days surviving pre-processing |
| `ateco_raw` | text | Activity code as registered |
| `ateco_l1`, `ateco_l2`, `ateco_l3` | text | The code truncated to 2, 4 and 6 digits. `None` where the code does not reach that depth |
| `domestic` | bool | True when the code begins with `DO` |
| `tariff`, `tariff_desc` | text | As registered |
| `power_meta_kW` | float | Contractual power as declared in the metadata workbook |
| `deactivated` | bool | True when the point was decommissioned |

### 3.4 `results/preprocessing/funnel.csv`

One row per filtering step, recording how many points and how many POD-days
survived it, with a note. Steps, in the order applied:

| Step | Effect |
|---|---|
| 0. as read | The archive as loaded, with the count of duplicates resolved |
| 1. daylight saving days removed | The two civil days that do not hold 96 quarter-hours |
| 2. readings above contractual power | Censored to missing. A quarter-hour of x kWh is a mean power of 4x kW, so the comparable limit is the contractual power divided by four, times `preprocessing.power_margin` |
| 3. zero runs classified | A run of zero days is a genuine closure when it consists of non-working days alone, or lasts at least `preprocessing.zero_run_hours`, or recurs at least `preprocessing.zero_run_min_recurrences` times. Runs failing all three are treated as faults and censored |
| 4. long gaps | Gaps up to `preprocessing.gap_short_max` quarter-hours are interpolated linearly; up to `preprocessing.gap_medium_max` they are completed with the median of the same quarter-hour on the same day type and season of the same point; beyond that the day is dropped |
| 5. days marked | Each day marked valid or not, and zero or not |

---

## 4. Stage 2: eligibility

**Command:** `python -m synthgen.eligibility`

**Reads:** `cache/synthgen/days.parquet`, `cache/synthgen/users.parquet`.

**Writes:**

| File | Content |
|---|---|
| `results/eligibility/eligible_pods.csv` | One row per point meeting the requirement, with its day counts, monthly coverage, energy, power, tariff and activity code |
| `results/eligibility/rejected_pods.csv` | The same, for points that do not, with the reason: `too few valid days` or `a calendar month is missing` |
| `results/eligibility/census_level{1,2,3}.csv` | Points per typology at each level, before and after the requirement, with the median contractual power |

The requirement is `eligibility.min_valid_days` distinct valid days, and, when
`eligibility.require_all_months` is enabled, every calendar month present with
at least `eligibility.min_days_per_month` of them.

---

## 5. Stage 3: estimation

**Command:** `python -m synthgen.estimate --all --level 1`

**Reads:** the three cache files.

**Writes** to `models/`, a top-level directory placed beside `results/` rather
than inside it, since it is the only artefact generation cannot rebuild for
itself and clearing the results of a run must not be able to remove it.

### 5.1 `models/<key>.npz`

One compressed NumPy archive per typology and size stratum. Keys are formed as
`L<level>_<typology>` for a single-stratum typology and `L<level>_<typology>_s<n>`
where a typology is split by size.

Notation: `S` is the number of states (positive bins plus zero), `C` the number
of momentum contexts, `R` the number of daily regimes, `B` the number of blocks
of hours, and `Kc` the number of closure classes.

| Array | Shape | Description |
|---|---|---|
| `trans` | `(R, seasons, daytypes, B, S·C, S)` | Quarter-hourly transition probabilities, after backoff |
| `regime_trans` | `(Kc, seasons, daytypes, R, R)` | Daily transition probabilities between regimes, one chain per closure class |
| `emission` | `(S, B, n_quantiles)` | Empirical quantiles of the readings observed in each state and block, used to return from a state to a value |
| `edges` | `(n_bins-1,)` | Interior bin edges, kWh per quarter-hour |
| `rho` | `(1,)` | AR(1) coefficient of the position within a bin |
| `closure_class` | `(n_pods,)` | Closure class of each point, aligned with `pods` |
| `closure_cuts` | `(Kc-1,)` | Thresholds on a point's own share of zero days that define the classes |
| `pods` | `(n_pods,)` | Identifiers of the points the model was estimated on |
| `annual_kWh` | `(n_pods,)` | Annual energy of each, the anchors a generated profile draws from |
| `power_kW` | `(n_pods,)` | Contractual power of each |
| `sample_pod`, `sample_power_kW`, `sample_annual_kWh` | `(n_sample,)` | A subset of the points, retained for the optional per-point chain |
| `pod_counts` | | Transition counts of that subset |
| `seasons`, `daytypes`, `regimes` | | Labels, in the order the axes above use |
| `n_blocks`, `n_ctx`, `flat_long`, `n_quarters` | `(1,)` | Structural parameters, stored so that generation never has to re-derive them |

The arrays `pods`, `annual_kWh`, `power_kW` and `closure_class` share one
ordering. `sample_pod` and its companions use a different one and must not be
indexed with a position taken from the first set.

### 5.2 `models/manifest.csv`

One row per model, describing what it was estimated on.

| Column | Description |
|---|---|
| `key` | Model identifier, matching the `.npz` filename |
| `level`, `typology`, `size_stratum` | What the model covers |
| `stratified_on` | `annual_kWh`, `power_kW`, or `none` where the typology was not split |
| `power_kW_min`, `power_kW_max` | Range of contractual power in the stratum |
| `annual_kWh_min`, `annual_kWh_max` | Range of annual energy in the stratum |
| `n_pods`, `n_pod_days` | Size of the estimation pool |
| `n_states`, `n_bins_effective` | Discretisation actually achieved |
| `zero_day_share` | Share of all days in the stratum that are at zero |
| `zero_day_share_median_pod` | Share of its own days that the median point of the stratum spends at zero. Where the two differ substantially, closure belongs to a minority of the points rather than to the population |
| `closure_class_shares` | Percentage of points in each closure class |
| `rho_within_bin` | The estimated AR(1) coefficient |
| `thin_months` | Calendar months whose transitions rest on the backoff rather than on their own observations |

---

## 6. Stage 4: generation

**Command:** see Section 6.4 of `README.md`.

**Reads:** `models/` and, when `--validation` is given, the three cache files as
well.

**Writes** to the directory given by `--outdir`:

### 6.1 `<typology>_<nnn>.csv`

One file per generated profile.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | ISO 8601, timezone aware | | Start of the interval, in the civil time of the requested year |
| `kWh` | float, six decimals | kWh | Energy drawn during the interval |

Timestamps represent the civil year accurately: the March daylight saving day
contains 92 quarter-hours and the October day 100. A year at quarter-hourly
resolution therefore contains 35,040 rows, and at hourly resolution 8,760.

Multiplying `kWh` by four yields the mean power in kW over a quarter-hour
interval; at hourly resolution the value in kWh is already the mean power in kW.

### 6.2 `generation_manifest.csv`

One row per profile, recording its provenance and every parameter drawn for it.

| Column | Description |
|---|---|
| `file` | Filename of the profile |
| `typology`, `level`, `model_key` | Which model produced it |
| `resolution`, `year` | As requested |
| `anchor_pod` | The real point whose size and closure class the profile inherited |
| `anchor_power_kW`, `anchor_annual_kWh` | Properties of that point |
| `unscaled_annual_kWh` | Annual energy the walk produced before scaling |
| `generated_annual_kWh` | Annual energy actually written. Equal to `anchor_annual_kWh` by construction |
| `peak_kW` | Highest mean power over an interval |
| `clipped_share` | Share of intervals limited by the contractual power. Values above a few per cent indicate a walk that reached levels the anchor could not support |
| `zero_share` | Share of intervals at zero |
| `clock_shift_quarters` | The profile's own clock offset, drawn once and held |
| `regularity` | The sharpening exponent drawn for the profile |
| `level_identity` | The share of within-bin variance held by the point |
| `closure_class` | Inherited from the anchor |
| `personal_chain` | Whether the optional per-point chain was used |

### 6.3 `validation/`

Written only with `--validation`.

| File | Content |
|---|---|
| `summary.csv` | One row per check, with the metered value and the synthetic one. The checks are listed in Section 8 of `README.md` |
| `total_variation_by_cell.csv` | Total variation of the mean day, by season and day type |
| `01_daily_energy.png` | Distribution of daily energy |
| `02_load_factor.png` | Distribution of the daily load factor, mean over peak |
| `03_peak_hour.png` | Hour at which the daily peak falls |
| `04_autocorrelation.png` | Autocorrelation against lag, with the daily and weekly lags marked |
| `05_mean_day.png` | Mean day by season and day type, with the metered interquartile band |
| `06_diversity.png` | Distance between the mean days of different points, within each population |
| `07_example_weeks.png` | A winter and a summer week for four profiles, each beside the metered point closest to it in annual energy |
| `08_load_duration.png` | Load duration curve |
| `09_annual_energy.png` | Distribution of annual energy per point |

---

## 7. Reproducibility

A generation run is fully determined by four things: the models in `models/`,
the configuration file, the command line arguments, and the seed given to
`--seed`. Two runs agreeing on all four produce byte-identical profiles.

Changing a parameter under `generation` in the configuration alters the output
without requiring re-estimation. Changing anything under `data`,
`preprocessing`, `eligibility` or `estimation` requires the affected stages to
be re-run, since their output is cached.

The `generation_manifest.csv` of a run records every parameter drawn per
profile, and `models/manifest.csv` records the estimation pool of every model.
Together they allow any profile to be traced back to the metered points it
derives from.
