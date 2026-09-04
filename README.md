# synthgen

**Synthetic generation of quarter-hourly electrical load profiles by user typology**

---

## 1. Document and authorship

| | |
|---|---|
| **Software** | synthgen |
| **Version** | 1.0 |
| **Date** | September 2026 |
| **Author** | Lorenzo Giannuzzo |
| **Institution** | Politecnico di Torino |
| **Department** | Department of Energy "Galileo Ferraris" (DENERG) |
| **Research group** | Energy Center Lab |
| **Address** | Corso Duca degli Abruzzi 24, 10129 Torino, Italy |
| **Contact** | lorenzo.giannuzzo@polito.it |

This software was developed at Politecnico di Torino by Lorenzo Giannuzzo. The
methodology, the implementation and the validation framework described in this
document are the work of the author. Correspondence regarding the model, its
assumptions or its results should be addressed to the contact above.

All figures reported in this document were obtained from the archive supplied
with the project and are reproducible with the commands given in Section 6.

---

## 2. Scope and intended use

synthgen estimates a statistical model of electrical consumption from a set of
metered points sharing an activity code, and draws from that model new load
profiles covering a full civil year at quarter-hourly or hourly resolution.

The generated profiles are not copies of any metered point. They reproduce the
statistical properties of the population from which they are drawn: annual and
daily energy, peak power, load duration, and the daily and seasonal shape of
consumption.

The intended application is the simulation of network operation over realistic
and diversified portfolios of users, in cases where the number of available
metered points is smaller than the number of points a scenario requires.

Section 10 states the limits of validity of the model and must be read before
the output is used.

---

## 3. Model description

### 3.1 State and conditioning

The state variable is the power level drawn during a quarter of an hour,
discretised on the quantiles of the consumption observed within the typology.
The zero level is retained as a separate state rather than assigned to the
lowest bin, since a point drawing nothing is closed rather than consuming a
small amount.

Each state additionally carries a momentum term, defined as the direction of the
preceding step. A chain conditioned on the level alone underestimates
persistence and produces curves substantially more irregular than any metered
day.

Transitions are non-homogeneous in time. They are conditioned on four variables:
the regime of the day, the season, the day type, and the block of hours. Day
types are weekday, Saturday and Sunday, with Italian national holidays assigned
to Sunday.

### 3.2 The daily layer

A second chain operates at daily resolution over four regimes: closed, low,
medium and high. This layer differentiates one generated day from the next;
without it, every day of a synthetic year converges on the same average shape.

The regime of a day is determined relative to the ordinary day of the point
itself, not relative to the population. Daily energy varies considerably more
between points than within a single point, so terciles computed on pooled daily
energy would classify points by size rather than days by intensity of use.

### 3.3 Sparsity

Sparsity is handled by backoff. A cell whose transition row contains few
observations is interpolated toward its parent, defined as the same cell with
one conditioning variable removed. A row containing n transitions retains a
weight of n / (n + k) of its own distribution and takes the remainder from the
level above, where k is the shrinkage parameter. No row is left degenerate.

### 3.4 Point-level properties

Two properties belong to the point rather than to the day. Both are drawn once
per generated profile and held constant for the whole year.

**Size.** Each generated profile is anchored to one real point of the population
and reproduces its annual energy exactly.

**Closure class.** The propensity of a point to remain closed for an entire day
is taken from the same anchor. This conditioning is necessary because closure is
not uniformly distributed across a population. In the domestic typology of this
archive, 19.6 percent of all metered days are at zero, while the median domestic
point is at zero on 0.4 percent of its days: the zero days belong to a minority
of unoccupied dwellings and second homes, whereas an inhabited dwelling does not
stop drawing. A single chain estimated on the pooled population assigns that
19.6 percent to every generated profile.

### 3.5 Reconstruction of values

The return from a discretised bin to a power value is obtained by inverting the
empirical distribution of the readings observed within that bin, rather than by
uniform sampling inside the bin. The position within the bin follows an AR(1)
process, so that a point situated in the upper part of its bin tends to remain
there.

---

## 4. Installation

Python 3.12 is required. Development and validation were carried out on Python
3.12.2.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # Linux, macOS
pip install -r requirements.txt
```

The dependency list comprises six packages. Two of them, `pyarrow` and
`openpyxl`, do not appear in any import statement: pandas loads them internally
to read the Parquet cache and the metadata workbooks respectively. They are
required and must not be removed.

Installation may be verified with:

```bash
python -c "import synthgen.generate; print('ok')"
```

---

## 5. Input data specification

### 5.1 Directory structure

Measurements are read from the directory named by `data.root` in the
configuration file, resolved relative to the parent directory of the package.
The archive is organised as one directory per month, in any capitalisation:

```
data/
├── ago24/
│   ├── misure_ago24.csv
│   └── Metadati POD ago24.xlsx
├── Set24/
│   ├── misure_set24.csv
│   └── Metadati POD set24.xlsx
└── ...
```

### 5.2 Measurement file

The measurement file is semicolon separated, uses a decimal comma and dates in
`%d/%m/%Y` format, and carries a byte order mark on the first header. A single
POD-day is represented by several rows, one for each recorded quantity; only the
active withdrawn row, identified by `Tipologia == "AP"`, carries the consumption
curve.

| Column | Description |
|---|---|
| `POD` | Point identifier |
| `DataMisura` | Civil date |
| `Tipologia` | `AP` withdrawn, `AN` injected, `RLP` / `RLN` / `RCN` reactive |
| `PotenzaContrattuale` | Contractual power, kW |
| `K` | Meter constant |
| `Q1` … `Q96` | The ninety-six quarter-hours, in Wh |

Three characteristics of this format require specific handling, all of which is
implemented in `synthgen/io.py`.

The meter constant `K` equals 1 for the majority of points and 20, 25 or 40 for
those metered through a current transformer, whose readings are raw meter counts
rather than energy. The constant is applied per row rather than per point, since
a single point may change it within one month.

The columns `Q97` to `Q100` exist so that every row has uniform width. They are
zero padding and do not represent the additional quarter-hours of the October
daylight saving day. They are not read.

A limited number of POD-day pairs appear more than once. Duplicates are resolved
deterministically in favour of the row containing the greatest number of valued
quarter-hours, with ties broken on the record identifier.

### 5.3 Metadata workbook

The metadata workbook is an `.xlsx` register of the points. The following
columns are read: `POD`, `CCATETE` (activity code), `D_POTC` (contractual power,
expressed with a decimal point, unlike the measurement file), `D_TIPTA` and
`D_49DES` (tariff), and `D_DTSMO` (non-empty when the point has been
decommissioned).

### 5.4 Typologies

The activity code resolves into three levels. Their names in the code reflect
what they select rather than the column names of the source register:

| Identifier | Digits | Example | NACE level |
|---|---|---|---|
| `ateco_l1` | 2 | `47` | Division |
| `ateco_l2` | 4 | `47.71` | Group |
| `ateco_l3` | 6 | `47.71.10` | Class |

A code may be truncated but not extended. A point registered as `56.2` therefore
answers at levels 1 and 2 and is absent from level 3.

Two identifiers in the register are not activity codes. `IL` denotes public
lighting and `DO` denotes domestic use, with `DO.01` non-resident and `DO.02`
resident. Both are retained as typologies in their own right, under the names
`IL` and `domestic`, and are excluded from the numeric rollup.

---

## 6. Pipeline execution

The pipeline comprises four stages executed in sequence. Each stage caches the
output required by the following one, so that a stage need only be re-executed
when something upstream of it has changed.

### 6.1 Stage 1: pre-processing

```bash
python -m synthgen.preprocessing
```

Reads the archive and applies the following operations in order: removal of the
two daylight saving days, which cannot be represented on a 96-slot grid;
censoring of readings exceeding the contractual power; classification of zero
runs into genuine closures and instrumentation faults; filling of gaps according
to their length, with days exceeding the threshold discarded; and marking of
each day as valid or invalid.

Outputs: `cache/synthgen/curves.npy`, `cache/synthgen/days.parquet`,
`cache/synthgen/users.parquet`, and `results/preprocessing/funnel.csv`
documenting the effect of each step.

### 6.2 Stage 2: eligibility

```bash
python -m synthgen.eligibility
```

Determines which points hold a complete year of data and the resulting size of
each typology.

Outputs: `results/eligibility/eligible_pods.csv`,
`results/eligibility/rejected_pods.csv`, and a census per level.

### 6.3 Stage 3: estimation

```bash
python -m synthgen.estimate --all --level 1
python -m synthgen.estimate --typology 47 --level 1
```

Estimates one model per typology and size stratum.

Outputs: `models/<key>.npz` and `models/manifest.csv`.

The models are written to a top-level `models/` directory, placed beside
`results/` rather than inside it. Estimation is the only stage whose output
generation cannot rebuild for itself, so clearing the results of a run cannot
remove it.

This is the computationally expensive stage. The models it produces constitute
the input to generation and need not be rebuilt unless the underlying data or
the estimation parameters change.

### 6.4 Stage 4: generation

```bash
python -m synthgen.generate \
    --typology 47 --ateco-level 1 \
    --n 50 --resolution 15min --year 2025 \
    --outdir results/synthetic --seed 42 --validation
```

| Option | Description |
|---|---|
| `--typology` | Activity code, or `domestic`, or `IL` |
| `--ateco-level` | 1, 2 or 3: the depth at which the code is read |
| `--n` | Number of profiles to generate |
| `--resolution` | `15min` or `1h` |
| `--year` | Civil year to produce, including holidays and daylight saving days |
| `--outdir` | Destination directory |
| `--seed` | Fixes the random draw, making a run exactly reproducible |
| `--validation` | Additionally executes the comparison against the metered points |

The complete list of options is available through
`python -m synthgen.generate --help`.

---

## 7. Output specification

### 7.1 Profiles

One CSV file per profile, named `<typology>_<nnn>.csv`, containing two columns:

```csv
timestamp,kWh
2025-01-01 00:00:00+01:00,0.041
2025-01-01 00:15:00+01:00,0.038
```

Timestamps are timezone-aware and represent the civil year accurately. The March
daylight saving day therefore contains 92 quarter-hours and the October day 100.

Values represent the energy drawn during the corresponding interval, in kWh.
Multiplying by four yields the mean power in kW over that interval.

### 7.2 Generation manifest

`generation_manifest.csv` records the provenance of each profile, one row per
profile: the model key, the identity of the anchor point together with its
annual energy and contractual power, the generated annual energy, the peak, the
share of the year at zero, the clock shift, the closure class, and the value of
every per-profile parameter drawn. It constitutes the audit trail of the run.

---

## 8. Validation

When `--validation` is specified, the run writes a comparison against the
metered points of the same strata into `<outdir>/validation/`, comprising nine
figures, a `summary.csv` of scalar checks, and the total variation of the mean
day broken down by season and day type.

The metered reference is restricted to the points on which the selected models
were estimated, and is not rescaled. Since every generated profile carries the
annual energy of a real point belonging to that same set, the two populations
are already of comparable magnitude and any residual difference constitutes a
result rather than an artefact of normalisation.

| Check | Property assessed |
|---|---|
| Median daily energy | Whether the profiles consume the correct amounts |
| Median annual energy, and the ratio of its ninth decile to its first | Whether the population has the correct magnitude and dispersion. A set may exhibit the correct median while being substantially too wide |
| Median load factor | Whether consumption is distributed in a plausible shape, since a flat curve and a peaked curve may carry identical totals |
| Autocorrelation at 24 and 168 hours | Whether the daily and weekly periodicity is preserved |
| Day-to-day repeatability | Whether a point repeats its own routine. This cannot be established from the mean day |
| Share of the year at zero | Whether closures match those of the population |
| Ramp percentiles | The magnitude of step-to-step variation |
| 99th percentile power | Whether peaks are of the correct magnitude |
| Total variation of the mean day | Whether the seasonal and day-type conditioning is reproduced |

Two of these checks must be interpreted jointly rather than in isolation.
Day-to-day repeatability and the autocorrelation at 24 hours are both satisfied
by a constant series. A degenerate model would consequently score well on both
while failing every other check. They are informative only when read alongside
the share of the year at zero, the ramp percentiles and the total variation of
the mean day.

---

## 9. Configuration reference

All parameters are defined in `config_synthgen.yaml`, located in the parent
directory of the package. This file, and not any copy placed within the package
directory, is the one read at runtime.

| Parameter | Default | Description |
|---|---|---|
| `estimation.n_bins` | 20 | Number of positive levels into which the state is discretised, in addition to the zero state. Wide bins permit the value to vary within a single bin, which manifests as a load factor below the metered one |
| `estimation.n_blocks` | 24 | Number of blocks of hours on which transitions are conditioned. Coarser blocks allow an opening time to fall anywhere within a block, producing a correct mean day and an imprecise individual one |
| `estimation.shrinkage` | 30 | Backoff weight. A row containing n transitions retains n / (n + 30) of its own distribution |
| `output.models_dir` | `models` | Directory receiving the estimated models, resolved relative to the parent of the package |
| `estimation.closure_cuts` | [0.02, 0.20] | Thresholds on the share of zero days of an individual point, determining its closure class |
| `generation.profile_shift_quarters` | 6 | Maximum shift of the internal clock of a generated point, drawn once and held for the year, preventing all points of a stratum from opening at the same hour |
| `generation.regularity` | see file | Exponent, drawn per profile and held for the year, to which every transition row is raised before sampling. Values above one concentrate the row on its most probable destination, so that a point holds the level it has reached. It is applied off the diagonal and away from the zero state, and therefore modifies where the walk proceeds when it moves, never how frequently it stops |
| `generation.regularity_by_typology` | see file | Per-typology exceptions to the above, each accompanied in the file by the measurements justifying it |

---

## 10. Known limitations

The following are properties of the model as specified rather than defects
pending correction. They define the limits of validity of the output.

### 10.1 The daily routine is not reproduced

A metered point operating on a fixed schedule opens at the same time each
morning, and its consecutive days closely resemble one another. A chain
conditioned on the hour, the level and the momentum reproduces the statistics of
that schedule but not its repetition: conditional on the hour, each day
constitutes an independent realisation.

For the domestic typology, the total variation between the shape of one day and
that of the following day is 0.603 for the generated profiles against 0.409 for
the metered points, and the autocorrelation at 24 hours is 0.048 against 0.287.

The aggregate of many profiles is unaffected, since the mean day is reproduced
correctly. An individual profile examined in isolation will not resemble a meter
trace. Removing this limitation requires a conditioning variable that persists
across the day boundary, which the specified model does not include.

### 10.2 Load factor

The load factor responds principally to the persistence of the position within a
bin, which is estimated by pooling within-bin ranks across all points of a
stratum. This estimator conflates the persistence of an individual point with
the difference in magnitude between points.

For the domestic typology the generated median load factor is 0.122 against a
metered 0.211. Peak power and the load duration curve are unaffected and are
reproduced closely: the 99th percentile of power is 1.478 kW against a metered
1.520 kW.

### 10.3 Validation coverage

Only the domestic typology has been validated end to end against the current
models. The remaining 31 typologies have been estimated with identical code, but
their generated output has not been compared against the corresponding metered
points.

The per-typology exceptions defined in `generation.regularity_by_typology` rest
on measurements performed on two typologies and extrapolated to three further
ones. Each entry states its status in the configuration file.

### 10.4 Incomplete calendar coverage

In the archive supplied, February, April and May are present for 2024 only, and
December for 2025 only. Twenty-nine of the forty-eight estimated models
consequently contain at least one month whose transitions rest on the backoff
rather than on their own observations.

The affected months are identified per model in the `thin_months` column of
`models/manifest.csv` and are reported at the beginning of every generation
run.

### 10.5 Open question on the level of domestic consumption

The median annual energy of the domestic points in this archive is approximately
670 kWh, which is substantially below the two to three thousand kWh typically
drawn by an Italian household. The generated profiles reproduce this figure
faithfully, at approximately 590 kWh, so the discrepancy originates upstream of
the model.

Two explanations are consistent with the evidence available. The first is the
composition of the population, which contains a large share of unoccupied
dwellings and second homes. The second is the meter constant or the unit of the
readings on part of the archive.

This question has not been resolved. Absolute figures expressed in kWh derived
from the domestic typology should be regarded as provisional until it is.

---

## 11. Repository layout

```
.
├── config_synthgen.yaml    All parameters; the file read at runtime
├── requirements.txt        Dependencies, with tested versions
├── README.md               This document
├── synthgen/
│   ├── config.py           YAML loader, dotted-key access, path resolution
│   ├── io.py               Archive reader: encoding, meter constant, duplicates
│   ├── calendar.py         Day types, seasons, Italian holidays, DST days
│   ├── taxonomy.py         Activity codes, levels, census
│   ├── preprocessing.py    Stage 1
│   ├── eligibility.py      Stage 2
│   ├── estimate.py         Stage 3
│   ├── generate.py         Stage 4
│   └── validation.py       Comparison against the metered points
├── cache/synthgen/         Intermediate artefacts, rebuildable from Stage 1
├── models/                 Estimated models and their manifest (Stage 3)
└── results/
    ├── preprocessing/
    └── eligibility/
```
