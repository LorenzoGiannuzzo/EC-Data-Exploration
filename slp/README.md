# SLP framework

Implementation of the methodology of Section 2 of *What Do Standard Load Profiles
Actually Represent?*. One script per subsection, so that a claim in the paper and
the code that produces it are one file apart.

## Layout

```
slp/
├── config.yaml            every parameter the paper declares, and nothing else
├── main.py                runs the stages in order
├── preprocessing.py       Section 2.2   cleaning, filters, Eq. 1-2
├── clustering.py          Section 2.3   dictionary (Eq. 3) + users (Eq. 4-5), K, sensitivity
├── generation.py          Section 2.4   profiles (Eq. 6), 1000 kWh, dispersion
├── comparison.py          Section 2.5   nRMSD (Eq. 7), TV (Eq. 8), B1-B3, Eq. 13-14
├── mapping.py             Section 2.6   M1, M2, lift, M3 and reach (Eq. 9-12), null, K +/- 2
├── numerical_results.py   every number the paper quotes, in one table
├── figures.py             figures of Section 2.5 (and runs mapping_figures.py)
├── mapping_figures.py     figures of Section 2.6
├── selection.py           criteria for D and K
├── common/
│   ├── config.py          YAML loader, results folders, SLP_CONFIG override
│   ├── io.py              monthly folders; encoding, separator, decimal, meter constant
│   ├── cache.py           manifest of the configuration that wrote cache/
│   ├── ward.py            Ward linkage on weighted micro-clusters
│   ├── calendar.py        reference calendar, holidays, tariff bands
│   ├── national.py        ARERA and GSE profiles
│   └── assignment.py      national profile the regulation assigns to each POD
├── cache/                 .npy / .parquet handed from one stage to the next
└── paper_results/
    ├── numerical_results.csv
    ├── 1_preprocessing/
    ├── 2_clustering/
    ├── 3_standard_lp_generation/
    ├── 4_lp_comparison/
    └── 5_lp_mapping/{multiplicity,aggregation,coverage}/
```

## Running

```bash
python main.py                        # all stages
python main.py --stage preprocessing  # one stage
python main.py --from clustering      # from a stage onwards
python main.py --stage numbers        # rebuild numerical_results.csv only
python main.py --config other.yaml    # every stage reads other.yaml
```

Every stage reads what the previous one cached, so changing `lambda` and
re-running `clustering` does not re-read the data.

## Data

Read from `../data`, one folder per month named `<mesYY>` in any case
(`ago24`, `Ago25`), each holding `Metadati POD <mesYY>.xlsx` and `misure_*.csv`.
Encoding, field separator and decimal mark are detected per file. The price
series of Eq. 14 is a CSV declared under `comparison.price_file`.

## Things worth knowing

**ATECO levels.** The `CCATETE` code splits into three levels whose names do not
match NACE:

| in the code | digits | NACE level |
|---|---|---|
| `ateco_l1` | 2 | Division |
| `ateco_l2` | 4 | Class |
| `ateco_l3` | 6 | Subcategory |

**Eq. 1 and Eq. 2.** The annualised energy of Eq. 1 is the scale feature; the day
weights of Eq. 2 normalise on the observed energy so that they sum to one.

**The published catalogue.** Groups below `n_min` carry no profile. They are
excluded from generation, comparison and mapping alike, so every metric is
computed on the catalogue the paper publishes.

**Domestic.** Everywhere in the figures and in the text, domestic means an
activity label starting with `DO`. The tariff category of `assignment.py` is used
only to assign the national profile, and both shares are reported in
`4_lp_comparison/ddslp_composition.csv`.

**Eq. 14.** It needs a time-varying price. With a constant price the signed
discrepancy of every user-month sums to zero, so no fallback is provided.
