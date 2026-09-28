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
├── clustering.py          Section 2.3   dictionary (Eq. 3) + users (Eq. 4-5)
├── generation.py          Section 2.4   profiles (Eq. 6), 1000 kWh, dispersion
├── comparison.py          Section 2.5   five measures (Eq. 7-9), assignment
├── mapping.py             Section 2.6   M1, M2, M3 (Eq. 10-14) + impact (Eq. 15-16)
├── common/
│   ├── config.py          YAML loader, path resolution
│   └── io.py              monthly folders; sniffs encoding, separator, decimal
├── cache/                 .npy / .parquet handed from one stage to the next
└── paper_results/
    ├── preprocessing_results/
    ├── clustering_results/
    ├── generation_results/
    ├── comparison_results/
    └── mapping_results/
```

## Running

```bash
python main.py                        # all stages
python main.py --stage preprocessing  # one stage
python main.py --from clustering      # from a stage onwards
```

Every stage reads what the previous one cached, so changing `lambda` and
re-running `clustering` does not rebuild the dictionary.

## Data

Read from `../data`, one folder per month named `<mesYY>` in any case
(`ago24`, `Ago25`), each holding `Metadati POD <mesYY>.xlsx` and `misure_*.csv`.
Encoding, field separator and decimal mark are detected per file.

## Two things worth knowing

**ATECO levels.** The `CCATETE` code splits into three levels whose names do not
match NACE:

| in the code | digits | NACE level |
|---|---|---|
| `ateco_l1` | 2 | Division |
| `ateco_l2` | 4 | Class |
| `ateco_l3` | 6 | Subcategory |

The paper's "division level" is therefore `ateco_l1`, set in `config.yaml` as
`labels.level: 1`.

**Eq. 1 and Eq. 2.** The annualised energy of Eq. 1 is the scale feature; the day
weights of Eq. 2 normalise on the observed energy so that they sum to one. Using
the annualised E as their denominator would make them sum to |D_i|/365.
