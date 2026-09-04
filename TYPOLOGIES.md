# synthgen — Available user typologies

Companion document to `README.md`. It lists the user typologies for which a
model has been estimated on the supplied archive, and states for each what it
was estimated on and what is known about its behaviour.

| | |
|---|---|
| **Software** | synthgen, version 1.0 |
| **Author** | Lorenzo Giannuzzo, Politecnico di Torino, DENERG, Energy Center Lab |
| **Contact** | lorenzo.giannuzzo@polito.it |
| **Date** | September 2026 |
| **Source** | `models/manifest.csv` |

The table below reflects the models estimated on the archive supplied with the
project: 32 typologies, 48 models, 11,345 points and 3,977,536 metered days, all
at level 1. It must be regenerated whenever the models are re-estimated on a
different archive; the command is given in Section 5.

---

## 1. How a typology is requested

A typology is passed to the generator as the value of `--typology`, together
with the depth at which the activity code is read:

```bash
python -m synthgen.generate --typology 47 --ateco-level 1 --n 50 \
    --resolution 15min --year 2025 --outdir results/retail --seed 42
```

Three forms are accepted:

- **A numeric activity code**, truncated to the requested level. `47` at level 1,
  `47.71` at level 2, `47.71.10` at level 3.
- **`domestic`**, which covers both `DO.01` (non-resident) and `DO.02`
  (resident) and ignores the level.
- **`IL`**, public lighting, which likewise ignores the level.

A code may be truncated but not extended. A point registered as `56.2` therefore
answers at levels 1 and 2 and is absent from level 3. Requesting a typology at a
level deeper than its code reaches raises an explicit error naming the level at
which it can be answered.

Only level 1 models have been estimated on the supplied archive. Levels 2 and 3
are supported by the code but require Stage 3 to be re-run with `--level 2` or
`--level 3`, and will produce far fewer typologies, since the estimation pool of
each is correspondingly smaller.

---

## 2. Available typologies

Ordered by the size of the estimation pool. `Strata` is the number of models the
typology was split into by size; a typology with more than one is generated from
all of them in proportion to their populations.

| Typology | Description | Strata | Points | Metered days | Split on |
|---|---|---|---|---|---|
| `domestic` | Households, resident and non-resident | 3 | 10,009 | 3,520,454 | annual energy |
| `47` | Retail trade, except of motor vehicles | 3 | 171 | 55,873 | annual energy |
| `84` | Public administration and defence | 3 | 114 | 36,271 | contractual power |
| `55` | Accommodation | 3 | 103 | 37,488 | contractual power |
| `01` | Crop and animal production, hunting | 3 | 99 | 33,974 | contractual power |
| `IL` | Public lighting | 3 | 95 | 32,799 | contractual power |
| `56` | Food and beverage service activities | 3 | 90 | 29,936 | contractual power |
| `61` | Telecommunications | 2 | 63 | 21,860 | contractual power |
| `52` | Warehousing and support activities for transportation | 2 | 60 | 20,992 | contractual power |
| `96` | Other personal service activities | 1 | 49 | 15,946 | not split |
| `94` | Activities of membership organisations | 1 | 48 | 16,339 | not split |
| `43` | Specialised construction activities | 1 | 44 | 16,957 | not split |
| `49` | Land transport and transport via pipelines | 1 | 42 | 14,241 | not split |
| `16` | Manufacture of wood and of products of wood | 1 | 34 | 13,238 | not split |
| `41` | Construction of buildings | 1 | 34 | 10,226 | not split |
| `68` | Real estate activities | 1 | 32 | 11,850 | not split |
| `36` | Water collection, treatment and supply | 1 | 26 | 8,810 | not split |
| `CO` | See Section 4 | 1 | 25 | 9,056 | not split |
| `85` | Education | 1 | 21 | 7,671 | not split |
| `93` | Sports, amusement and recreation activities | 1 | 19 | 6,595 | not split |
| `74` | Other professional, scientific and technical activities | 1 | 18 | 5,439 | not split |
| `62` | Computer programming and consultancy | 1 | 18 | 6,383 | not split |
| `10` | Manufacture of food products | 1 | 15 | 5,202 | not split |
| `45` | Wholesale and retail trade and repair of motor vehicles | 1 | 15 | 6,469 | not split |
| `60` | Programming and broadcasting activities | 1 | 15 | 5,142 | not split |
| `70` | Activities of head offices, management consultancy | 1 | 14 | 3,834 | not split |
| `71` | Architectural and engineering activities | 1 | 14 | 4,620 | not split |
| `64` | Financial service activities | 1 | 13 | 4,257 | not split |
| `98` | Undifferentiated goods and services producing activities of private households for own use | 1 | 13 | 5,556 | not split |
| `35` | Electricity, gas, steam and air conditioning supply | 1 | 12 | 3,407 | not split |
| `25` | Manufacture of fabricated metal products | 1 | 10 | 4,099 | not split |
| `86` | Human health activities | 1 | 10 | 2,552 | not split |

Descriptions follow the NACE Rev. 2 division titles. `domestic` and `IL` are not
activity codes; see Section 4.

---

## 3. Behaviour by typology

Three quantities recorded per model govern how the generator behaves and how the
output should be read. All are in `models/manifest.csv`.

| Typology | Days at zero, pooled | Days at zero, median point | rho | Sharpening |
|---|---|---|---|---|
| `domestic` | 0.196 | 0.157 | 0.46 – 0.80 | off |
| `47` | 0.031 | 0.000 | 0.62 – 0.85 | default |
| `84` | 0.120 | 0.000 | 0.61 – 0.72 | default |
| `55` | 0.035 | 0.000 | 0.37 – 0.52 | default |
| `01` | 0.307 | 0.280 | 0.52 – 0.82 | off |
| `IL` | 0.028 | 0.000 | 0.79 – 0.93 | default |
| `56` | 0.046 | 0.000 | 0.34 – 0.41 | default |
| `61` | 0.095 | 0.000 | 0.78 – 0.89 | default |
| `52` | 0.230 | 0.000 | 0.80 – 0.91 | off |
| `96` | 0.146 | 0.000 | 0.65 | default |
| `94` | 0.088 | 0.000 | 0.84 | default |
| `43` | 0.051 | 0.000 | 0.64 | default |
| `49` | 0.147 | 0.000 | 0.69 | default |
| `16` | 0.077 | 0.000 | 0.71 | default |
| `41` | 0.314 | 0.282 | 0.63 | off |
| `68` | 0.137 | 0.000 | 0.69 | default |
| `36` | 0.048 | 0.000 | 0.86 | default |
| `CO` | 0.000 | 0.000 | 0.83 | default |
| `85` | 0.042 | 0.000 | 0.73 | default |
| `93` | 0.017 | 0.000 | 0.75 | default |
| `74` | 0.154 | 0.000 | 0.82 | default |
| `62` | 0.086 | 0.000 | 0.83 | default |
| `10` | 0.037 | 0.000 | 0.50 | default |
| `45` | 0.001 | 0.000 | 0.56 | default |
| `60` | 0.000 | 0.000 | 0.79 | default |
| `70` | 0.052 | 0.000 | 0.71 | default |
| `71` | 0.011 | 0.000 | 0.66 | default |
| `64` | 0.000 | 0.000 | 0.84 | default |
| `98` | 0.261 | 0.006 | 0.77 | off |
| `35` | 0.007 | 0.000 | 0.75 | default |
| `25` | 0.000 | 0.000 | 0.66 | default |
| `86` | 0.021 | 0.000 | 0.81 | default |

**Days at zero, pooled and per point.** The first is the share of all metered
days in the typology that are at zero; the second is the share of its own days
that the median point spends at zero. Where the two diverge, closure belongs to
a minority of the points rather than to the population, and the closure class
described in Section 3.4 of `README.md` is what separates them.

The divergence is largest in `domestic` (0.196 against 0.157 across the three
strata, and 0.196 against 0.004 when the strata are pooled), `52` (0.230 against
0.000) and `98` (0.261 against 0.006). In `01` and `41` the two figures agree,
which means closure in those typologies is seasonal and common to most points
rather than confined to a few.

**rho.** The persistence of the position within a discretised level. It governs
the load factor of the generated profiles, as stated in Section 10.2 of
`README.md`. Typologies at the low end of the range, `56` at 0.34 to 0.41 and
`55` at 0.37 to 0.52, describe genuinely irregular consumption and are the ones
the model reproduces most faithfully. Typologies at the high end, `IL` at 0.79
to 0.93 and `52` at 0.80 to 0.91, describe regular, cyclical loads and are the
ones on which the absence of a daily routine, Section 10.1 of `README.md`, is
most visible.

**Sharpening.** Whether the regularity exponent is applied, per
`generation.regularity_by_typology`. It is switched off for the five typologies
whose points spend a substantial share of the year at zero, since the exponent
increases that share further. `domestic` and `01` were measured; `41`, `52` and
`98` were set by extrapolation and are marked as untested in the configuration
file.

---

## 4. Typologies requiring interpretation

**`domestic`** covers the register codes `DO.01`, non-resident, and `DO.02`,
resident. It is the largest typology of the archive and the one whose population
is least homogeneous: the three size strata separate into 36 / 6 / 58 percent,
73 / 7 / 20 and 95 / 2 / 3 across the three closure classes, meaning that the
stratum of lowest annual energy consists in majority of points that close for
extended periods, while the stratum of highest annual energy consists almost
entirely of points that never stop drawing.

Section 10.5 of `README.md` records an unresolved question on the absolute level
of domestic consumption in this archive, which should be read before absolute
figures in kWh are taken from this typology.

**`IL`** denotes public lighting. It is not an activity code and is retained as
a typology in its own right. Its consumption is close to deterministic, opening
and closing at fixed times each day, and it is consequently the typology on
which the limitation described in Section 10.1 of `README.md` weighs most
heavily.

**`CO`** appears in the register with 25 points and no zero days at all. It is
not an activity code and its meaning has not been established from the archive
alone. It is retained so that its points are not silently discarded, but the
typology should be identified with the data owner before its output is used.

---

## 5. Regenerating this list

The table in Section 2 and the figures in Section 3 are derived from
`models/manifest.csv`. After re-estimating the models on a different archive,
they can be reproduced with:

```bash
python -c "import pandas as pd; m = pd.read_csv('models/manifest.csv', dtype={'typology': str}); print(m.groupby('typology').agg(strata=('key','size'), pods=('n_pods','sum'), days=('n_pod_days','sum'), split=('stratified_on','first'), zero_pooled=('zero_day_share','mean'), zero_median_pod=('zero_day_share_median_pod','mean'), rho_min=('rho_within_bin','min'), rho_max=('rho_within_bin','max')).sort_values('pods', ascending=False).to_string())"
```

The descriptions in Section 2 are NACE Rev. 2 division titles and are
independent of the archive.
