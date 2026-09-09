# How to run

```bash
pip install -r requirements.txt
python main.py --stage preprocessing     # ~4 min, reads ../data
python main.py --stage clustering        # ~2 min, reads the cache
```

The clustering re-reads what preprocessing cached, so changing a clustering
parameter does not re-read the data.

## The configuration in config.yaml

This is the one that came out best on this dataset, and the numbers behind each
choice are below. Everything is in `config.yaml`; nothing is hard-coded.

| | | why |
|---|---|---|
| `shape_normalisation` | `unit_integral` | wider margin over the mean curves than min_max: 8.6x against 3.3x. Also required by Eq. 2, by the JSD of Eq. 8, and by the comparison with the national profiles, which are normalised on energy |
| `n_codewords` | `10` | at D=20 the user partition collapses to 0.042, below the mean curves it is meant to beat: more forms means more structural zeros, and the CLR drowns the informative coordinates |
| `scale_weight` | `0.5` | carries 2-3% of the distance, so size does not drive the partition. At 0.0 the groups are unchanged, which is what showed lambda was not the problem |
| `n_profiles` | `null` | chosen by the silhouette |
| `dictionary_method` | `summarised_ward` | Ward on every one of the 2.5M shapes, through a summary of fixed size. Ward on the raw pool would need 26 TB |

## What was tested, and settled

| hypothesis | verdict |
|---|---|
| lambda separates by size | no: at lambda=0 the groups are unchanged |
| a larger D helps | no: at D=20 the silhouette drops to 0.042 |
| min-max is the better normalisation | no: margin 3.3x against 8.6x, and it lifts the rival |
| the dictionary beats the mean curves | **yes: 0.432 against 0.155** |

## What the numbers say about the data

The silhouette of the user partition is weak in absolute terms, and no setting
fixes it. `dominant_share` runs from 0.205 to 0.877 without a break: the users
lie on a continuum, not in categories, and any K cuts it arbitrarily.

That is not a failure of the method. The mean curves, which is what the
regulator's construction amounts to, score 0.024: applied to its own data, it
separates nothing at all.
