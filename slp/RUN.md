# How to run

```bash
pip install -r requirements.txt
python main.py                      # preprocessing -> figures, about 25 minutes
python main.py --from comparison    # after setting comparison.price_file
```

## The configuration in config.yaml

| | | why |
|---|---|---|
| `shape_normalisation` | `unit_integral` | wider margin over the mean curves than min_max; required by Eq. 2 and by the total variation of Eq. 8 |
| `dictionary_method` | `summarised_ward` | mini-batch k-means summarises the 2.5M shapes into `n_micro` micro-clusters; Ward runs on them weighted by cardinality |
| `n_codewords` | `null` | chosen on the quantization error: smallest D beyond which no codeword buys 1% or more, every codeword above 0.5% of the days (D = 19) |
| `scale_weight` | `0.5` | with block normalisation by variance the realised share equals the declared one |
| `n_profiles` | `null` | most stable K within the admissible range 6-20 (K = 7, mean ARI 0.54, below the 0.65 of Steinley 2004 and reported as such) |
| `generation.curve_weighting` | `energy` | cell curve and cell weight describe the same aggregate |
| `labels.min_class_size` | `25` | divisions below max(25, 3K) are pooled as minor classes |
| `comparison.price_file` | `null` | no zonal price series for 2025: Eq. 14 is not computed and is left to future work |

## Where the numbers come from

`paper_results/numerical_results.csv` holds every number the paper quotes, with
the section, the figure, the unit, the file it was read from and its definition.
