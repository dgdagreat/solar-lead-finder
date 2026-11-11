# California transfer evaluation (pilot)

## Purpose

This pilot checks whether the deployed roof gate and solar CNN transfer from
Queens 2018 orthophotography to independent California aerial imagery. It is
not a training set and does not measure performance on live Google imagery.

## Data and labeling

The source imagery is 30 cm USGS orthophotography from Fresno, Modesto, Oxnard,
and Stockton. The solar locations come from the manually annotated California
PV dataset by Bradbury et al. The dataset contains independently drawn PV
polygons and reports annotation agreement; its metadata is CC0 and the article
is CC BY 4.0. See the [paper and data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC5148580/).

The local, ignored `data/ca_external_eval_v1/` directory contains four source
tiles, 98 crops centered on annotated PV locations, and a manually reviewed set
of 11 no-visible-panel roof crops. The latter were selected from 200 candidates
only when the center was visibly on a roof and the crop showed no panels. The
full candidate pools include streets, trees, water, and open lots and must not
be treated as negatives. Source imagery and generated crops are not committed.

## Pilot results

The CNN was run with its existing 0.42 cutoff. The existing roof verifier was
also run first, as it is in the app. Solar labels are from the published PV
annotations; no-solar labels are based on visual review of the 11 selected
roof crops.

| Measure | Result |
|---|---:|
| Solar crops flagged by CNN | 51 / 98 (52.0% recall) |
| No-solar roof crops correctly rejected by CNN | 7 / 11 (63.6% specificity) |
| Balanced accuracy, CNN only | 57.8% |
| Solar crops accepted by both roof gate and CNN | 25 / 98 (25.5%) |
| Solar crops accepted by roof gate | 39 / 98 (39.8%) |
| Visually verified no-solar roofs accepted by roof gate | 7 / 11 (63.6%) |

Per-city CNN solar hits were Fresno 19/25, Modesto 14/25, Oxnard 7/23, and
Stockton 11/25. The threshold was not tuned on this set. The 11-sample
specificity estimate is very imprecise, and the source imagery is older than
current imagery. Treat these as a transfer warning, not as a production
accuracy claim.

## Decision

Do not retrain from this pilot. It is too small and is held out for evaluation.
The measured transfer is poor, and the roof gate rejects many annotated solar
locations before the CNN can score them. Next work should build a separate,
larger California training set with roof-centered positives and negatives,
keep geographically separate validation and test splits, and improve roof
verification using those labels. Do not use Google Maps imagery to train,
validate, or test: the current [Google Maps Platform terms](https://cloud.google.com/maps-platform/terms)
restrict use of Maps Content to improve ML/AI models.

The current CNN's Queens test result remains 81.5% balanced accuracy at the
0.42 cutoff, but that does not describe California or live-provider performance.
See [the Queens evaluation](solar_cnn_external_evaluation.md).
