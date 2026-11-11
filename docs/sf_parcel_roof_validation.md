# San Francisco parcel-aware roof validation

## Dataset

`data/sf_parcel_roof_review/` contains 60 georeferenced samples: 30 one-unit residential parcel candidates and 30 commercial or open-space/parking controls. North, central, and south San Francisco are assigned to train, validation, and sealed test splits (20 samples each). Each review image draws the selected parcel in yellow and its centroid in red; the corresponding clean chip is in `raw/`.

Parcel class is a sampling proxy, not a visual roof label. The `roof_label` fields are still blank and must be visually reviewed before this set can serve as verified roof/non-roof ground truth. A parcel classified as one-unit residential can still have a missing, obscured, or misclassified roof.

The chips use 2022 USGS NAIP imagery. Output pixels are enlarged to a nominal 0.12 m/pixel to approximate the app's zoom-20 view near San Francisco; the source is about 0.3 m/pixel, so this does not restore the detail available in higher-resolution imagery. Parcel classes come from [DataSF San Francisco Land Use](https://data.sf.gov/d/c5ge-t6pj); imagery comes from the [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer).

## Initial detector result

`scripts/evaluate_sf_parcel_roof_review.py` evaluates only train or validation and intentionally refuses test predictions. It checks whether highlighted pixels overlap the target parcel and requires at least 50% of the highlighted pixels to be inside that parcel. This is a diagnostic criterion, not a roof label: a colored roof mask may include pavement or another roof, while an actual roof can extend beyond parcel boundaries.

| Split | TP | FN | FP | TN | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6 | 4 | 4 | 6 | 0.60 |
| Validation | 2 | 8 | 4 | 6 | 0.40 |

At this setting validation recall is 0.20 and specificity is 0.60. Tightening the nearest-roof distance from 100 pixels to 35 pixels did not change the result. Sweeping the parcel-overlap threshold on train improved validation accuracy only by suppressing residential detections: at 0.70 overlap, validation was 1/10 residential candidates detected and 0/10 controls accepted (0.55 accuracy). That is not a useful residential roof detector.

Visual inspection of the marked validation chips shows roofs inside several selected house parcels, while the detector either highlights only part of the roof, highlights a region that crosses parcel edges, or emits a circular fallback marker. The current method is not reliable enough for deployment and does not establish that a highlighted roof belongs to a residential house.

## Next work before deployment

1. Visually label all train and validation samples as `roof`, `nonroof`, or `uncertain` using the parcel outline. Do not treat land-use class as the roof label.
2. Improve the detector so it returns a roof mask contained in the target building, or explicitly declines when target identity is ambiguous. Color segmentation alone cannot distinguish house roofs from commercial roofs or pavement reliably.
3. Re-run train and validation after freezing the change. Do not inspect test predictions until then; run the sealed test once.
4. Deploy only if validation supports the intended residential-house use, then inspect a small live sample before wider use.
