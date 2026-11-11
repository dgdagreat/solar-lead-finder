# California residential-roof review v2

## Target definition

The positive class is a **residential house roof**. Businesses, shopping centers, warehouses, roads, vegetation, and open land are negative even when commercial roofs are visible. Solar presence is not labeled or inferred.

## Data and limitations

The local batch has 48 640 × 640 NAIP chips at 0.3 m per pixel from 2022. Cities were split before review: Riverside and Bakersfield (train), San Jose and Long Beach (validation), and Oakland and San Diego (test). The chips are centered on sampled coordinates, not confirmed addresses, parcels, or selected buildings. Labels are provisional visual judgments.

Review showed that the sample design is insufficient for property-level evaluation: LB-R-02 is a shopping-center lot, SJ-R-04 is a business, and the reviewer could not identify the target in LB-R-04. These are now respectively negative, negative, and uncertain for the residential-house target class. Clear commercial roofs in B-C-04, B-R-03, SJ-C-01, O-R-02, and O-R-04 are also negative. LB-C-03 and LB-C-04 remain uncertain; SJ-R-02 is nonroof because its center is tree cover.

An earlier coordinate-level run was 86.7% on the small train subset and 78.6% on the then-labeled validation subset. Those figures are **superseded and must not be treated as residential property-level accuracy**: the prior labels counted commercial roofs as positives and used a coordinate center without parcel boundaries. No valid property-level score is currently available. Do not deploy based on this batch.

## Next step

Build a parcel-aware review set containing identifiable single-family homes and matched nonresidential controls, with labels for the building containing the target address point. Keep the city-level test areas sealed until the detector is frozen. Then evaluate the residential-house gate on validation and only run the holdout once.

## Reproduce exploratory coordinate-level results

The evaluator reads labels in `data/ca_roof_review_v2/manifest.csv` and reports roof/nonroof classification for train or validation only:

```sh
venv/bin/python scripts/evaluate_ca_roof_review.py --split validation
```

Its output is exploratory and is not a property-level performance measure. It intentionally refuses the test split.
