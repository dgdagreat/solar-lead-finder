# External solar classifier evaluation

## Dataset and split

The evaluation uses the Queens 2018 aerial PV dataset. Its authors manually
outlined about 14,000 PV polygons and reviewed the annotations as a three-person
team. Roof chips were assembled from the matching New York State 2018 orthophoto
tiles. Negative chips were centered on building footprints with no PV annotation
on or within 75 feet of the roof. Chips were balanced by class and separated by
5,000-foot geographic blocks:

| Split | Solar | No solar | Total |
|---|---:|---:|---:|
| Train | 100 | 100 | 200 |
| Validation | 100 | 100 | 200 |
| Test | 100 | 100 | 200 |

The previously evaluated 20 positive source tiles and 20 negative building
footprints were excluded from these splits. No Google Maps imagery was used for
training, validation, or testing.

## Held-out test results

The classifier checkpoint was selected by validation balanced accuracy. At a
0.50 cutoff, the 200-image test score was 79.5% accuracy and balanced accuracy
(Wilson 95% interval: 73.4–84.5%). The deployed 0.42 cutoff, selected on
validation only, scores 81.5% accuracy and balanced accuracy on that same test
set (Wilson 95% interval: 75.5–86.3%).

| Model / cutoff | True negatives | False positives | False negatives | True positives | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Heuristic | 24 | 76 | 60 | 40 | 32.0% | 32.0% |
| MobileNetV3 / 0.50 | 87 | 13 | 28 | 72 | 79.5% | 79.5% |
| MobileNetV3 / 0.42 | 86 | 14 | 23 | 77 | 81.5% | 81.5% |

The deployed cutoff of 0.42 was selected by maximizing balanced accuracy on the
200-image validation split (0.65 at the selected cutoff, versus 0.645 at 0.50).
On the untouched test set the cutoff gives 77% recall and 86% specificity, with
163/200 correct. The ONNX export matched the PyTorch probabilities across all
200 test images; maximum probability difference was 0.00005. Scores are not
calibrated probabilities.

## Limits

This measures roof-centered classification on 2018 Queens imagery. It does not
measure address-centering error, other cities or image providers, newer imagery,
or live Google Maps scan accuracy. The model score is not probability-calibrated.
Google Maps imagery was not used to train, test, or validate the model, consistent
with the [Google Maps Platform terms](https://cloud.google.com/archive/maps-platform/terms-20250604).

Source: [Zenodo dataset](https://zenodo.org/records/15084216) and
[Scientific Data paper](https://www.nature.com/articles/s41597-025-06523-2).
