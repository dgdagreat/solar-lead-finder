# Solar classifier model data attribution

`solar_cnn_queens2018.onnx` was trained on image crops and rooftop PV labels
derived from the Queens 2018 aerial imagery dataset:

- Furedi et al., “Labeled photovoltaic installations for orthographic aerial
  imagery in Queens, New York,” *Scientific Data* 13, 207 (2026),
  [paper](https://www.nature.com/articles/s41597-025-06523-2),
  [dataset](https://zenodo.org/records/15084216).
- The authors' PV annotations are licensed CC BY 4.0. The reproduced 2018
  imagery is also identified as CC BY 4.0 in the dataset paper.
- Imagery service attribution: NYS ITS Geospatial Services / Westchester County
  GIS.

Training used 200 balanced images. Model selection used a separate 200-image
validation set. The 200-image test set was held out by 5,000-foot geographic
blocks. The earlier 40-roof evaluation examples were excluded from all splits.
No Google Maps imagery was used to train, validate, or test this model.
