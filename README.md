# Benthic Habitat Classification: Drone-to-Satellite Upscaling (Model & Evaluation Code)


This repository contains the **classifier training, evaluation, and
habitat-mapping** code, together with the **processed per-site sample
tables** needed to reproduce the quantitative results (F1-Macro, Cohen's
Kappa, confusion matrices) reported in the paper.

## What's in this repository

- `src/benthic_upscale/` — a small Python package:
  - `harmonizer.py` — EUNIS habitat classification scheme utilities (L1/L2/L3
    label catalogues and conversion logic). EUNIS is a published, standard
    European marine habitat classification system.
  - `classifier.py` — LightGBM multi-class classifier with training,
    evaluation (accuracy, F1-macro/weighted, Cohen's Kappa, confusion
    matrix), SHAP-free feature-importance plotting, and persistence.
  - `mapper.py` — full-scene prediction and habitat-map visualisation,
    operating on already-extracted feature arrays.
- `data/` — processed, pixel-level sample tables (spectral feature values +
  harmonized EUNIS L1 class label) for each Learning Site, ready to feed
  directly into `BenthicClassifier`.
- `notebooks/reproduce_results.ipynb` — loads a sample table, trains/evaluates
  the classifier, and reproduces the paper's metrics and figures.



## Data format

Each CSV in `data/` has one row per pixel-level training/test sample:

| column | description |
|---|---|
| `<feature columns>` | spectral feature values: multi-temporal marine indices (`_t0`...`_t3`) plus their per-pixel mean/std/range aggregates |
| `class` | harmonized EUNIS L1 habitat class name |
| `class_id` | EUNIS L1 numeric class ID |
| `region` | Learning Site identifier |
| `x_coord`, `y_coord` | pixel column/row in the satellite grid (for reference only) |

`data/ls8a_samples.csv` contains the real, processed training samples for
the **LS8A (Formentera)** Learning Site: 2,521 pixel samples across 3
classes (ANGIO, ROCK, SEDIMENT), 130 spectral feature columns. Running
`notebooks/reproduce_results.ipynb` on this file reproduces the paper's
reported LS8A metrics (Accuracy 0.834, F1-Macro 0.820, Cohen's Kappa 0.731).

Additional per-site tables (LS1, LS3A, LS3BN, LS3BS, LS3C) can be added
following the same schema.

## Installation

```bash
git clone <repo-url>
cd benthic-drone-to-satellite
pip install -r requirements.txt
```

## Reproducing results

```bash
cd notebooks
jupyter notebook reproduce_results.ipynb
```

Or use the package directly:

```python
from benthic_upscale import BenthicClassifier, BenthicMapper
import pandas as pd

samples = pd.read_csv("data/ls8a_samples.csv")
feature_cols = [c for c in samples.columns
                if c not in {"class", "class_id", "region", "x_coord", "y_coord"}]

X, y = samples[feature_cols].values, samples["class"].values
clf = BenthicClassifier(n_estimators=500, max_depth=15, learning_rate=0.05)
clf.train(X, y)
```



## Funding

This work was supported by the **OBAMA-NEXT** project, funded by the
European Union's Horizon Europe research and innovation programme
(Grant Agreement No. 101081642).

