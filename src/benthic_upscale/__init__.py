"""
benthic_upscale
================

Lightweight, reproducible companion package for:

    "Upscaling Benthic Habitat Classification from Drone to Satellite
    Imagery Using Machine Learning"

This package contains the **model-training, evaluation, and mapping**
components of the full drone-to-satellite benthic classification pipeline.
It is designed to run directly on the processed per-site sample tables
shipped in ``data/`` -- no raw Sentinel-2 scenes, ACOLITE atmospheric
correction, or drone-raster pre-processing is required.

Modules
-------
harmonizer  : EUNIS classification-scheme utilities (L1/L2/L3 label maps)
classifier  : LightGBM multi-class classifier with evaluation utilities
mapper      : Full-scene prediction and habitat-map visualisation

Note
----
The feature-extraction and drone-sample-extraction stages used to produce
the sample tables in ``data/`` are part of the institute's internal
pipeline and are not included in this repository. See the README for
details on the processed-data format and how to cite this work.
"""

from .harmonizer import ClassificationHarmonizer
from .classifier import BenthicClassifier
from .mapper import BenthicMapper

__all__ = ["ClassificationHarmonizer", "BenthicClassifier", "BenthicMapper"]
__version__ = "0.1.0"
