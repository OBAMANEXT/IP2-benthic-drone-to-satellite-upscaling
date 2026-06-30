"""
EUNIS classification-scheme harmonization utilities.

Maps the 33-class EUNIS Level-2 (L2) benthic habitat scheme down to the
15-class EUNIS Level-1 (L1) scheme used as the common training target
across all Learning Sites. Also documents the site-specific Level-3 (L3)
mapping used for sites with custom/local classification schemes.

EUNIS is a standard, published European habitat classification system
(European Nature Information System, European Environment Agency) -- the
class catalogues and mapping logic below are not site-specific or
proprietary.
"""

from typing import Dict, List

import numpy as np


class ClassificationHarmonizer:
    """Map 33-class EUNIS L2 rasters to the 15-class EUNIS L1 scheme."""

    # L1 class catalogue (15 classes, IDs 1-15)
    L1_LABELS: Dict[int, str] = {
        1: "ALGAE",    2: "ANGIO",    3: "ANTHRO",   4: "BEACHCAST",
        5: "DEEP",     6: "GASTRO",   7: "MAERL",    8: "MUSSELS",
        9: "ROCK",    10: "SALTMARS", 11: "SEDIMENT", 12: "STARFISH",
       13: "TERR_VEG", 14: "URCHIN",  15: "WOOD",
    }

    # L2 class catalogue (33 classes, IDs 1-33)
    L2_LABELS: Dict[int, str] = {
        1: "ALGAE",    2: "ANGIO",    3: "ANTHRO",   4: "BEACH_AR",
        5: "BEACH_BR", 6: "BEACHCAST",7: "BEDROCK",  8: "BOULDER",
        9: "BROWN",   10: "COBBLE",  11: "DEEP",    12: "GASTRO",
       13: "GRAVEL",  14: "GREEN",   15: "INFRA",   16: "LITTER",
       17: "MAERL",   18: "MUD",     19: "MUSSELS", 20: "RED",
       21: "ROCK",    22: "SALTMARS",23: "SALTMARS", 24: "SALTMARS",
       25: "SALTMARS",26: "SAND",    27: "SEDIMENT", 28: "STARFISH",
       29: "TERR_VEG",30: "TURF",    31: "URCHIN",   32: "VEHICLES",
       33: "WOOD",
    }

    # L2 ID -> L1 ID conversion table
    L2_TO_L1: Dict[int, int] = {
        # Direct matches
        1: 1,  2: 2,  3: 3,   6: 4,  11: 5,  12: 6,  17: 7,  19: 8,
       21: 9, 28: 12, 29: 13, 31: 14, 33: 15,
        # Beach types
        4: 11,  5: 9,
        # Rock / substrate grain size
        7: 9,  8: 9,  10: 9,  13: 11, 18: 11, 26: 11, 27: 11,
        # Algae subtypes -> ALGAE
        9: 1, 14: 1, 20: 1, 30: 1,
        # Salt marsh duplicates
       22: 10, 23: 10, 24: 10, 25: 10,
        # Anthropogenic
       15: 3,  16: 3,  32: 3,
    }

    # Example site-specific L3 -> L1 mapping (illustrative; a real L3
    # scheme is defined per-site in the internal pipeline depending on
    # local survey conventions).
    L3_TO_L1: Dict[int, str] = {
        47: "ROCK",      # BEDROCK
        53: "SEDIMENT",  # SAND_SEA
        87: "ANGIO",     # seagrass (living)
        88: "ANGIO",     # seagrass (dead leaf litter)
        89: "ANGIO",     # seagrass (dead matter)
    }

    def __init__(self):
        self.l1_id_to_name = self.L1_LABELS
        self.l2_id_to_name = self.L2_LABELS
        self.l1_name_to_id = {v: k for k, v in self.L1_LABELS.items()}

    # -- Conversion -----------------------------------------------------

    def convert_l2_to_l1(self, l2_raster: np.ndarray) -> np.ndarray:
        """
        Convert a 2-D L2 raster (IDs 1-33) to an L1 raster (IDs 1-15).

        Parameters
        ----------
        l2_raster : ndarray, shape (H, W), dtype int

        Returns
        -------
        l1_raster : ndarray, shape (H, W), dtype int16
        """
        l1_raster = np.zeros_like(l2_raster, dtype=np.int16)
        present = np.unique(l2_raster[l2_raster > 0])

        stats: Dict[str, List] = {}
        for l2_id in present:
            if l2_id not in self.L2_TO_L1:
                print(f"  Warning: L2 class {l2_id} "
                      f"({self.l2_id_to_name.get(l2_id, '?')}) has no L1 mapping")
                continue
            l1_id = self.L2_TO_L1[l2_id]
            mask = l2_raster == l2_id
            l1_raster[mask] = l1_id
            l1_name = self.l1_id_to_name.get(l1_id, "?")
            l2_name = self.l2_id_to_name.get(l2_id, "?")
            stats.setdefault(l1_name, []).append((l2_name, int(mask.sum())))

        print("  L2 -> L1 conversion summary:")
        for l1_name in sorted(stats):
            total = sum(c for _, c in stats[l1_name])
            sources = ", ".join(f"{n}({c:,})" for n, c in stats[l1_name])
            print(f"    {l1_name:12s}: {total:>8,} px  <- {sources}")

        self._validate(l2_raster, l1_raster)
        return l1_raster

    def detect_level(self, data: np.ndarray) -> str:
        """
        Auto-detect classification level from max class ID.

        Thresholds
        ----------
        max <= 15 -> L1 (15 EUNIS classes, IDs 1-15)
        max <= 33 -> L2 (33 EUNIS classes, IDs 1-33)
        max > 33  -> L3 (site-specific scheme with arbitrary IDs)
        """
        valid = data[data > 0]
        if len(valid) == 0:
            return "L1"
        m = int(valid.max())
        if m <= 15:
            return "L1"
        if m <= 33:
            return "L2"
        return "L3"

    # -- Private ----------------------------------------------------------

    def _validate(self, l2: np.ndarray, l1: np.ndarray) -> None:
        n2, n1 = (l2 > 0).sum(), (l1 > 0).sum()
        if n2 == n1:
            print(f"  OK: {n2:,} pixels preserved")
        else:
            print(f"  Warning: {n2 - n1:,} pixels lost in conversion "
                  f"(L2={n2:,} -> L1={n1:,})")
