"""
Full-scene prediction and habitat-map visualisation.

This module classifies and visualises full feature stacks. It operates on
already-extracted feature arrays plus an optional reference xarray.Dataset
(used only for spatial extent and an RGB backdrop) -- it does not perform
any raw-imagery loading, atmospheric correction, or drone-raster handling
itself.

If no reference dataset is available, ``plot_results`` falls back to a
two-panel figure (prediction only, no RGB backdrop).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .classifier import BenthicClassifier

# Default EUNIS L1 visualisation colours. Override via the `color_code`
# argument if you want a custom palette.
DEFAULT_COLOR_CODE: Dict[str, str] = {
    "ALGAE":     "#996633",
    "ANGIO":     "#A7FD67",
    "ANTHRO":    "#037AC9",
    "BEACHCAST": "#645210",
    "DEEP":      "#14436A",
    "GASTRO":    "#F0530A",
    "MAERL":     "#CAA2A9",
    "MUSSELS":   "#B8BBDE",
    "ROCK":      "#B2B2B2",
    "SALTMARS":  "#C7E5B3",
    "SEDIMENT":  "#ACF5D5",
    "STARFISH":  "#80258F",
    "TERR_VEG":  "#5E7403",
    "URCHIN":    "#868509",
    "WOOD":      "#F1DEC9",
}


class BenthicMapper:
    """Classify full feature stacks and create habitat maps."""

    def __init__(self, color_code: Optional[Dict[str, str]] = None):
        self.color_code = color_code or DEFAULT_COLOR_CODE

    # -- Full-scene prediction --------------------------------------------

    def predict_full_image(
        self,
        feature_stack: np.ndarray,
        classifier: BenthicClassifier,
        batch_size: int = 50_000,
    ) -> np.ndarray:
        """
        Classify every pixel in the feature stack.

        Parameters
        ----------
        feature_stack : (H, W, F) float32 ndarray
        classifier    : trained BenthicClassifier
        batch_size    : pixels per prediction batch

        Returns
        -------
        prediction_map : int16 ndarray, shape (H, W)
            Class index (0-based LabelEncoder output), -1 = nodata / NaN input.
        """
        print("\nPredicting full-scene classification ...")
        H, W, F = feature_stack.shape
        flat = feature_stack.reshape(-1, F)
        valid = ~np.isnan(flat).any(axis=1)
        preds = np.full(H * W, -1, dtype=np.int16)
        vidx = np.where(valid)[0]
        n_valid = len(vidx)

        print(f"  Processing {n_valid:,} valid pixels (batch={batch_size:,})")

        for start in range(0, n_valid, batch_size):
            idx = vidx[start:start + batch_size]
            enc_pred = classifier.label_encoder.transform(
                classifier.predict(flat[idx]))
            preds[idx] = enc_pred

            done = min(start + batch_size, n_valid)
            print(f"  {done:,} / {n_valid:,} ({100 * done / n_valid:.0f}%)",
                  end="\r")

        print()
        print("  Prediction complete")
        return preds.reshape(H, W)

    # -- RGB composite helper (optional, requires a reference dataset) ----

    @staticmethod
    def create_rgb_composite(ds, enhance: bool = True) -> np.ndarray:
        """
        Build a display-ready (H, W, 3) RGB array from a reference
        xarray.Dataset containing Rrs_* or rhos_* reflectance bands.

        This is a visualisation convenience only -- it is not required to
        run predictions or compute metrics.
        """
        def find(wls):
            for wl in wls:
                for prefix in ("Rrs_", "rhos_"):
                    name = f"{prefix}{wl}"
                    if name in ds.variables:
                        data = ds[name].values
                        return data / np.pi if prefix == "rhos_" else data
            return None

        r = find(["665", "664"])
        g = find(["560", "559"])
        b = find(["492", "490"])

        if r is None or g is None or b is None:
            H = ds.dims.get("y", 100)
            W = ds.dims.get("x", 100)
            return np.zeros((H, W, 3))

        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)
        rgb = np.clip(rgb, 0, None)

        if enhance:
            for i in range(3):
                band = rgb[:, :, i]
                valid = band[band > 0]
                if len(valid):
                    lo, hi = np.percentile(valid, [2, 98])
                    rgb[:, :, i] = np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            rgb = np.clip(rgb * 10, 0, 1)

        return rgb

    # -- Visualisation ------------------------------------------------------

    def plot_results(
        self,
        prediction_map: np.ndarray,
        classifier: BenthicClassifier,
        masked_prediction_map: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        extent: Optional[list] = None,
        region: str = "",
        output_path=None,
    ) -> None:
        """
        Plot classification results.

        Renders a 1-, 2-, or 3-panel figure depending on what's provided:
        RGB backdrop (optional) | full prediction | drone-extent mask (optional)

        Parameters
        ----------
        prediction_map         : int16 ndarray, output of predict_full_image
        classifier              : trained BenthicClassifier (for class names)
        masked_prediction_map   : optional second prediction map (e.g. masked
                                   to a reference/validation extent)
        rgb                     : optional (H, W, 3) RGB array, e.g. from
                                   create_rgb_composite
        extent                  : optional [xmin, xmax, ymin, ymax] for axes
        region                  : label used in the figure title
        output_path             : if given, saves the figure to this path
        """
        from matplotlib.colors import ListedColormap, BoundaryNorm

        class_names = classifier.class_names
        n_classes = len(class_names)

        colors = [
            self.color_code.get(str(cls).upper(), plt.cm.tab20(i % 20))
            for i, cls in enumerate(class_names)
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, n_classes), cmap.N)

        panels = []
        if rgb is not None:
            panels.append(("rgb", rgb))
        panels.append(("full", prediction_map))
        if masked_prediction_map is not None:
            panels.append(("masked", masked_prediction_map))

        fig, axes = plt.subplots(1, len(panels), figsize=(11 * len(panels), 11))
        if len(panels) == 1:
            axes = [axes]
        fig.suptitle(f"Benthic Habitat Classification — {region.upper()}".strip(" —"),
                     fontsize=16, fontweight="bold", y=1.01)

        for ax, (kind, data) in zip(axes, panels):
            if kind == "rgb":
                ax.imshow(data, extent=extent)
                ax.set_title("Reference RGB", fontsize=14, fontweight="bold")
            else:
                masked = np.ma.masked_where(data == -1, data)
                ax.imshow(masked, cmap=cmap, norm=norm,
                          interpolation="nearest", extent=extent)
                title = "Classification (full scene)" if kind == "full" \
                    else "Classification (masked extent)"
                ax.set_title(title, fontsize=14, fontweight="bold")

        handles = [
            mpatches.Patch(facecolor=c, edgecolor="k", linewidth=1.2,
                           label=str(cls).upper())
            for cls, c in zip(class_names, colors)
        ]
        axes[-1].legend(handles=handles, loc="center left",
                        bbox_to_anchor=(1.02, 0.5), fontsize=9,
                        framealpha=0.9, edgecolor="gray")

        for ax in axes:
            ax.set_xlabel("Easting (m)", fontsize=11)
            ax.set_ylabel("Northing (m)", fontsize=11)
            if extent:
                ax.ticklabel_format(style="plain", axis="both")
            ax.grid(True, alpha=0.3, linestyle="--", color="gray")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"  Figure saved -> {output_path}")
        plt.show()

    # -- Class areas --------------------------------------------------------

    @staticmethod
    def class_areas(
        prediction_map: np.ndarray,
        classifier: BenthicClassifier,
        pixel_size: float = 10.0,
    ) -> pd.DataFrame:
        """Return a DataFrame of per-class pixel counts and areas in hectares."""
        valid = prediction_map[prediction_map != -1]
        unique, counts = np.unique(valid, return_counts=True)
        px_ha = pixel_size ** 2 / 10_000
        rows = [
            {"Class": classifier.class_names[idx],
             "Pixels": int(cnt),
             "Area_ha": round(cnt * px_ha, 2),
             "Pct": round(100 * cnt / counts.sum(), 2)}
            for idx, cnt in zip(unique, counts)
            if 0 <= idx < len(classifier.class_names)
        ]
        return pd.DataFrame(rows).sort_values("Area_ha", ascending=False)
