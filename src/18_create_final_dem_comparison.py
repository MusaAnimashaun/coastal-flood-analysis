from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

SCENARIO_A_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scenario_a_topographic.tif"
)

SCENARIO_B_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scenario_b_topobathy.tif"
)

FIGURE_DIR = PROJECT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    FIGURE_DIR
    / "18_final_dem_comparison.png"
)


# --------------------------------------------------
# Read Scenario A
# --------------------------------------------------

with rasterio.open(SCENARIO_A_FILE) as src:

    scenario_a = src.read(1, masked=True)
    bounds = src.bounds


# --------------------------------------------------
# Read Scenario B
# --------------------------------------------------

with rasterio.open(SCENARIO_B_FILE) as src:

    scenario_b = src.read(1, masked=True)


# --------------------------------------------------
# Use a common display range
#
# We limit the visualization to -5 to +10 m so
# coastal and low-lying terrain remains visible.
# This affects only the figure, not the raster data.
# --------------------------------------------------

vmin = -5
vmax = 10


# --------------------------------------------------
# Create side-by-side comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7),
    sharex=True,
    sharey=True,
)


# Scenario A
image_a = axes[0].imshow(
    scenario_a,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=vmin,
    vmax=vmax,
)

axes[0].set_title(
    "Scenario A: Topographic DEM"
)

axes[0].set_xlabel(
    "Easting (m)"
)

axes[0].set_ylabel(
    "Northing (m)"
)


# Scenario B
image_b = axes[1].imshow(
    scenario_b,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=vmin,
    vmax=vmax,
)

axes[1].set_title(
    "Scenario B: Bathymetry-Enhanced DEM"
)

axes[1].set_xlabel(
    "Easting (m)"
)


# --------------------------------------------------
# Shared colorbar
# --------------------------------------------------

colorbar = fig.colorbar(
    image_b,
    ax=axes,
    orientation="horizontal",
    fraction=0.05,
    pad=0.10,
)

colorbar.set_label(
    "Elevation (m NAVD88)"
)


# --------------------------------------------------
# Main title
# --------------------------------------------------

fig.suptitle(
    "Topographic and Bathymetry-Enhanced Elevation Surfaces",
    fontsize=14,
)


# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


print("FINAL DEM COMPARISON")
print("Scenario A shape:", scenario_a.shape)
print("Scenario B shape:", scenario_b.shape)
print("Display range:", vmin, "to", vmax, "m NAVD88")

print("\nSaved figure:")
print(OUTPUT_FILE)