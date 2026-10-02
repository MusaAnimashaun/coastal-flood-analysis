from pathlib import Path

import matplotlib.pyplot as plt
import rasterio


# Project paths

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
    / "11_dem_comparison.png"
)


# Check input files

if not SCENARIO_A_FILE.exists():
    raise FileNotFoundError(
        f"Scenario A raster not found: {SCENARIO_A_FILE}"
    )

if not SCENARIO_B_FILE.exists():
    raise FileNotFoundError(
        f"Scenario B raster not found: {SCENARIO_B_FILE}"
    )


# Read both scenarios

with rasterio.open(SCENARIO_A_FILE) as src_a, rasterio.open(
    SCENARIO_B_FILE
) as src_b:

    if (
        src_a.crs != src_b.crs
        or src_a.transform != src_b.transform
        or src_a.width != src_b.width
        or src_a.height != src_b.height
    ):
        raise ValueError(
            "Scenario A and Scenario B are not on the same grid."
        )

    scenario_a = src_a.read(1, masked=True)
    scenario_b = src_b.read(1, masked=True)

    bounds = src_a.bounds


# Use the same display range for both scenarios

vmin = -5
vmax = 10


# Plot both elevation surfaces

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7),
    sharex=True,
    sharey=True,
)

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


# Add one colorbar for both panels

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


fig.suptitle(
    "Topographic and Bathymetry-Enhanced Elevation Surfaces",
    fontsize=14,
)


# Save figure

plt.savefig(
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight",
)

plt.close()


print("DEM COMPARISON")
print("Scenario A shape:", scenario_a.shape)
print("Scenario B shape:", scenario_b.shape)
print("Display range:", vmin, "to", vmax, "m NAVD88")

print("\nSaved figure:")
print(OUTPUT_FILE)