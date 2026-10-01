from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEM_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "tampa_topobathy"
    / "noaa_2019"
    / "2019_ngs_tampabay_topobath_dem_J1461017tR0_C0.tif"
)

FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Read DEM
# --------------------------------------------------

with rasterio.open(DEM_FILE) as src:
    elevation = src.read(1, masked=True)

    total_cells = elevation.size
    valid_cells = elevation.count()
    nodata_cells = total_cells - valid_cells

    pixel_width, pixel_height = src.res


# --------------------------------------------------
# Calculate data coverage
# --------------------------------------------------

valid_percent = (valid_cells / total_cells) * 100
nodata_percent = (nodata_cells / total_cells) * 100

print("--- Raster coverage ---")
print(f"Total cells: {total_cells:,}")
print(f"Valid cells: {valid_cells:,}")
print(f"NoData cells: {nodata_cells:,}")
print(f"Valid coverage: {valid_percent:.2f}%")
print(f"NoData coverage: {nodata_percent:.2f}%")


# --------------------------------------------------
# Elevation distribution
# --------------------------------------------------

valid_elevation = elevation.compressed()

percentiles = np.percentile(
    valid_elevation,
    [1, 5, 25, 50, 75, 95, 99],
)

print("\n--- Elevation percentiles ---")

for percentile, value in zip(
    [1, 5, 25, 50, 75, 95, 99],
    percentiles,
):
    print(f"{percentile:>2}th percentile: {value:8.2f} ft")


# --------------------------------------------------
# Elevations relative to NAVD88 zero
# --------------------------------------------------

below_zero = np.sum(valid_elevation < 0)
equal_zero = np.sum(valid_elevation == 0)
above_zero = np.sum(valid_elevation > 0)

print("\n--- Elevation relative to NAVD88 zero ---")

print(
    f"Below 0 ft: {below_zero:,} "
    f"({below_zero / valid_cells * 100:.2f}%)"
)

print(
    f"Equal to 0 ft: {equal_zero:,} "
    f"({equal_zero / valid_cells * 100:.2f}%)"
)

print(
    f"Above 0 ft: {above_zero:,} "
    f"({above_zero / valid_cells * 100:.2f}%)"
)


# --------------------------------------------------
# Plot elevation histogram
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    valid_elevation,
    bins=100,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
    label="0 ft NAVD88",
)

ax.set_title("Elevation Distribution - Tampa Bay Topobathymetric DEM")
ax.set_xlabel("Elevation (ft, NAVD88)")
ax.set_ylabel("Number of Raster Cells")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

output_file = FIGURES_DIR / "05_topobathy_elevation_histogram.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nFigure saved to: {output_file}")