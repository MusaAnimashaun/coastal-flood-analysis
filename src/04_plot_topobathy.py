from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent


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
    extent = plotting_extent(src)

    crs = src.crs


# --------------------------------------------------
# Limit extreme values for visualization
# --------------------------------------------------

vmin = np.percentile(elevation.compressed(), 2)
vmax = np.percentile(elevation.compressed(), 98)

print("CRS:", crs)
print(f"Plot range: {vmin:.2f} to {vmax:.2f} ft")


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(
    elevation,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=vmin,
    vmax=vmax,
)

colorbar = plt.colorbar(
    image,
    ax=ax,
    shrink=0.8,
)

colorbar.set_label("Elevation (ft, NAVD88)")

ax.set_title(
    "2019 NOAA Topobathymetric DEM\n"
    "Tampa Bay, Florida"
)

ax.set_xlabel("Easting (US survey feet)")
ax.set_ylabel("Northing (US survey feet)")

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

output_file = FIGURES_DIR / "04_tampa_topobathy.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Figure saved to: {output_file}")