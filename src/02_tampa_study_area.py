from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Study-area coordinates
# --------------------------------------------------

WEST = -82.600
SOUTH = 27.930
EAST = -82.540
NORTH = 27.990


# --------------------------------------------------
# Create study-area polygon
# --------------------------------------------------

study_area = box(
    WEST,
    SOUTH,
    EAST,
    NORTH,
)

aoi = gpd.GeoDataFrame(
    {"name": ["Tampa Bay AOI"]},
    geometry=[study_area],
    crs="EPSG:4326",
)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print(aoi)
print()
print("CRS:", aoi.crs)
print("Bounds:", aoi.total_bounds)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

aoi.boundary.plot(
    ax=ax,
    linewidth=2,
)

ax.set_title("Tampa Bay Study Area - AOI Version 1")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(alpha=0.3)

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

output_file = FIGURES_DIR / "02_tampa_study_area.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nFigure saved to: {output_file}")