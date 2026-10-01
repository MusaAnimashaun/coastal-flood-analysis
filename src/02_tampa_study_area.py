from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Study-area coordinates
# --------------------------------------------------

coordinates = [
    (-82.6904, 28.0278),
    (-82.6731, 28.0020),
    (-82.6924, 27.9514),
    (-82.7298, 27.98267),
]


# --------------------------------------------------
# Create study-area polygon
# --------------------------------------------------

study_area = Polygon(coordinates)

aoi = gpd.GeoDataFrame(
    {"name": ["Tampa Bay AOI - Version 2"]},
    geometry=[study_area],
    crs="EPSG:4326",
)


# --------------------------------------------------
# Display information
# --------------------------------------------------

print(aoi)

print("\nCRS:", aoi.crs)
print("Bounds:", aoi.total_bounds)
print("Polygon valid:", study_area.is_valid)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

aoi.boundary.plot(
    ax=ax,
    linewidth=2,
)

ax.scatter(
    [point[0] for point in coordinates],
    [point[1] for point in coordinates],
)

ax.set_title("Tampa Bay Study Area - AOI Version 2")
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