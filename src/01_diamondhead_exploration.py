from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box


# --------------------------------------------------
# Project directories
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Important locations
# --------------------------------------------------

locations = {
    "Diamondhead": (-89.380025, 30.379432),
    "Rotten Bayou": (-89.392299, 30.381783),
    "Jourdan River": (-89.405474, 30.372861),
}


# --------------------------------------------------
# Convert locations to geographic features
# --------------------------------------------------

points = gpd.GeoDataFrame(
    {
        "name": locations.keys(),
        "geometry": [
            Point(longitude, latitude)
            for longitude, latitude in locations.values()
        ],
    },
    crs="EPSG:4326",
)


# --------------------------------------------------
# Preliminary study extent
# --------------------------------------------------

study_extent = box(
    -89.416,    # west
    30.3279,    # south
    -89.34967,  # east
    30.386,     # north
)

extent = gpd.GeoDataFrame(
    {"name": ["Preliminary study extent"]},
    geometry=[study_extent],
    crs="EPSG:4326",
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))

extent.boundary.plot(
    ax=ax,
    linewidth=2,
    linestyle="--",
)

points.plot(
    ax=ax,
    markersize=60,
)

for _, row in points.iterrows():
    ax.annotate(
        row["name"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.set_title(
    "Preliminary Study Area\n"
    "Diamondhead–Jourdan River, Mississippi"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(alpha=0.3)

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

output_file = FIGURES_DIR / "01_preliminary_study_area.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Figure saved to: {output_file}")