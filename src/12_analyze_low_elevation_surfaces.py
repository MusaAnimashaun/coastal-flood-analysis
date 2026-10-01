from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

USGS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "usgs_1m_aoi.tif"
)

NOAA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "noaa_topobathy_aligned.tif"
)

FIGURE_DIR = PROJECT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Read aligned DEMs
# --------------------------------------------------

with rasterio.open(USGS_FILE) as usgs, rasterio.open(NOAA_FILE) as noaa:

    usgs_data = usgs.read(1, masked=True)
    noaa_data = noaa.read(1, masked=True)

    bounds = usgs.bounds


# --------------------------------------------------
# Common valid cells
# --------------------------------------------------

usgs_valid = ~np.ma.getmaskarray(usgs_data)
noaa_valid = ~np.ma.getmaskarray(noaa_data)

both_valid = usgs_valid & noaa_valid

usgs_values = usgs_data.data[both_valid]
noaa_values = noaa_data.data[both_valid]


# --------------------------------------------------
# Basic elevation distributions
# --------------------------------------------------

print("COMMON VALID CELLS")
print(f"Cells: {both_valid.sum():,}")


print("\nUSGS ELEVATION DISTRIBUTION")

for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    value = np.percentile(usgs_values, percentile)

    print(
        f"P{percentile:02d}:",
        f"{value:.3f} m",
    )


print("\nNOAA ELEVATION DISTRIBUTION")

for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    value = np.percentile(noaa_values, percentile)

    print(
        f"P{percentile:02d}:",
        f"{value:.3f} m",
    )


# --------------------------------------------------
# Count cells in useful elevation ranges
# --------------------------------------------------

ranges = [
    (-10.0, -2.0),
    (-2.0, -1.0),
    (-1.0, -0.5),
    (-0.5, 0.0),
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
]


print("\nELEVATION RANGE COUNTS")
print(
    f"{'Range (m)':<18}"
    f"{'USGS cells':>15}"
    f"{'NOAA cells':>15}"
)


for lower, upper in ranges:

    usgs_count = np.sum(
        (usgs_values >= lower)
        & (usgs_values < upper)
    )

    noaa_count = np.sum(
        (noaa_values >= lower)
        & (noaa_values < upper)
    )

    label = f"{lower:.1f} to {upper:.1f}"

    print(
        f"{label:<18}"
        f"{usgs_count:>15,}"
        f"{noaa_count:>15,}"
    )


# --------------------------------------------------
# Examine USGS cells close to its low-elevation mode
# --------------------------------------------------

# Previous inspection suggested many USGS cells are
# clustered near approximately -0.24 m NAVD88.
water_surface_mask = (
    both_valid
    & (usgs_data.data >= -0.30)
    & (usgs_data.data <= -0.18)
)

print("\nUSGS -0.30 TO -0.18 m BAND")
print(
    "Cells:",
    f"{water_surface_mask.sum():,}",
)

if water_surface_mask.any():

    corresponding_noaa = (
        noaa_data.data[water_surface_mask]
    )

    print(
        "Corresponding NOAA minimum:",
        f"{np.min(corresponding_noaa):.3f} m",
    )

    print(
        "Corresponding NOAA median:",
        f"{np.median(corresponding_noaa):.3f} m",
    )

    print(
        "Corresponding NOAA mean:",
        f"{np.mean(corresponding_noaa):.3f} m",
    )

    print(
        "Corresponding NOAA maximum:",
        f"{np.max(corresponding_noaa):.3f} m",
    )


# --------------------------------------------------
# Plot USGS low elevations
# --------------------------------------------------

usgs_low = np.where(
    both_valid & (usgs_data.data <= 1.0),
    usgs_data.data,
    np.nan,
)

plt.figure(figsize=(9, 8))

image = plt.imshow(
    usgs_low,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=-1,
    vmax=1,
)

plt.title("USGS Low-Elevation Surface (≤ 1 m NAVD88)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    image,
    label="Elevation (m, NAVD88)",
)

plt.tight_layout()

usgs_figure = (
    FIGURE_DIR
    / "12_usgs_low_elevation_surface.png"
)

plt.savefig(
    usgs_figure,
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Plot NOAA low elevations
# --------------------------------------------------

noaa_low = np.where(
    both_valid & (noaa_data.data <= 1.0),
    noaa_data.data,
    np.nan,
)

plt.figure(figsize=(9, 8))

image = plt.imshow(
    noaa_low,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=-7.5,
    vmax=1,
)

plt.title("NOAA Low-Elevation Topobathy (≤ 1 m NAVD88)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    image,
    label="Elevation (m, NAVD88)",
)

plt.tight_layout()

noaa_figure = (
    FIGURE_DIR
    / "12_noaa_low_elevation_surface.png"
)

plt.savefig(
    noaa_figure,
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Plot NOAA elevation where USGS is near -0.24 m
# --------------------------------------------------

comparison_map = np.full(
    usgs_data.shape,
    np.nan,
    dtype="float32",
)

comparison_map[water_surface_mask] = (
    noaa_data.data[water_surface_mask]
)

plt.figure(figsize=(9, 8))

image = plt.imshow(
    comparison_map,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=-7.5,
    vmax=1,
)

plt.title(
    "NOAA Elevation Where USGS Is Between -0.30 and -0.18 m"
)

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    image,
    label="NOAA elevation (m, NAVD88)",
)

plt.tight_layout()

comparison_figure = (
    FIGURE_DIR
    / "12_noaa_where_usgs_flat_water.png"
)

plt.savefig(
    comparison_figure,
    dpi=200,
)

plt.close()


print("\nSaved figures:")
print(usgs_figure)
print(noaa_figure)
print(comparison_figure)