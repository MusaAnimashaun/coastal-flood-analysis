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
# Read both DEMs
# --------------------------------------------------

with rasterio.open(USGS_FILE) as usgs, rasterio.open(NOAA_FILE) as noaa:

    # Check that both rasters use exactly the same grid.
    same_crs = usgs.crs == noaa.crs
    same_size = (
        usgs.width == noaa.width
        and usgs.height == noaa.height
    )
    same_transform = usgs.transform == noaa.transform
    same_resolution = usgs.res == noaa.res

    print("GRID CHECK")
    print("Same CRS:", same_crs)
    print("Same size:", same_size)
    print("Same transform:", same_transform)
    print("Same resolution:", same_resolution)

    # Stop immediately if alignment failed.
    if not all(
        [
            same_crs,
            same_size,
            same_transform,
            same_resolution,
        ]
    ):
        raise ValueError("The two DEMs are not on the same grid.")

    usgs_data = usgs.read(1, masked=True)
    noaa_data = noaa.read(1, masked=True)

    bounds = usgs.bounds


# --------------------------------------------------
# Valid-data masks
# --------------------------------------------------

usgs_valid = ~np.ma.getmaskarray(usgs_data)
noaa_valid = ~np.ma.getmaskarray(noaa_data)

both_valid = usgs_valid & noaa_valid

total_cells = usgs_data.size

print("\nDATA COVERAGE")
print("Total grid cells:", f"{total_cells:,}")
print("USGS valid:", f"{usgs_valid.sum():,}")
print("NOAA valid:", f"{noaa_valid.sum():,}")
print("Both valid:", f"{both_valid.sum():,}")

print(
    "Common valid coverage:",
    f"{100 * both_valid.sum() / total_cells:.2f}%",
)


# --------------------------------------------------
# Compare likely terrestrial cells
# --------------------------------------------------

# Require both DEMs to be above 0 m NAVD88.
# This is only a preliminary land comparison mask.
land_mask = (
    both_valid
    & (usgs_data.data > 0)
    & (noaa_data.data > 0)
)

usgs_land = usgs_data.data[land_mask]
noaa_land = noaa_data.data[land_mask]

difference = noaa_land - usgs_land


print("\nPRELIMINARY LAND COMPARISON")
print("Cells compared:", f"{difference.size:,}")

print(
    "Mean NOAA - USGS:",
    f"{np.mean(difference):.3f} m",
)

print(
    "Median NOAA - USGS:",
    f"{np.median(difference):.3f} m",
)

print(
    "RMSE:",
    f"{np.sqrt(np.mean(difference ** 2)):.3f} m",
)

print(
    "5th percentile:",
    f"{np.percentile(difference, 5):.3f} m",
)

print(
    "95th percentile:",
    f"{np.percentile(difference, 95):.3f} m",
)


# --------------------------------------------------
# Plot common valid-data coverage
# --------------------------------------------------

coverage = np.zeros(usgs_data.shape, dtype=np.uint8)

coverage[usgs_valid] = 1
coverage[noaa_valid] = 2
coverage[both_valid] = 3

plt.figure(figsize=(9, 8))

plt.imshow(
    coverage,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
)

plt.title("DEM Data Coverage")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    label="0 = No data, 1 = USGS only, 2 = NOAA only, 3 = Both"
)

plt.tight_layout()

coverage_file = (
    FIGURE_DIR
    / "11_dem_data_coverage.png"
)

plt.savefig(
    coverage_file,
    dpi=200,
)

plt.close()


# --------------------------------------------------
# Plot land elevation differences
# --------------------------------------------------

difference_map = np.full(
    usgs_data.shape,
    np.nan,
    dtype="float32",
)

difference_map[land_mask] = (
    noaa_data.data[land_mask]
    - usgs_data.data[land_mask]
)

plt.figure(figsize=(9, 8))

image = plt.imshow(
    difference_map,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)

plt.title("NOAA - USGS Elevation Difference")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    image,
    label="Elevation difference (m)",
)

plt.tight_layout()

difference_file = (
    FIGURE_DIR
    / "11_land_elevation_difference.png"
)

plt.savefig(
    difference_file,
    dpi=200,
)

plt.close()


print("\nSaved figures:")
print(coverage_file)
print(difference_file)