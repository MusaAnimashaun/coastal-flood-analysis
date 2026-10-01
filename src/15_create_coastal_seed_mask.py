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
    / "scenario_a_topographic.tif"
)

NOAA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scenario_b_topobathy.tif"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEED_FILE = OUTPUT_DIR / "coastal_seed_mask.tif"


# --------------------------------------------------
# Read both scenarios
# --------------------------------------------------

with rasterio.open(USGS_FILE) as src_a:

    scenario_a = src_a.read(1, masked=True)

    metadata = src_a.meta.copy()
    bounds = src_a.bounds
    transform = src_a.transform
    crs = src_a.crs
    nodata = src_a.nodata


with rasterio.open(NOAA_FILE) as src_b:

    scenario_b = src_b.read(1, masked=True)


# --------------------------------------------------
# Verify identical grids
# --------------------------------------------------

if scenario_a.shape != scenario_b.shape:
    raise ValueError(
        "Scenario A and Scenario B have different shapes."
    )


# --------------------------------------------------
# Valid-data mask
# --------------------------------------------------

valid_a = ~np.ma.getmaskarray(scenario_a)
valid_b = ~np.ma.getmaskarray(scenario_b)

both_valid = valid_a & valid_b


# --------------------------------------------------
# Recover bathymetry-replacement mask
#
# Scenario B differs from Scenario A only where
# NOAA bathymetry replaced the flattened USGS
# water representation.
# --------------------------------------------------

difference = (
    scenario_b.data.astype("float64")
    - scenario_a.data.astype("float64")
)

seed_mask = (
    both_valid
    & (difference < -0.001)
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

seed_cells = int(seed_mask.sum())

cell_area = abs(
    transform.a * transform.e
)

seed_area_m2 = seed_cells * cell_area
seed_area_km2 = seed_area_m2 / 1_000_000


print("COASTAL SEED MASK")
print("CRS:", crs)
print("Raster shape:", scenario_a.shape)
print("Seed cells:", f"{seed_cells:,}")
print("Cell area:", f"{cell_area:.3f} m²")
print("Seed area:", f"{seed_area_km2:.3f} km²")


# --------------------------------------------------
# Save binary seed raster
#
# 1 = coastal water seed
# 0 = non-seed valid cell
# 255 = NoData
# --------------------------------------------------

seed_output = np.zeros(
    scenario_a.shape,
    dtype="uint8",
)

seed_output[seed_mask] = 1

invalid = ~both_valid

seed_output[invalid] = 255


seed_metadata = metadata.copy()

seed_metadata.update(
    dtype="uint8",
    count=1,
    nodata=255,
)


with rasterio.open(
    SEED_FILE,
    "w",
    **seed_metadata,
) as dst:

    dst.write(seed_output, 1)

    dst.update_tags(
        description=(
            "Coastal water seed mask derived from cells "
            "where Scenario B contains substituted NOAA "
            "bathymetry relative to Scenario A."
        ),
        seed_value="1",
        non_seed_value="0",
        vertical_datum="NAVD88",
    )


# --------------------------------------------------
# Create QC figure
# --------------------------------------------------

display_mask = np.where(
    seed_mask,
    1.0,
    np.nan,
)


plt.figure(figsize=(9, 8))

plt.imshow(
    display_mask,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
)

plt.title("Coastal Water Seed Mask")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.tight_layout()


FIGURE_FILE = (
    FIGURE_DIR
    / "15_coastal_seed_mask.png"
)

plt.savefig(
    FIGURE_FILE,
    dpi=200,
)

plt.close()


print("\nSaved raster:")
print(SEED_FILE)

print("\nSaved figure:")
print(FIGURE_FILE)