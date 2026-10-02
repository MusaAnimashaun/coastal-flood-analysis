from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Project paths

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

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_A_FILE = OUTPUT_DIR / "scenario_a_topographic.tif"
SCENARIO_B_FILE = OUTPUT_DIR / "scenario_b_topobathy.tif"


# Check input files

if not USGS_FILE.exists():
    raise FileNotFoundError(f"USGS raster not found: {USGS_FILE}")

if not NOAA_FILE.exists():
    raise FileNotFoundError(f"NOAA raster not found: {NOAA_FILE}")


# Read aligned elevation surfaces

with rasterio.open(USGS_FILE) as usgs, rasterio.open(NOAA_FILE) as noaa:

    if (
        usgs.crs != noaa.crs
        or usgs.transform != noaa.transform
        or usgs.width != noaa.width
        or usgs.height != noaa.height
    ):
        raise ValueError("USGS and NOAA rasters are not on the same grid.")

    usgs_data = usgs.read(1, masked=True)
    noaa_data = noaa.read(1, masked=True)

    metadata = usgs.meta.copy()
    bounds = usgs.bounds
    nodata = usgs.nodata


# Find cells with valid data in both DEMs

usgs_valid = ~np.ma.getmaskarray(usgs_data)
noaa_valid = ~np.ma.getmaskarray(noaa_data)

both_valid = usgs_valid & noaa_valid


# Identify the flattened-water signature in the USGS DEM

flat_water_mask = (
    both_valid
    & (usgs_data.data >= -0.30)
    & (usgs_data.data <= -0.18)
    & (noaa_data.data < usgs_data.data)
)

print("SCENARIO CONSTRUCTION")
print(
    "Flattened-water cells replaced:",
    f"{flat_water_mask.sum():,}",
)

if not np.any(flat_water_mask):
    raise ValueError("No flattened-water cells were identified.")


# Scenario A uses the original USGS DEM

scenario_a = usgs_data.filled(
    nodata
).astype("float32")


# Scenario B replaces identified water cells with NOAA bathymetry

scenario_b = scenario_a.copy()

scenario_b[flat_water_mask] = (
    noaa_data.data[flat_water_mask]
)


# Save Scenario A

with rasterio.open(
    SCENARIO_A_FILE,
    "w",
    **metadata,
) as dst:

    dst.write(scenario_a, 1)

    dst.update_tags(
        scenario="A",
        description="USGS 3DEP 1 m topographic DEM",
        vertical_datum="NAVD88",
        vertical_units="meters",
    )


# Save Scenario B

with rasterio.open(
    SCENARIO_B_FILE,
    "w",
    **metadata,
) as dst:

    dst.write(scenario_b, 1)

    dst.update_tags(
        scenario="B",
        description=(
            "USGS terrestrial surface with NOAA "
            "topobathymetry substituted in identified "
            "flattened-water cells"
        ),
        vertical_datum="NAVD88",
        vertical_units="meters",
    )


# Quantify the bathymetry replacement

replacement_values = noaa_data.data[flat_water_mask]
original_values = usgs_data.data[flat_water_mask]

change = replacement_values - original_values

print("\nREPLACED CELLS")

print(
    "USGS median:",
    f"{np.median(original_values):.3f} m",
)

print(
    "NOAA median:",
    f"{np.median(replacement_values):.3f} m",
)

print(
    "Median elevation change:",
    f"{np.median(change):.3f} m",
)

print(
    "Mean elevation change:",
    f"{np.mean(change):.3f} m",
)

print(
    "Minimum change:",
    f"{np.min(change):.3f} m",
)

print(
    "Maximum change:",
    f"{np.max(change):.3f} m",
)


# Plot replacement mask

replacement_map = np.where(
    flat_water_mask,
    1,
    np.nan,
)

plt.figure(figsize=(9, 8))

plt.imshow(
    replacement_map,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
)

plt.title("Cells Replaced with NOAA Bathymetry")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.tight_layout()

mask_figure = (
    FIGURE_DIR
    / "07_bathymetry_replacement_mask.png"
)

plt.savefig(
    mask_figure,
    dpi=200,
)

plt.close()


# Plot elevation change

change_map = np.full(
    usgs_data.shape,
    np.nan,
    dtype="float32",
)

change_map[flat_water_mask] = change

plt.figure(figsize=(9, 8))

image = plt.imshow(
    change_map,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    ],
    origin="upper",
    cmap="terrain",
    vmin=-7,
    vmax=0,
)

plt.title("Elevation Change from Added Bathymetry")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

plt.colorbar(
    image,
    label="Scenario B - Scenario A (m)",
)

plt.tight_layout()

change_figure = (
    FIGURE_DIR
    / "07_bathymetry_elevation_change.png"
)

plt.savefig(
    change_figure,
    dpi=200,
)

plt.close()


print("\nSaved rasters:")
print(SCENARIO_A_FILE)
print(SCENARIO_B_FILE)

print("\nSaved figures:")
print(mask_figure)
print(change_figure)