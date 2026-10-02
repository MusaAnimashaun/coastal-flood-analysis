from pathlib import Path

import numpy as np
import rasterio
import xarray as xr


# Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
INUNDATION_DIR = PROJECT_DIR / "outputs" / "inundation"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SCENARIO_A_FILE = (
    PROCESSED_DIR
    / "scenario_a_topographic.tif"
)

SCENARIO_B_FILE = (
    PROCESSED_DIR
    / "scenario_b_topobathy.tif"
)

SEED_FILE = (
    PROCESSED_DIR
    / "coastal_seed_mask.tif"
)

NETCDF_FILE = (
    OUTPUT_DIR
    / "coastal_flood_analysis.nc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Water levels in meters NAVD88

WATER_LEVELS = [
    0.5,
    1.0,
    1.5,
    2.0,
]


# Check input files

for input_file in [
    SCENARIO_A_FILE,
    SCENARIO_B_FILE,
    SEED_FILE,
]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )


# Read elevation surfaces and seed mask

with rasterio.open(SCENARIO_A_FILE) as src_a, \
        rasterio.open(SCENARIO_B_FILE) as src_b, \
        rasterio.open(SEED_FILE) as src_seed:

    if (
        src_a.crs != src_b.crs
        or src_a.crs != src_seed.crs
        or src_a.transform != src_b.transform
        or src_a.transform != src_seed.transform
        or src_a.width != src_b.width
        or src_a.width != src_seed.width
        or src_a.height != src_b.height
        or src_a.height != src_seed.height
    ):
        raise ValueError(
            "Input rasters are not on the same grid."
        )

    scenario_a = src_a.read(1, masked=True)
    scenario_b = src_b.read(1, masked=True)

    seed_data = src_seed.read(1)
    seed_mask = seed_data == 1

    transform = src_a.transform
    crs = src_a.crs
    height = src_a.height
    width = src_a.width


# Create projected coordinates at cell centers

x = (
    transform.c
    + (np.arange(width) + 0.5)
    * transform.a
)

y = (
    transform.f
    + (np.arange(height) + 0.5)
    * transform.e
)


# Read inundation masks

inundation_a = []
inundation_b = []

for water_level in WATER_LEVELS:

    level_name = str(
        water_level
    ).replace(".", "p")

    mask_a_file = (
        INUNDATION_DIR
        / f"scenario_a_wl_{level_name}m_mask.tif"
    )

    mask_b_file = (
        INUNDATION_DIR
        / f"scenario_b_wl_{level_name}m_mask.tif"
    )

    if not mask_a_file.exists():
        raise FileNotFoundError(
            f"Inundation mask not found: {mask_a_file}"
        )

    if not mask_b_file.exists():
        raise FileNotFoundError(
            f"Inundation mask not found: {mask_b_file}"
        )

    with rasterio.open(mask_a_file) as src:
        mask_a = src.read(1)

    with rasterio.open(mask_b_file) as src:
        mask_b = src.read(1)

    inundation_a.append(
        (mask_a == 1).astype("uint8")
    )

    inundation_b.append(
        (mask_b == 1).astype("uint8")
    )


# Stack water levels into a new dimension

inundation_a = np.stack(
    inundation_a,
    axis=0,
)

inundation_b = np.stack(
    inundation_b,
    axis=0,
)


# Convert masked elevations to NaN

scenario_a_data = (
    scenario_a.filled(np.nan)
    .astype("float32")
)

scenario_b_data = (
    scenario_b.filled(np.nan)
    .astype("float32")
)


# Build xarray dataset

dataset = xr.Dataset(
    data_vars={
        "scenario_a_elevation": (
            ("y", "x"),
            scenario_a_data,
        ),
        "scenario_b_elevation": (
            ("y", "x"),
            scenario_b_data,
        ),
        "coastal_seed": (
            ("y", "x"),
            seed_mask.astype("uint8"),
        ),
        "inundation_a": (
            ("water_level", "y", "x"),
            inundation_a,
        ),
        "inundation_b": (
            ("water_level", "y", "x"),
            inundation_b,
        ),
    },
    coords={
        "water_level": np.array(
            WATER_LEVELS,
            dtype="float32",
        ),
        "x": x,
        "y": y,
    },
)


# Add dataset metadata

dataset.attrs = {
    "title": (
        "Tampa Bay Coastal Flood Analysis"
    ),
    "description": (
        "Topographic and bathymetry-enhanced "
        "elevation scenarios with static "
        "connectivity-based inundation results."
    ),
    "crs": crs.to_string(),
    "vertical_datum": "NAVD88",
    "elevation_units": "meters",
    "connectivity": "8-neighbor",
}


# Add variable metadata

dataset["scenario_a_elevation"].attrs = {
    "long_name": "Scenario A topographic elevation",
    "units": "m",
    "vertical_datum": "NAVD88",
}

dataset["scenario_b_elevation"].attrs = {
    "long_name": (
        "Scenario B bathymetry-enhanced elevation"
    ),
    "units": "m",
    "vertical_datum": "NAVD88",
}

dataset["coastal_seed"].attrs = {
    "long_name": "Coastal water seed mask",
    "flag_values": [0, 1],
    "flag_meanings": "non_seed seed",
}

dataset["inundation_a"].attrs = {
    "long_name": (
        "Scenario A connected inundation mask"
    ),
    "flag_values": [0, 1],
    "flag_meanings": "not_inundated inundated",
}

dataset["inundation_b"].attrs = {
    "long_name": (
        "Scenario B connected inundation mask"
    ),
    "flag_values": [0, 1],
    "flag_meanings": "not_inundated inundated",
}

dataset["water_level"].attrs = {
    "long_name": "Static water level",
    "units": "m",
    "vertical_datum": "NAVD88",
}

dataset["x"].attrs = {
    "long_name": "Easting",
    "units": "m",
}

dataset["y"].attrs = {
    "long_name": "Northing",
    "units": "m",
}


# Compress large variables

encoding = {
    "scenario_a_elevation": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
    },
    "scenario_b_elevation": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
    },
    "coastal_seed": {
        "zlib": True,
        "complevel": 4,
        "dtype": "uint8",
    },
    "inundation_a": {
        "zlib": True,
        "complevel": 4,
        "dtype": "uint8",
    },
    "inundation_b": {
        "zlib": True,
        "complevel": 4,
        "dtype": "uint8",
    },
}


# Save NetCDF

print("NETCDF EXPORT")
print("Dataset dimensions:")
print(dataset.sizes)

print("\nVariables:")
for variable in dataset.data_vars:
    print(
        variable,
        dataset[variable].dims,
        dataset[variable].dtype,
    )

print("\nWriting:")
print(NETCDF_FILE)

dataset.to_netcdf(
    NETCDF_FILE,
    engine="netcdf4",
    encoding=encoding,
)

dataset.close()


# Verify saved file

with xr.open_dataset(
    NETCDF_FILE,
    engine="netcdf4",
) as check:

    print("\nSAVED NETCDF")
    print(check)

    print(
        "\nFile size:"
        f" {NETCDF_FILE.stat().st_size / 1_000_000:.1f} MB"
    )