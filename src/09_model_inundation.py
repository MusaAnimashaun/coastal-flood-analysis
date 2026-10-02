from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_propagation


# Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

SCENARIO_A_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scenario_a_topographic.tif"
)

SCENARIO_B_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "scenario_b_topobathy.tif"
)

SEED_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "coastal_seed_mask.tif"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "inundation"
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


# Read scenarios and seed mask

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
            "Scenarios and seed mask are not on the same grid."
        )

    scenario_a = src_a.read(1, masked=True)
    scenario_b = src_b.read(1, masked=True)

    seed_data = src_seed.read(1)
    seed_mask = seed_data == 1

    metadata = src_a.meta.copy()
    transform = src_a.transform


# Calculate cell area

cell_area_m2 = abs(
    transform.a * transform.e
)


# Use 8-neighbor connectivity

connectivity = np.ones(
    (3, 3),
    dtype=bool,
)


# Model connected inundation

def model_inundation(
    dem,
    water_level,
    seed_mask,
):
    """Find inundated cells connected to the coastal seed."""

    valid = ~np.ma.getmaskarray(dem)
    elevation = dem.data

    # Cells at or below the water level
    eligible = (
        valid
        & (elevation <= water_level)
    )

    # Coastal seeds that are also eligible
    active_seeds = (
        seed_mask
        & eligible
    )

    # Connect neighboring eligible cells to the seed
    inundated = binary_propagation(
        active_seeds,
        structure=connectivity,
        mask=eligible,
    )

    # Calculate water depth
    depth = np.full(
        dem.shape,
        np.nan,
        dtype="float32",
    )

    depth[inundated] = (
        water_level
        - elevation[inundated]
    )

    return inundated, depth


# Run both scenarios

results = []

scenarios = {
    "A": scenario_a,
    "B": scenario_b,
}

for water_level in WATER_LEVELS:

    print(
        f"\nWATER LEVEL: "
        f"{water_level:.1f} m NAVD88"
    )

    for scenario_name, dem in scenarios.items():

        inundated, depth = model_inundation(
            dem=dem,
            water_level=water_level,
            seed_mask=seed_mask,
        )

        # Calculate inundation statistics
        inundated_cells = int(
            inundated.sum()
        )

        area_m2 = (
            inundated_cells
            * cell_area_m2
        )

        area_km2 = (
            area_m2
            / 1_000_000
        )

        if inundated_cells > 0:

            mean_depth = float(
                np.nanmean(depth)
            )

            median_depth = float(
                np.nanmedian(depth)
            )

            max_depth = float(
                np.nanmax(depth)
            )

        else:

            mean_depth = np.nan
            median_depth = np.nan
            max_depth = np.nan

        print(
            f"Scenario {scenario_name}: "
            f"{inundated_cells:,} cells, "
            f"{area_km2:.3f} km², "
            f"mean depth {mean_depth:.3f} m"
        )

        level_name = str(
            water_level
        ).replace(".", "p")

        # Save inundation mask
        # 1 = inundated, 0 = not inundated, 255 = NoData

        valid = ~np.ma.getmaskarray(dem)

        mask_output = np.zeros(
            dem.shape,
            dtype="uint8",
        )

        mask_output[inundated] = 1
        mask_output[~valid] = 255

        mask_metadata = metadata.copy()

        mask_metadata.update(
            dtype="uint8",
            nodata=255,
            count=1,
        )

        mask_file = (
            OUTPUT_DIR
            / (
                f"scenario_{scenario_name.lower()}"
                f"_wl_{level_name}m_mask.tif"
            )
        )

        with rasterio.open(
            mask_file,
            "w",
            **mask_metadata,
        ) as dst:

            dst.write(
                mask_output,
                1,
            )

            dst.update_tags(
                scenario=scenario_name,
                water_level_m=water_level,
                water_level_datum="NAVD88",
                connectivity="8-neighbor",
            )

        # Save inundation depth

        depth_nodata = -999999.0

        depth_output = np.full(
            dem.shape,
            depth_nodata,
            dtype="float32",
        )

        depth_output[inundated] = (
            depth[inundated]
        )

        depth_metadata = metadata.copy()

        depth_metadata.update(
            dtype="float32",
            nodata=depth_nodata,
            count=1,
        )

        depth_file = (
            OUTPUT_DIR
            / (
                f"scenario_{scenario_name.lower()}"
                f"_wl_{level_name}m_depth.tif"
            )
        )

        with rasterio.open(
            depth_file,
            "w",
            **depth_metadata,
        ) as dst:

            dst.write(
                depth_output,
                1,
            )

            dst.update_tags(
                scenario=scenario_name,
                water_level_m=water_level,
                water_level_datum="NAVD88",
                units="meters",
                connectivity="8-neighbor",
            )

        # Add statistics to results table

        results.append(
            {
                "water_level_m": water_level,
                "scenario": scenario_name,
                "inundated_cells": inundated_cells,
                "inundated_area_km2": area_km2,
                "mean_depth_m": mean_depth,
                "median_depth_m": median_depth,
                "max_depth_m": max_depth,
            }
        )


# Save summary table

results_df = pd.DataFrame(results)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "inundation_summary.csv"
)

results_df.to_csv(
    SUMMARY_FILE,
    index=False,
)

print("\nSUMMARY")
print(results_df.to_string(index=False))

print("\nSaved summary:")
print(SUMMARY_FILE)