from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_propagation


# --------------------------------------------------
# Project paths
# --------------------------------------------------

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


# --------------------------------------------------
# Water levels
# All elevations are meters NAVD88
# --------------------------------------------------

WATER_LEVELS = [
    0.5,
    1.0,
    1.5,
    2.0,
]


# --------------------------------------------------
# Read DEMs
# --------------------------------------------------

with rasterio.open(SCENARIO_A_FILE) as src:

    scenario_a = src.read(1, masked=True)

    metadata = src.meta.copy()
    transform = src.transform
    crs = src.crs


with rasterio.open(SCENARIO_B_FILE) as src:

    scenario_b = src.read(1, masked=True)


# --------------------------------------------------
# Read coastal seed mask
# --------------------------------------------------

with rasterio.open(SEED_FILE) as src:

    seed_data = src.read(1)

seed_mask = seed_data == 1


# --------------------------------------------------
# Check raster dimensions
# --------------------------------------------------

if scenario_a.shape != scenario_b.shape:
    raise ValueError(
        "Scenario A and Scenario B shapes do not match."
    )

if scenario_a.shape != seed_mask.shape:
    raise ValueError(
        "DEM and seed-mask shapes do not match."
    )


# --------------------------------------------------
# Cell area
# --------------------------------------------------

cell_area_m2 = abs(
    transform.a * transform.e
)


# --------------------------------------------------
# 8-neighbor connectivity structure
# --------------------------------------------------

connectivity = np.ones(
    (3, 3),
    dtype=bool,
)


# --------------------------------------------------
# Function to model inundation
# --------------------------------------------------

def model_inundation(
    dem,
    water_level,
    seed_mask,
):
    """
    Identify cells at or below the specified water
    level that are connected to the coastal seed area.

    Returns
    -------
    inundated : ndarray of bool
        Connected inundation mask.

    depth : ndarray of float32
        Water depth in inundated cells.
        Non-inundated cells contain NaN.
    """

    valid = ~np.ma.getmaskarray(dem)

    elevation = dem.data


    # Cells that could potentially contain water
    eligible = (
        valid
        & (elevation <= water_level)
    )


    # Seeds must also be eligible
    active_seeds = (
        seed_mask
        & eligible
    )


    # Propagate outward from coastal water through
    # neighboring eligible cells
    inundated = binary_propagation(
        active_seeds,
        structure=connectivity,
        mask=eligible,
    )


    # Flood depth
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


# --------------------------------------------------
# Results table
# --------------------------------------------------

results = []


# --------------------------------------------------
# Run both scenarios
# --------------------------------------------------

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


        # ------------------------------------------
        # Inundation statistics
        # ------------------------------------------

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


        # ------------------------------------------
        # Save inundation mask
        #
        # 1 = inundated
        # 0 = not inundated
        # 255 = NoData
        # ------------------------------------------

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


        level_name = str(
            water_level
        ).replace(".", "p")


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
                vertical_datum="NAVD88",
                connectivity="8-neighbor",
            )


        # ------------------------------------------
        # Save flood-depth raster
        # ------------------------------------------

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
                vertical_datum="NAVD88",
                units="meters",
                connectivity="8-neighbor",
            )


        # ------------------------------------------
        # Add statistics to results table
        # ------------------------------------------

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


# --------------------------------------------------
# Save summary CSV
# --------------------------------------------------

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
