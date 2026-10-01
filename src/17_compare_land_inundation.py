from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "outputs" / "inundation"
FIGURE_DIR = PROJECT_DIR / "figures"

SEED_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "coastal_seed_mask.tif"
)

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

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Water levels
# --------------------------------------------------

WATER_LEVELS = [
    0.5,
    1.0,
    1.5,
    2.0,
]


# --------------------------------------------------
# Read coastal-water seed mask
# --------------------------------------------------

with rasterio.open(SEED_FILE) as src:

    seed_data = src.read(1)
    seed_mask = seed_data == 1

    transform = src.transform
    bounds = src.bounds


cell_area_m2 = abs(
    transform.a * transform.e
)


# --------------------------------------------------
# Read DEMs
# --------------------------------------------------

with rasterio.open(SCENARIO_A_FILE) as src:

    scenario_a = src.read(1, masked=True)


with rasterio.open(SCENARIO_B_FILE) as src:

    scenario_b = src.read(1, masked=True)


# --------------------------------------------------
# Results
# --------------------------------------------------

results = []


for water_level in WATER_LEVELS:

    level_name = str(
        water_level
    ).replace(".", "p")


    # ----------------------------------------------
    # Input files from Step 16
    # ----------------------------------------------

    mask_a_file = (
        OUTPUT_DIR
        / f"scenario_a_wl_{level_name}m_mask.tif"
    )

    mask_b_file = (
        OUTPUT_DIR
        / f"scenario_b_wl_{level_name}m_mask.tif"
    )


    # ----------------------------------------------
    # Read inundation masks
    # ----------------------------------------------

    with rasterio.open(mask_a_file) as src:

        inundated_a = src.read(1) == 1


    with rasterio.open(mask_b_file) as src:

        inundated_b = src.read(1) == 1


    # ----------------------------------------------
    # Remove existing coastal-water seed area
    #
    # Remaining cells represent expansion beyond
    # the identified pre-existing water area.
    # ----------------------------------------------

    land_flood_a = (
        inundated_a
        & ~seed_mask
    )

    land_flood_b = (
        inundated_b
        & ~seed_mask
    )


    # ----------------------------------------------
    # Area statistics
    # ----------------------------------------------

    cells_a = int(
        land_flood_a.sum()
    )

    cells_b = int(
        land_flood_b.sum()
    )

    area_a_km2 = (
        cells_a
        * cell_area_m2
        / 1_000_000
    )

    area_b_km2 = (
        cells_b
        * cell_area_m2
        / 1_000_000
    )


    # ----------------------------------------------
    # Compare spatial extent
    # ----------------------------------------------

    only_a = (
        land_flood_a
        & ~land_flood_b
    )

    only_b = (
        land_flood_b
        & ~land_flood_a
    )

    common = (
        land_flood_a
        & land_flood_b
    )


    only_a_cells = int(
        only_a.sum()
    )

    only_b_cells = int(
        only_b.sum()
    )

    common_cells = int(
        common.sum()
    )


    # ----------------------------------------------
    # Land flood depths
    # ----------------------------------------------

    depth_a = np.full(
        scenario_a.shape,
        np.nan,
        dtype="float32",
    )

    depth_b = np.full(
        scenario_b.shape,
        np.nan,
        dtype="float32",
    )


    depth_a[land_flood_a] = (
        water_level
        - scenario_a.data[land_flood_a]
    )

    depth_b[land_flood_b] = (
        water_level
        - scenario_b.data[land_flood_b]
    )


    mean_depth_a = float(
        np.nanmean(depth_a)
    )

    mean_depth_b = float(
        np.nanmean(depth_b)
    )

    median_depth_a = float(
        np.nanmedian(depth_a)
    )

    median_depth_b = float(
        np.nanmedian(depth_b)
    )


    # ----------------------------------------------
    # Depth difference where both scenarios flood
    # ----------------------------------------------

    if common_cells > 0:

        depth_difference = (
            depth_b[common]
            - depth_a[common]
        )

        mean_depth_difference = float(
            np.nanmean(depth_difference)
        )

        max_abs_depth_difference = float(
            np.nanmax(
                np.abs(depth_difference)
            )
        )

    else:

        mean_depth_difference = np.nan
        max_abs_depth_difference = np.nan


    # ----------------------------------------------
    # Print
    # ----------------------------------------------

    print(
        f"\nWATER LEVEL: "
        f"{water_level:.1f} m NAVD88"
    )

    print(
        f"Scenario A land inundation: "
        f"{area_a_km2:.3f} km²"
    )

    print(
        f"Scenario B land inundation: "
        f"{area_b_km2:.3f} km²"
    )

    print(
        f"Common cells: "
        f"{common_cells:,}"
    )

    print(
        f"Only A: "
        f"{only_a_cells:,}"
    )

    print(
        f"Only B: "
        f"{only_b_cells:,}"
    )

    print(
        f"Mean land depth A: "
        f"{mean_depth_a:.3f} m"
    )

    print(
        f"Mean land depth B: "
        f"{mean_depth_b:.3f} m"
    )

    print(
        f"Mean B - A depth difference: "
        f"{mean_depth_difference:.6f} m"
    )

    print(
        f"Maximum absolute depth difference: "
        f"{max_abs_depth_difference:.6f} m"
    )


    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append(
        {
            "water_level_m": water_level,
            "land_area_a_km2": area_a_km2,
            "land_area_b_km2": area_b_km2,
            "common_cells": common_cells,
            "only_a_cells": only_a_cells,
            "only_b_cells": only_b_cells,
            "mean_depth_a_m": mean_depth_a,
            "mean_depth_b_m": mean_depth_b,
            "mean_depth_difference_m":
                mean_depth_difference,
            "max_abs_depth_difference_m":
                max_abs_depth_difference,
        }
    )


# --------------------------------------------------
# Save comparison table
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "land_inundation_comparison.csv"
)

results_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# --------------------------------------------------
# Plot land inundation area
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["water_level_m"],
    results_df["land_area_a_km2"],
    marker="o",
    label="Scenario A",
)

plt.plot(
    results_df["water_level_m"],
    results_df["land_area_b_km2"],
    marker="s",
    label="Scenario B",
)

plt.xlabel(
    "Water Level (m NAVD88)"
)

plt.ylabel(
    "Inundated Area Beyond Seed Mask (km²)"
)

plt.title(
    "Connected Inundation Beyond Existing Water Area"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


AREA_FIGURE = (
    FIGURE_DIR
    / "17_land_inundation_area_comparison.png"
)

plt.savefig(
    AREA_FIGURE,
    dpi=200,
)

plt.close()


print("\nSUMMARY")
print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\nSaved figure:")
print(AREA_FIGURE)