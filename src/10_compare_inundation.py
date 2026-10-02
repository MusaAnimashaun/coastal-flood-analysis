from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


# Project paths

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


# Water levels in meters NAVD88

WATER_LEVELS = [
    0.5,
    1.0,
    1.5,
    2.0,
]


# Check input files

for input_file in [
    SEED_FILE,
    SCENARIO_A_FILE,
    SCENARIO_B_FILE,
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

    transform = src_a.transform


# Calculate cell area

cell_area_m2 = abs(
    transform.a * transform.e
)


# Compare inundation results

results = []

for water_level in WATER_LEVELS:

    level_name = str(
        water_level
    ).replace(".", "p")

    mask_a_file = (
        OUTPUT_DIR
        / f"scenario_a_wl_{level_name}m_mask.tif"
    )

    mask_b_file = (
        OUTPUT_DIR
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

    # Read inundation masks

    with rasterio.open(mask_a_file) as src:
        inundated_a = src.read(1) == 1

    with rasterio.open(mask_b_file) as src:
        inundated_b = src.read(1) == 1

    # Remove the existing coastal-water seed area

    beyond_seed_a = (
        inundated_a
        & ~seed_mask
    )

    beyond_seed_b = (
        inundated_b
        & ~seed_mask
    )

    # Calculate inundated area

    cells_a = int(
        beyond_seed_a.sum()
    )

    cells_b = int(
        beyond_seed_b.sum()
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

    # Compare spatial extent

    only_a = (
        beyond_seed_a
        & ~beyond_seed_b
    )

    only_b = (
        beyond_seed_b
        & ~beyond_seed_a
    )

    common = (
        beyond_seed_a
        & beyond_seed_b
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

    # Calculate depth beyond the seed area

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

    depth_a[beyond_seed_a] = (
        water_level
        - scenario_a.data[beyond_seed_a]
    )

    depth_b[beyond_seed_b] = (
        water_level
        - scenario_b.data[beyond_seed_b]
    )

    if cells_a > 0:
        mean_depth_a = float(
            np.nanmean(depth_a)
        )
        median_depth_a = float(
            np.nanmedian(depth_a)
        )
    else:
        mean_depth_a = np.nan
        median_depth_a = np.nan

    if cells_b > 0:
        mean_depth_b = float(
            np.nanmean(depth_b)
        )
        median_depth_b = float(
            np.nanmedian(depth_b)
        )
    else:
        mean_depth_b = np.nan
        median_depth_b = np.nan

    # Compare depth where both scenarios are inundated

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

    print(
        f"\nWATER LEVEL: "
        f"{water_level:.1f} m NAVD88"
    )

    print(
        f"Scenario A area beyond seed: "
        f"{area_a_km2:.3f} km²"
    )

    print(
        f"Scenario B area beyond seed: "
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
        f"Mean depth A: "
        f"{mean_depth_a:.3f} m"
    )

    print(
        f"Mean depth B: "
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

    results.append(
        {
            "water_level_m": water_level,
            "area_a_km2": area_a_km2,
            "area_b_km2": area_b_km2,
            "common_cells": common_cells,
            "only_a_cells": only_a_cells,
            "only_b_cells": only_b_cells,
            "mean_depth_a_m": mean_depth_a,
            "median_depth_a_m": median_depth_a,
            "mean_depth_b_m": mean_depth_b,
            "median_depth_b_m": median_depth_b,
            "mean_depth_difference_m":
                mean_depth_difference,
            "max_abs_depth_difference_m":
                max_abs_depth_difference,
        }
    )


# Save comparison table

results_df = pd.DataFrame(
    results
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "inundation_comparison.csv"
)

results_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# Plot inundation area beyond the seed

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["water_level_m"],
    results_df["area_a_km2"],
    marker="o",
    label="Scenario A",
)

plt.plot(
    results_df["water_level_m"],
    results_df["area_b_km2"],
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
    / "10_inundation_area_comparison.png"
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