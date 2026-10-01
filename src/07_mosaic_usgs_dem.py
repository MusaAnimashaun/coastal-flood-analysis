from pathlib import Path

import rasterio
from rasterio.merge import merge


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "usgs_1m"
    / "FL_Peninsular_2018_D18"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "usgs_1m_mosaic.tif"


# --------------------------------------------------
# Find USGS DEM tiles
# --------------------------------------------------

dem_files = sorted(INPUT_DIR.glob("*.tif"))

print(f"USGS DEM tiles found: {len(dem_files)}")

for file_path in dem_files:
    print(" ", file_path.name)


# --------------------------------------------------
# Open the raster tiles
# --------------------------------------------------

datasets = [rasterio.open(file_path) for file_path in dem_files]


# --------------------------------------------------
# Mosaic the tiles
# --------------------------------------------------

mosaic, mosaic_transform = merge(datasets)

print("\nMosaic created.")
print("Array shape:", mosaic.shape)
print("Transform:", mosaic_transform)


# --------------------------------------------------
# Prepare output metadata
# --------------------------------------------------

output_metadata = datasets[0].meta.copy()

output_metadata.update(
    {
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": mosaic_transform,
    }
)


# --------------------------------------------------
# Save the mosaic
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with rasterio.open(
    OUTPUT_FILE,
    "w",
    **output_metadata,
) as dst:
    dst.write(mosaic)


# --------------------------------------------------
# Close input datasets
# --------------------------------------------------

for dataset in datasets:
    dataset.close()


print("\nSaved:")
print(OUTPUT_FILE)