from pathlib import Path

import numpy as np
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

NOAA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "tampa_topobathy"
    / "noaa_2019_aoi2"
)

USGS_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "usgs_1m"
    / "FL_Peninsular_2018_D18"
)


# --------------------------------------------------
# Find raster files
# --------------------------------------------------

noaa_files = sorted(NOAA_DIR.glob("*.tif"))
usgs_files = sorted(USGS_DIR.glob("*.tif"))

print(f"NOAA TIFF files found: {len(noaa_files)}")
print(f"USGS TIFF files found: {len(usgs_files)}")


# --------------------------------------------------
# Function for inspecting one raster
# --------------------------------------------------

def inspect_raster(file_path):

    with rasterio.open(file_path) as src:

        elevation = src.read(1, masked=True)

        print("\n" + "=" * 70)
        print("File:", file_path.name)

        print("\nRaster properties")
        print("CRS:", src.crs)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("Data type:", src.dtypes[0])
        print("NoData:", src.nodata)

        print("\nSpatial properties")
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)

        print("\nElevation statistics")
        print("Valid cells:", f"{elevation.count():,}")
        print("Minimum:", float(elevation.min()))
        print("Maximum:", float(elevation.max()))
        print("Mean:", float(elevation.mean()))
        print("Median:", float(np.ma.median(elevation)))


# --------------------------------------------------
# Inspect NOAA raster
# --------------------------------------------------

print("\n\nNOAA TOPOBATHY")

for file_path in noaa_files:
    inspect_raster(file_path)


# --------------------------------------------------
# Inspect USGS rasters
# --------------------------------------------------

print("\n\nUSGS 1-METER DEM")

for file_path in usgs_files:
    inspect_raster(file_path)