from pathlib import Path

import numpy as np
import rasterio


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEM_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "tampa_topobathy"
    / "noaa_2019"
    / "2019_ngs_tampabay_topobath_dem_J1461017tR0_C0.tif"
)


# --------------------------------------------------
# Open DEM
# --------------------------------------------------

with rasterio.open(DEM_FILE) as src:

    print("File:", DEM_FILE.name)

    print("\n--- Raster properties ---")
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Data type:", src.dtypes[0])
    print("NoData value:", src.nodata)

    print("\n--- Spatial properties ---")
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    print("Transform:", src.transform)

    # Read first raster band as a masked NumPy array
    elevation = src.read(1, masked=True)


# --------------------------------------------------
# Elevation statistics
# --------------------------------------------------

print("\n--- Elevation statistics ---")
print("Valid cells:", elevation.count())
print("Minimum elevation:", float(elevation.min()))
print("Maximum elevation:", float(elevation.max()))
print("Mean elevation:", float(elevation.mean()))
print("Median elevation:", float(np.ma.median(elevation)))