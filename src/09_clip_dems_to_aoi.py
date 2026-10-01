from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

USGS_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "usgs_1m_mosaic.tif"
)

NOAA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "tampa_topobathy"
    / "noaa_2019_aoi2"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

USGS_OUTPUT = OUTPUT_DIR / "usgs_1m_aoi.tif"
NOAA_OUTPUT = OUTPUT_DIR / "noaa_topobathy_aoi.tif"


# --------------------------------------------------
# Define study area
# Coordinates are (longitude, latitude)
# --------------------------------------------------

coordinates = [
    (-82.6904, 28.0278),    # A
    (-82.6731, 28.0020),    # B
    (-82.6924, 27.9514),    # C
    (-82.7298, 27.98267),   # D
]

study_area = Polygon(coordinates)


# --------------------------------------------------
# Find NOAA raster
# --------------------------------------------------

noaa_files = sorted(NOAA_DIR.glob("*.tif"))

if len(noaa_files) != 1:
    raise ValueError(
        f"Expected 1 NOAA TIFF, found {len(noaa_files)}"
    )

NOAA_FILE = noaa_files[0]


# --------------------------------------------------
# Function to clip a raster
# --------------------------------------------------

def clip_raster(input_file, output_file):

    with rasterio.open(input_file) as src:

        # Transform the AOI from longitude/latitude
        # into the coordinate system of this raster.
        aoi_in_raster_crs = transform_geom(
            "EPSG:4326",
            src.crs,
            mapping(study_area),
        )

        # Clip the raster to the polygon.
        clipped_data, clipped_transform = mask(
            src,
            [aoi_in_raster_crs],
            crop=True,
            nodata=src.nodata,
        )

        # Copy original raster metadata.
        metadata = src.meta.copy()

        # Update properties that changed after clipping.
        metadata.update(
            {
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform,
            }
        )

        # Save clipped raster.
        with rasterio.open(
            output_file,
            "w",
            **metadata,
        ) as dst:
            dst.write(clipped_data)

        print("\nInput:", input_file.name)
        print("CRS:", src.crs)
        print("Original size:", src.width, "x", src.height)
        print(
            "Clipped size:",
            clipped_data.shape[2],
            "x",
            clipped_data.shape[1],
        )
        print("Resolution:", src.res)
        print("Saved:", output_file)


# --------------------------------------------------
# Clip USGS
# --------------------------------------------------

clip_raster(
    USGS_FILE,
    USGS_OUTPUT,
)


# --------------------------------------------------
# Clip NOAA
# --------------------------------------------------

clip_raster(
    NOAA_FILE,
    NOAA_OUTPUT,
)