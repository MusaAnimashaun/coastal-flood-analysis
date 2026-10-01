from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# --------------------------------------------------
# Project paths
# --------------------------------------------------

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
    / "noaa_topobathy_aoi.tif"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "noaa_topobathy_aligned.tif"
)


# --------------------------------------------------
# Unit conversion
# --------------------------------------------------

FTUS_TO_M = 1200 / 3937


# --------------------------------------------------
# Open reference USGS raster
# --------------------------------------------------

with rasterio.open(USGS_FILE) as usgs:

    reference_crs = usgs.crs
    reference_transform = usgs.transform
    reference_width = usgs.width
    reference_height = usgs.height

    print("Reference grid")
    print("CRS:", reference_crs)
    print("Size:", reference_width, "x", reference_height)
    print("Resolution:", usgs.res)
    print("Transform:")
    print(reference_transform)

    # --------------------------------------------------
    # Open NOAA topobathy
    # --------------------------------------------------

    with rasterio.open(NOAA_FILE) as noaa:

        print("\nOriginal NOAA")
        print("CRS:", noaa.crs)
        print("Size:", noaa.width, "x", noaa.height)
        print("Resolution:", noaa.res)

        # Read NOAA elevations as a masked array.
        noaa_data = noaa.read(1, masked=True)

        # Convert valid elevation values from
        # US survey feet to meters.
        noaa_meters = noaa_data.astype("float32") * FTUS_TO_M

        print("\nNOAA elevation conversion")
        print(
            "Original range (ftUS):",
            float(noaa_data.min()),
            "to",
            float(noaa_data.max()),
        )
        print(
            "Converted range (m):",
            float(noaa_meters.min()),
            "to",
            float(noaa_meters.max()),
        )

        # --------------------------------------------------
        # Create destination array
        # --------------------------------------------------

        destination_nodata = -999999.0

        aligned = np.full(
            (reference_height, reference_width),
            destination_nodata,
            dtype="float32",
        )

        # --------------------------------------------------
        # Reproject NOAA onto exact USGS grid
        # --------------------------------------------------

        reproject(
            source=noaa_meters.filled(destination_nodata),
            destination=aligned,
            src_transform=noaa.transform,
            src_crs=noaa.crs,
            src_nodata=destination_nodata,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=destination_nodata,
            resampling=Resampling.bilinear,
        )

        # --------------------------------------------------
        # Prepare output metadata
        # --------------------------------------------------

        metadata = noaa.meta.copy()

        metadata.update(
            {
                "crs": reference_crs,
                "transform": reference_transform,
                "width": reference_width,
                "height": reference_height,
                "dtype": "float32",
                "nodata": destination_nodata,
            }
        )

        # --------------------------------------------------
        # Save aligned NOAA raster
        # --------------------------------------------------

        with rasterio.open(
            OUTPUT_FILE,
            "w",
            **metadata,
        ) as dst:

            dst.write(aligned, 1)

            dst.update_tags(
                vertical_datum="NAVD88",
                vertical_units="meters",
                source_vertical_units="US survey feet",
                alignment_reference="usgs_1m_aoi.tif",
            )


# --------------------------------------------------
# Verify output
# --------------------------------------------------

with rasterio.open(OUTPUT_FILE) as aligned_src:

    aligned_data = aligned_src.read(1, masked=True)

    print("\nAligned NOAA")
    print("CRS:", aligned_src.crs)
    print("Size:", aligned_src.width, "x", aligned_src.height)
    print("Resolution:", aligned_src.res)
    print("NoData:", aligned_src.nodata)
    print("Valid cells:", f"{aligned_data.count():,}")
    print(
        "Elevation range:",
        float(aligned_data.min()),
        "to",
        float(aligned_data.max()),
        "m",
    )
    print("Transform:")
    print(aligned_src.transform)


print("\nSaved:")
print(OUTPUT_FILE)