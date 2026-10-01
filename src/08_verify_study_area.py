from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape


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

FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Define study-area polygon
# Coordinates are (longitude, latitude)
# --------------------------------------------------

coordinates = [
    (-82.6904, 28.0278),    # A
    (-82.6731, 28.0020),    # B
    (-82.6924, 27.9514),    # C
    (-82.7298, 27.98267),   # D
]

study_area = Polygon(coordinates)

print("Study area valid:", study_area.is_valid)
print("Study area bounds:", study_area.bounds)


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
# Function to plot raster with AOI
# --------------------------------------------------

def plot_raster_with_aoi(raster_file, output_file, title):

    with rasterio.open(raster_file) as src:

        # Transform AOI from geographic coordinates
        # (EPSG:4326) into the raster's CRS.
        transformed_geometry = transform_geom(
            "EPSG:4326",
            src.crs,
            mapping(study_area),
        )

        transformed_aoi = shape(transformed_geometry)

        # Read a lower-resolution preview.
        preview = src.read(
            1,
            out_shape=(
                max(1, src.height // 10),
                max(1, src.width // 10),
            ),
            masked=True,
        )

        left, bottom, right, top = src.bounds

        fig, ax = plt.subplots(figsize=(9, 8))

        image = ax.imshow(
            preview,
            extent=[left, right, bottom, top],
            cmap="terrain",
            origin="upper",
        )

        x, y = transformed_aoi.exterior.xy

        ax.plot(
            x,
            y,
            linewidth=2,
            label="Study area",
        )

        ax.set_title(title)
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.legend()

        fig.colorbar(
            image,
            ax=ax,
            label="Elevation",
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=200)
        plt.close()

        print("\nRaster:", raster_file.name)
        print("CRS:", src.crs)
        print("Raster bounds:", src.bounds)
        print("AOI bounds in raster CRS:", transformed_aoi.bounds)
        print("Saved figure:", output_file)


# --------------------------------------------------
# Plot USGS
# --------------------------------------------------

plot_raster_with_aoi(
    USGS_FILE,
    FIGURE_DIR / "08_usgs_aoi_check.png",
    "USGS 1 m DEM and Study Area",
)


# --------------------------------------------------
# Plot NOAA
# --------------------------------------------------

plot_raster_with_aoi(
    NOAA_FILE,
    FIGURE_DIR / "08_noaa_aoi_check.png",
    "NOAA Topobathy and Study Area",
)