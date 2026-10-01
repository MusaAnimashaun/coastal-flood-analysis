from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

SHORELINE_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "noaa_shoreline"
    / "N25W085"
    / "N25W085.shp"
)


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

aoi = gpd.GeoDataFrame(
    {"name": ["study_area"]},
    geometry=[study_area],
    crs="EPSG:4326",
)


# --------------------------------------------------
# Read NOAA shoreline
# --------------------------------------------------

shoreline = gpd.read_file(SHORELINE_FILE)


print("NOAA SHORELINE")
print("Features:", len(shoreline))
print("CRS:", shoreline.crs)

print("\nGeometry types:")
print(shoreline.geom_type.value_counts())

print("\nBounds:")
print(shoreline.total_bounds)

print("\nAttribute columns:")
print(list(shoreline.columns))

print("\nFirst five records:")
print(shoreline.head())


# --------------------------------------------------
# Transform AOI to shoreline CRS if necessary
# --------------------------------------------------

aoi_shoreline_crs = aoi.to_crs(shoreline.crs)


# --------------------------------------------------
# Find shoreline features intersecting AOI
# --------------------------------------------------

intersects_aoi = shoreline.intersects(
    aoi_shoreline_crs.geometry.iloc[0]
)

shoreline_aoi = shoreline[intersects_aoi].copy()


print("\nAOI INTERSECTION")
print(
    "Shoreline features intersecting AOI:",
    len(shoreline_aoi),
)

if len(shoreline_aoi) > 0:

    print("\nIntersecting geometry types:")
    print(
        shoreline_aoi.geom_type.value_counts()
    )

    print("\nIntersecting records:")
    print(shoreline_aoi)

# --------------------------------------------------
# Inspect attributes of AOI shoreline
# --------------------------------------------------

fields_to_check = [
    "SOURCE_ID",
    "SRC_DATE",
    "ATTRIBUTE",
    "DATA_SOURC",
    "EXT_METH",
    "INFORM",
    "HOR_ACC",
]

print("\nAOI ATTRIBUTE SUMMARY")

for field in fields_to_check:

    print("\n" + field)

    print(
        shoreline_aoi[field]
        .value_counts(dropna=False)
        .head(20)
    )