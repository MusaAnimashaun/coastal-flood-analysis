from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SCENARIO_A_FILE = (
    PROCESSED_DIR
    / "scenario_a_topographic.tif"
)

SEED_FILE = (
    PROCESSED_DIR
    / "coastal_seed_mask.tif"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "terrain_classification_results.csv"
)

IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "terrain_feature_importance.csv"
)

CONFUSION_FIGURE = (
    FIGURE_DIR
    / "13_confusion_matrix.png"
)

IMPORTANCE_FIGURE = (
    FIGURE_DIR
    / "13_feature_importance.png"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model settings

SAMPLE_PER_CLASS = 100_000
RANDOM_STATE = 42


# Check input files

for input_file in [
    SCENARIO_A_FILE,
    SEED_FILE,
]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )


# Read elevation and target mask

with rasterio.open(SCENARIO_A_FILE) as src_dem, \
        rasterio.open(SEED_FILE) as src_seed:

    if (
        src_dem.crs != src_seed.crs
        or src_dem.transform != src_seed.transform
        or src_dem.width != src_seed.width
        or src_dem.height != src_seed.height
    ):
        raise ValueError(
            "Input rasters are not on the same grid."
        )

    elevation_masked = src_dem.read(
        1,
        masked=True,
    )

    elevation = elevation_masked.filled(
        np.nan
    ).astype("float32")

    seed = src_seed.read(1) == 1

    x_resolution = abs(
        src_dem.transform.a
    )

    y_resolution = abs(
        src_dem.transform.e
    )


# Calculate slope

elevation_for_filter = np.nan_to_num(
    elevation,
    nan=0.0,
)

dz_dy, dz_dx = np.gradient(
    elevation_for_filter,
    y_resolution,
    x_resolution,
)

slope = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx ** 2
            + dz_dy ** 2
        )
    )
).astype("float32")


# Calculate local elevation variability

window_size = 5

local_mean = uniform_filter(
    elevation_for_filter,
    size=window_size,
)

local_mean_squared = uniform_filter(
    elevation_for_filter ** 2,
    size=window_size,
)

local_variance = (
    local_mean_squared
    - local_mean ** 2
)

local_variance = np.maximum(
    local_variance,
    0,
)

local_std = np.sqrt(
    local_variance
).astype("float32")


# Remove cells affected by DEM NoData

valid = (
    ~np.isnan(elevation)
    & np.isfinite(slope)
    & np.isfinite(local_std)
)

positive_indices = np.flatnonzero(
    valid & seed
)

negative_indices = np.flatnonzero(
    valid & ~seed
)

print("TERRAIN CLASSIFICATION")
print(
    f"Replacement cells available: "
    f"{len(positive_indices):,}"
)
print(
    f"Unchanged cells available: "
    f"{len(negative_indices):,}"
)


# Draw a balanced random sample

rng = np.random.default_rng(
    RANDOM_STATE
)

sample_size = min(
    SAMPLE_PER_CLASS,
    len(positive_indices),
    len(negative_indices),
)

positive_sample = rng.choice(
    positive_indices,
    size=sample_size,
    replace=False,
)

negative_sample = rng.choice(
    negative_indices,
    size=sample_size,
    replace=False,
)

sample_indices = np.concatenate(
    [
        positive_sample,
        negative_sample,
    ]
)

target = np.concatenate(
    [
        np.ones(
            sample_size,
            dtype="uint8",
        ),
        np.zeros(
            sample_size,
            dtype="uint8",
        ),
    ]
)


# Build feature table

elevation_flat = elevation.ravel()
slope_flat = slope.ravel()
local_std_flat = local_std.ravel()

features = pd.DataFrame(
    {
        "elevation": (
            elevation_flat[
                sample_indices
            ]
        ),
        "slope": (
            slope_flat[
                sample_indices
            ]
        ),
        "local_std": (
            local_std_flat[
                sample_indices
            ]
        ),
    }
)

print(
    f"Sample per class: "
    f"{sample_size:,}"
)
print(
    f"Total sample: "
    f"{len(features):,}"
)


# Split training and testing data

X_train, X_test, y_train, y_test = (
    train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )
)

print(
    f"Training samples: "
    f"{len(X_train):,}"
)
print(
    f"Testing samples: "
    f"{len(X_test):,}"
)


# Train Random Forest

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# Test model

prediction = model.predict(
    X_test
)

report = classification_report(
    y_test,
    prediction,
    target_names=[
        "unchanged",
        "replacement",
    ],
    output_dict=True,
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        prediction,
        target_names=[
            "unchanged",
            "replacement",
        ],
        digits=3,
    )
)


# Save model performance

results = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "unchanged_precision",
            "unchanged_recall",
            "unchanged_f1",
            "replacement_precision",
            "replacement_recall",
            "replacement_f1",
        ],
        "value": [
            report["accuracy"],
            report["unchanged"]["precision"],
            report["unchanged"]["recall"],
            report["unchanged"]["f1-score"],
            report["replacement"]["precision"],
            report["replacement"]["recall"],
            report["replacement"]["f1-score"],
        ],
    }
)

results.to_csv(
    RESULT_FILE,
    index=False,
)


# Calculate feature importance

importance = pd.DataFrame(
    {
        "feature": features.columns,
        "importance": (
            model.feature_importances_
        ),
    }
).sort_values(
    "importance",
    ascending=False,
)

importance.to_csv(
    IMPORTANCE_FILE,
    index=False,
)

print("FEATURE IMPORTANCE")
print(
    importance.to_string(
        index=False
    )
)


# Plot confusion matrix

matrix = confusion_matrix(
    y_test,
    prediction,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=[
        "Unchanged",
        "Replacement",
    ],
)

display.plot(
    values_format=",",
)

plt.title(
    "Terrain Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_FIGURE,
    dpi=300,
)

plt.close()


# Plot feature importance

importance_plot = importance.sort_values(
    "importance"
)

plt.figure(
    figsize=(7, 4)
)

plt.barh(
    importance_plot["feature"],
    importance_plot["importance"],
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.title(
    "Terrain Classification Feature Importance"
)

plt.tight_layout()

plt.savefig(
    IMPORTANCE_FIGURE,
    dpi=300,
)

plt.close()


print("\nSaved:")
print(RESULT_FILE)
print(IMPORTANCE_FILE)
print(CONFUSION_FIGURE)
print(IMPORTANCE_FIGURE)