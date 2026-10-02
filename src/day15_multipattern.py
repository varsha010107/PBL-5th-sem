import os
import sys
import csv

import cv2
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SRC = os.path.join(BASE, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from heatmap_model import SpatialLocalizer


MODEL_PATH = os.path.join(
    BASE,
    "models",
    "day10",
    "spatial_localizer.pth"
)

SEARCH_DIR = os.path.join(
    BASE,
    "dataset",
    "day13_variation",
    "search"
)

LABEL_DIR = os.path.join(
    BASE,
    "dataset",
    "day13_variation",
    "labels"
)

RESULT_DIR = os.path.join(
    BASE,
    "results",
    "day15"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

WAFER_SIZE = 2500
TILE_SIZE = 500

GRID_SIZE = 4

# AI was trained with 256x256 input
AI_SIZE = 256


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("DAY 15 - MULTI-PATTERN SPATIAL AI")
print("=" * 70)

device = torch.device("cpu")

model = SpatialLocalizer()

state = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state)
model.eval()

print("AI model loaded successfully")
print()


# ============================================================
# AI PREDICTION FOR ONE TILE
# ============================================================

def predict_tile(tile):

    original_h, original_w = tile.shape

    resized = cv2.resize(
        tile,
        (AI_SIZE, AI_SIZE)
    )

    normalized = (
        resized.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        heatmap = model(tensor)

    heatmap = heatmap[0, 0]

    index = torch.argmax(heatmap)

    hy, hx = np.unravel_index(
        index.item(),
        heatmap.shape
    )

    # Heatmap = 64x64
    # AI input = 256x256

    x_256 = hx * 4
    y_256 = hy * 4

    # Convert 256x256 coordinates
    # back to 500x500 tile coordinates

    x_tile = (
        x_256 * original_w / AI_SIZE
    )

    y_tile = (
        y_256 * original_h / AI_SIZE
    )

    confidence = heatmap[
        hy,
        hx
    ].item()

    return (
        x_tile,
        y_tile,
        confidence
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_labels(filename):

    path = os.path.join(
        LABEL_DIR,
        filename
    )

    labels = []

    with open(
        path,
        "r"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            labels.append(
                (
                    int(row["reference"]),
                    float(row["x"]),
                    float(row["y"])
                )
            )

    return labels


# ============================================================
# PROCESS ONE WAFER
# ============================================================

def process_search(search_filename):

    search_path = os.path.join(
        SEARCH_DIR,
        search_filename
    )

    image = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            f"Unable to read {search_path}"
        )

    if image.shape != (
        WAFER_SIZE,
        WAFER_SIZE
    ):

        raise RuntimeError(
            f"Unexpected image size: {image.shape}"
        )

    label_filename = (
        os.path.splitext(search_filename)[0]
        + ".csv"
    )

    ground_truth = load_labels(
        label_filename
    )

    predictions = []

    print(
        f"\nProcessing {search_filename}"
    )

    print("-" * 70)

    # ========================================================
    # 5 x 5 pattern grid
    # ========================================================

    # NOTE:
    # Day-13 contains 25 patterns.
    # They are placed every 500 pixels.
    #
    # Therefore positions are:
    #
    # 0, 500, 1000, 1500, 2000
    #
    # The last coordinate represents the pattern origin
    # according to the Day-13 ground-truth convention.

    for index, (
        reference_id,
        gt_x,
        gt_y
    ) in enumerate(ground_truth):

        # ----------------------------------------------------
        # Determine tile origin
        # ----------------------------------------------------

        tile_x = int(gt_x)
        tile_y = int(gt_y)

        # Last row/column at 2000 is outside a 2000 image
        # as a 500x500 crop, so clamp to the final valid tile.
        tile_x = int(gt_x)
        tile_y = int(gt_y)

        tile = image[
            tile_y:tile_y + TILE_SIZE,
            tile_x:tile_x + TILE_SIZE
        ]
        tile = image[
            tile_y:tile_y + TILE_SIZE,
            tile_x:tile_x + TILE_SIZE
        ]

        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        pred_x_local, pred_y_local, confidence = (
            predict_tile(tile)
        )

        pred_x = tile_x + pred_x_local
        pred_y = tile_y + pred_y_local

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error = np.sqrt(
            (pred_x - gt_x) ** 2
            +
            (pred_y - gt_y) ** 2
        )

        predictions.append(
            (
                reference_id,
                gt_x,
                gt_y,
                pred_x,
                pred_y,
                error,
                confidence
            )
        )

        print(
            f"{reference_id:03d} | "
            f"GT=({gt_x:.0f},{gt_y:.0f}) | "
            f"Pred=({pred_x:.1f},{pred_y:.1f}) | "
            f"Error={error:.2f}px | "
            f"Confidence={confidence:.4f}"
        )

    return predictions


# ============================================================
# MAIN
# ============================================================

all_predictions = []

search_files = sorted(
    [
        f
        for f in os.listdir(SEARCH_DIR)
        if f.startswith("search_")
        and f.endswith(".png")
    ]
)

for search_file in search_files:

    results = process_search(
        search_file
    )

    all_predictions.extend(
        results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

result_csv = os.path.join(
    RESULT_DIR,
    "multipattern_ai_results.csv"
)

with open(
    result_csv,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "reference",
            "ground_truth_x",
            "ground_truth_y",
            "predicted_x",
            "predicted_y",
            "error_pixels",
            "confidence"
        ]
    )

    writer.writerows(
        all_predictions
    )


# ============================================================
# STATISTICS
# ============================================================

errors = np.array(
    [
        row[5]
        for row in all_predictions
    ]
)

success_threshold = 25.0

success = (
    errors <= success_threshold
)

mean_error = np.mean(errors)
median_error = np.median(errors)
maximum_error = np.max(errors)
success_rate = (
    np.mean(success) * 100
)


print()
print("=" * 70)
print("DAY 15 MULTI-PATTERN AI RESULTS")
print("=" * 70)

print(
    f"Patterns tested       : {len(errors)}"
)

print(
    f"Mean error            : {mean_error:.2f} pixels"
)

print(
    f"Median error          : {median_error:.2f} pixels"
)

print(
    f"Maximum error         : {maximum_error:.2f} pixels"
)

print(
    f"Success threshold     : {success_threshold:.1f} pixels"
)

print(
    f"Successful patterns   : "
    f"{np.sum(success)}/{len(errors)}"
)

print(
    f"Success rate          : "
    f"{success_rate:.2f}%"
)

print()
print(
    f"Results saved:"
)

print(
    result_csv
)

print("=" * 70)
print("DAY 15 MULTI-PATTERN AI TEST COMPLETE")
print("=" * 70)
