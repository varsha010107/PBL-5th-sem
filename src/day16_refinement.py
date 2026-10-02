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
    "day16"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TILE_SIZE = 500
AI_SIZE = 256

HEATMAP_SIZE = 64

SUCCESS_THRESHOLD = 25.0


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("DAY 16 - REFINED SPATIAL AI LOCALIZATION")
print("=" * 70)

model = SpatialLocalizer()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("AI model loaded successfully")
print()


# ============================================================
# BASIC AI PREDICTION
# ============================================================

def get_heatmap(tile):

    resized = cv2.resize(
        tile,
        (AI_SIZE, AI_SIZE)
    )

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        output = model(
            tensor
        )

    heatmap = output[0, 0]

    return heatmap.cpu().numpy()


# ============================================================
# LOCAL PEAK REFINEMENT
# ============================================================

def refine_peak(heatmap):

    h, w = heatmap.shape

    # --------------------------------------------------------
    # Find strongest heatmap cell
    # --------------------------------------------------------

    max_index = np.argmax(
        heatmap
    )

    hy, hx = np.unravel_index(
        max_index,
        heatmap.shape
    )

    # --------------------------------------------------------
    # 3x3 neighbourhood
    # --------------------------------------------------------

    radius = 1

    x0 = max(
        0,
        hx - radius
    )

    x1 = min(
        w,
        hx + radius + 1
    )

    y0 = max(
        0,
        hy - radius
    )

    y1 = min(
        h,
        hy + radius + 1
    )

    neighbourhood = heatmap[
        y0:y1,
        x0:x1
    ]

    # --------------------------------------------------------
    # Remove negative influence
    # --------------------------------------------------------

    weights = (
        neighbourhood
        - neighbourhood.min()
    )

    weights = (
        weights + 1e-6
    )

    # --------------------------------------------------------
    # Weighted centroid
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        y0:y1,
        x0:x1
    ]

    total = weights.sum()

    refined_x = (
        (xx * weights).sum()
        / total
    )

    refined_y = (
        (yy * weights).sum()
        / total
    )

    confidence = float(
        heatmap[hy, hx]
    )

    return (
        hx,
        hy,
        refined_x,
        refined_y,
        confidence
    )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(
    filename
):

    path = os.path.join(
        LABEL_DIR,
        filename
    )

    labels = []

    with open(
        path,
        "r"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            labels.append(
                (
                    int(
                        row["reference"]
                    ),
                    float(
                        row["x"]
                    ),
                    float(
                        row["y"]
                    )
                )
            )

    return labels


# ============================================================
# PROCESS SEARCH IMAGE
# ============================================================

def process_search(
    search_filename
):

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

    print(
        f"Processing {search_filename}"
    )

    label_filename = (
        os.path.splitext(
            search_filename
        )[0]
        + ".csv"
    )

    labels = load_labels(
        label_filename
    )

    results = []

    for (
        reference_id,
        gt_x,
        gt_y
    ) in labels:

        # ----------------------------------------------------
        # 500x500 tile
        # ----------------------------------------------------

        tile_x = int(
            gt_x
        )

        tile_y = int(
            gt_y
        )

        tile = image[
            tile_y:tile_y + TILE_SIZE,
            tile_x:tile_x + TILE_SIZE
        ]

        if tile.shape != (
            TILE_SIZE,
            TILE_SIZE
        ):

            raise RuntimeError(
                f"Invalid tile shape "
                f"{tile.shape}"
            )

        # ----------------------------------------------------
        # Generate heatmap
        # ----------------------------------------------------

        heatmap = get_heatmap(
            tile
        )

        (
            peak_x,
            peak_y,
            refined_x,
            refined_y,
            confidence
        ) = refine_peak(
            heatmap
        )

        # ----------------------------------------------------
        # Convert heatmap coordinates
        # to 500x500 tile coordinates
        # ----------------------------------------------------

        predicted_local_x = (
            refined_x
            * TILE_SIZE
            / HEATMAP_SIZE
        )

        predicted_local_y = (
            refined_y
            * TILE_SIZE
            / HEATMAP_SIZE
        )

        # ----------------------------------------------------
        # Convert to wafer coordinates
        # ----------------------------------------------------

        predicted_x = (
            tile_x
            + predicted_local_x
        )

        predicted_y = (
            tile_y
            + predicted_local_y
        )

        # ----------------------------------------------------
        # Calculate error
        # ----------------------------------------------------

        error = np.sqrt(
            (
                predicted_x
                - gt_x
            ) ** 2
            +
            (
                predicted_y
                - gt_y
            ) ** 2
        )

        results.append(
            (
                reference_id,
                gt_x,
                gt_y,
                predicted_x,
                predicted_y,
                error,
                confidence,
                peak_x,
                peak_y,
                refined_x,
                refined_y
            )
        )

        print(
            f"{reference_id:03d} | "
            f"GT=({gt_x:.0f},{gt_y:.0f}) | "
            f"Pred=("
            f"{predicted_x:.2f},"
            f"{predicted_y:.2f}"
            f") | "
            f"Error={error:.2f}px | "
            f"Confidence={confidence:.4f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

all_results = []

search_files = sorted(
    [
        f
        for f in os.listdir(
            SEARCH_DIR
        )
        if f.startswith(
            "search_"
        )
        and f.endswith(
            ".png"
        )
    ]
)

for search_file in search_files:

    results = process_search(
        search_file
    )

    all_results.extend(
        results
    )


# ============================================================
# SAVE CSV
# ============================================================

output_csv = os.path.join(
    RESULT_DIR,
    "refined_ai_results.csv"
)

with open(
    output_csv,
    "w",
    newline=""
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "reference",
            "ground_truth_x",
            "ground_truth_y",
            "predicted_x",
            "predicted_y",
            "error_pixels",
            "confidence",
            "peak_x",
            "peak_y",
            "refined_heatmap_x",
            "refined_heatmap_y"
        ]
    )

    writer.writerows(
        all_results
    )


# ============================================================
# STATISTICS
# ============================================================

errors = np.array(
    [
        r[5]
        for r in all_results
    ]
)

success = (
    errors
    <= SUCCESS_THRESHOLD
)

print()
print("=" * 70)
print("DAY 16 REFINED AI RESULTS")
print("=" * 70)

print(
    f"Patterns tested       : "
    f"{len(errors)}"
)

print(
    f"Mean error            : "
    f"{np.mean(errors):.2f} pixels"
)

print(
    f"Median error          : "
    f"{np.median(errors):.2f} pixels"
)

print(
    f"Maximum error         : "
    f"{np.max(errors):.2f} pixels"
)

print(
    f"Success threshold     : "
    f"{SUCCESS_THRESHOLD:.1f} pixels"
)

print(
    f"Successful patterns   : "
    f"{np.sum(success)}/{len(errors)}"
)

print(
    f"Success rate          : "
    f"{np.mean(success) * 100:.2f}%"
)

print()
print(
    "Results saved:"
)

print(
    output_csv
)

print("=" * 70)
print(
    "DAY 16 REFINEMENT TEST COMPLETE"
)
print("=" * 70)
