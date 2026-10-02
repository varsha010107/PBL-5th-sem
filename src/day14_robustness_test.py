import os
import cv2
import csv
import time
import numpy as np
import torch

# ============================================================
# PATHS
# ============================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REFERENCE_DIR = os.path.join(
    BASE,
    "dataset",
    "day13_variation",
    "reference"
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

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "day10",
    "spatial_localizer.pth"
)

RESULT_DIR = os.path.join(
    BASE,
    "results",
    "day14"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# IMPORT AI MODEL
# ============================================================

import sys

SRC_DIR = os.path.dirname(__file__)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from heatmap_model import SpatialLocalizer


# ============================================================
# LOAD AI
# ============================================================

device = torch.device("cpu")

model = SpatialLocalizer()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def template_matching(reference_path, search_path):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise RuntimeError(
            f"Unable to read reference: {reference_path}"
        )

    if search is None:
        raise RuntimeError(
            f"Unable to read search: {search_path}"
        )

    start = time.perf_counter()

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, location = cv2.minMaxLoc(result)

    elapsed = (
        time.perf_counter() - start
    ) * 1000.0

    x, y = location

    return (
        x,
        y,
        float(score),
        elapsed
    )


# ============================================================
# SPATIAL AI
# ============================================================

def spatial_ai(search_path):

    image = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read search: {search_path}"
        )

    original_h, original_w = image.shape

    start = time.perf_counter()

    resized = cv2.resize(
        image,
        (256, 256)
    )

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.tensor(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        heatmap = model(
            tensor
        )

    heatmap = heatmap[0, 0]

    index = torch.argmax(
        heatmap
    )

    hy, hx = np.unravel_index(
        index.item(),
        heatmap.shape
    )

    heatmap_h, heatmap_w = heatmap.shape

    x = (
        hx *
        original_w /
        heatmap_w
    )

    y = (
        hy *
        original_h /
        heatmap_h
    )

    confidence = heatmap[
        hy,
        hx
    ].item()

    elapsed = (
        time.perf_counter() - start
    ) * 1000.0

    return (
        x,
        y,
        float(confidence),
        elapsed
    )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(csv_path):

    labels = []

    with open(
        csv_path,
        "r"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            labels.append(
                {
                    "reference":
                        int(row["reference"]),

                    "x":
                        int(row["x"]),

                    "y":
                        int(row["y"])
                }
            )

    return labels


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("DAY 14 - ROBUSTNESS EVALUATION")
print("=" * 70)

print("Device:", device)
print("Reference directory:", REFERENCE_DIR)
print("Search directory:", SEARCH_DIR)
print("Model:", MODEL_PATH)

print()


# ============================================================
# RESULTS
# ============================================================

template_results = []
ai_results = []


search_files = sorted(
    [
        f
        for f in os.listdir(SEARCH_DIR)
        if f.endswith(".png")
    ]
)

print(
    "Search images:",
    len(search_files)
)

print()


# ============================================================
# PROCESS EACH SEARCH IMAGE
# ============================================================

for search_file in search_files:

    search_number = (
        os.path.splitext(
            search_file
        )[0]
    )

    search_path = os.path.join(
        SEARCH_DIR,
        search_file
    )

    label_path = os.path.join(
        LABEL_DIR,
        search_number + ".csv"
    )

    labels = load_labels(
        label_path
    )

    print("-" * 70)
    print(
        f"Processing {search_file}"
    )
    print(
        f"Patterns: {len(labels)}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # SPATIAL AI
    # --------------------------------------------------------

    ai_x, ai_y, ai_confidence, ai_time = spatial_ai(
        search_path
    )

    print(
        f"AI whole-image prediction:"
    )

    print(
        f"X={ai_x:.2f}, "
        f"Y={ai_y:.2f}, "
        f"Confidence={ai_confidence:.4f}, "
        f"Time={ai_time:.2f} ms"
    )

    # --------------------------------------------------------
    # PROCESS EACH REFERENCE
    # --------------------------------------------------------

    for item in labels:

        ref_number = item["reference"]

        gt_x = item["x"]
        gt_y = item["y"]

        reference_file = (
            f"reference_{ref_number:04d}.png"
        )

        reference_path = os.path.join(
            REFERENCE_DIR,
            reference_file
        )

        # ====================================================
        # TEMPLATE MATCHING
        # ====================================================

        try:

            pred_x, pred_y, score, elapsed = (
                template_matching(
                    reference_path,
                    search_path
                )
            )

            error = np.sqrt(
                (pred_x - gt_x) ** 2 +
                (pred_y - gt_y) ** 2
            )

            success = (
                error <= 50.0
            )

            template_results.append(
                [
                    ref_number,
                    search_file,
                    gt_x,
                    gt_y,
                    pred_x,
                    pred_y,
                    error,
                    score,
                    elapsed,
                    success
                ]
            )

        except Exception as e:

            print(
                "Template error:",
                ref_number,
                e
            )

        # ====================================================
        # AI RESULT
        # ====================================================

        # AI is a whole-image prediction.
        # Therefore compare it with the corresponding
        # ground-truth position.

        ai_error = np.sqrt(
            (ai_x - gt_x) ** 2 +
            (ai_y - gt_y) ** 2
        )

        ai_success = (
            ai_error <= 50.0
        )

        ai_results.append(
            [
                ref_number,
                search_file,
                gt_x,
                gt_y,
                ai_x,
                ai_y,
                ai_error,
                ai_confidence,
                ai_time,
                ai_success
            ]
        )

        print(
            f"Ref {ref_number:03d} | "
            f"GT=({gt_x},{gt_y}) | "
            f"Template=({pred_x},{pred_y}) | "
            f"Error={error:.2f}px"
        )


# ============================================================
# SAVE TEMPLATE CSV
# ============================================================

template_csv = os.path.join(
    RESULT_DIR,
    "template_results.csv"
)

with open(
    template_csv,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "reference",
            "search_image",
            "ground_truth_x",
            "ground_truth_y",
            "predicted_x",
            "predicted_y",
            "error_pixels",
            "match_score",
            "processing_time_ms",
            "success"
        ]
    )

    writer.writerows(
        template_results
    )


# ============================================================
# SAVE AI CSV
# ============================================================

ai_csv = os.path.join(
    RESULT_DIR,
    "ai_results.csv"
)

with open(
    ai_csv,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "reference",
            "search_image",
            "ground_truth_x",
            "ground_truth_y",
            "predicted_x",
            "predicted_y",
            "error_pixels",
            "confidence",
            "processing_time_ms",
            "success"
        ]
    )

    writer.writerows(
        ai_results
    )


# ============================================================
# CALCULATE SUMMARY
# ============================================================

template_errors = np.array(
    [
        r[6]
        for r in template_results
    ]
)

ai_errors = np.array(
    [
        r[6]
        for r in ai_results
    ]
)

template_times = np.array(
    [
        r[8]
        for r in template_results
    ]
)

ai_times = np.array(
    [
        r[8]
        for r in ai_results
    ]
)

template_success = sum(
    r[9]
    for r in template_results
)

ai_success = sum(
    r[9]
    for r in ai_results
)

template_count = len(
    template_results
)

ai_count = len(
    ai_results
)


# ============================================================
# SUMMARY
# ============================================================

summary_csv = os.path.join(
    RESULT_DIR,
    "comparison_summary.csv"
)

with open(
    summary_csv,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "metric",
            "template_matching",
            "spatial_ai"
        ]
    )

    writer.writerow(
        [
            "samples",
            template_count,
            ai_count
        ]
    )

    writer.writerow(
        [
            "mean_error_pixels",
            np.mean(template_errors),
            np.mean(ai_errors)
        ]
    )

    writer.writerow(
        [
            "median_error_pixels",
            np.median(template_errors),
            np.median(ai_errors)
        ]
    )

    writer.writerow(
        [
            "maximum_error_pixels",
            np.max(template_errors),
            np.max(ai_errors)
        ]
    )

    writer.writerow(
        [
            "mean_processing_time_ms",
            np.mean(template_times),
            np.mean(ai_times)
        ]
    )

    writer.writerow(
        [
            "success_rate_percent",
            100.0 * template_success /
            template_count,
            100.0 * ai_success /
            ai_count
        ]
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("DAY 14 ROBUSTNESS RESULTS")
print("=" * 70)

print(
    f"Samples              : {template_count}"
)

print()

print(
    f"Template mean error  : "
    f"{np.mean(template_errors):.2f} pixels"
)

print(
    f"Template median      : "
    f"{np.median(template_errors):.2f} pixels"
)

print(
    f"Template maximum     : "
    f"{np.max(template_errors):.2f} pixels"
)

print(
    f"Template success     : "
    f"{100.0 * template_success / template_count:.2f}%"
)

print(
    f"Template time        : "
    f"{np.mean(template_times):.2f} ms"
)

print()

print(
    f"AI mean error        : "
    f"{np.mean(ai_errors):.2f} pixels"
)

print(
    f"AI median            : "
    f"{np.median(ai_errors):.2f} pixels"
)

print(
    f"AI maximum           : "
    f"{np.max(ai_errors):.2f} pixels"
)

print(
    f"AI success           : "
    f"{100.0 * ai_success / ai_count:.2f}%"
)

print(
    f"AI time              : "
    f"{np.mean(ai_times):.2f} ms"
)

print()

print("Results saved:")
print(template_csv)
print(ai_csv)
print(summary_csv)

print("=" * 70)
print("DAY 14 ROBUSTNESS TEST COMPLETE")
print("=" * 70)
