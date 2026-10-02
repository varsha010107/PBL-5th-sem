import os
import cv2
import glob
import time
import csv
import numpy as np

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(
    BASE, "dataset", "day4_reference"
)

SEARCH_DIR = os.path.join(
    BASE, "dataset", "day4_search"
)

GT_DIR = os.path.join(
    BASE, "dataset", "day4_ground_truth"
)

RESULT_DIR = os.path.join(
    BASE, "results", "day4"
)

os.makedirs(RESULT_DIR, exist_ok=True)


def read_ground_truth(path):

    x = None
    y = None

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if line.startswith("target_x="):
                x = int(line.split("=")[1])

            elif line.startswith("target_y="):
                y = int(line.split("=")[1])

    return x, y


def calculate_error(pred_x, pred_y, gt_x, gt_y):

    return np.sqrt(
        (pred_x - gt_x) ** 2 +
        (pred_y - gt_y) ** 2
    )


print("=" * 70)
print("DAY 4 - HARD DATASET TEMPLATE MATCHING EVALUATION")
print("=" * 70)

results = []

reference_files = sorted(
    glob.glob(
        os.path.join(
            REF_DIR,
            "reference_*.png"
        )
    )
)

for ref_path in reference_files:

    filename = os.path.basename(ref_path)

    sample = filename.replace(
        "reference_", ""
    ).replace(
        ".png", ""
    )

    search_path = os.path.join(
        SEARCH_DIR,
        f"search_{sample}.png"
    )

    gt_path = os.path.join(
        GT_DIR,
        f"ground_truth_{sample}.txt"
    )

    reference = cv2.imread(
        ref_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    gt_x, gt_y = read_ground_truth(
        gt_path
    )

    start = time.perf_counter()

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(
        result
    )

    end = time.perf_counter()

    pred_x, pred_y = max_loc

    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    processing_time = (
        end - start
    ) * 1000

    success = error <= 50

    results.append([
        sample,
        gt_x,
        gt_y,
        pred_x,
        pred_y,
        error,
        max_val,
        processing_time,
        success
    ])

    print(
        f"{sample} | "
        f"GT=({gt_x},{gt_y}) | "
        f"Pred=({pred_x},{pred_y}) | "
        f"Error={error:.2f} px | "
        f"Score={max_val:.4f} | "
        f"Time={processing_time:.2f} ms"
    )


errors = [
    r[5]
    for r in results
]

scores = [
    r[6]
    for r in results
]

times = [
    r[7]
    for r in results
]

successful = [
    r for r in results
    if r[8]
]


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Samples              : {len(results)}"
)

print(
    f"Mean error           : {np.mean(errors):.2f} pixels"
)

print(
    f"Median error         : {np.median(errors):.2f} pixels"
)

print(
    f"Maximum error        : {np.max(errors):.2f} pixels"
)

print(
    f"Mean match score     : {np.mean(scores):.4f}"
)

print(
    f"Mean processing time : {np.mean(times):.2f} ms"
)

print(
    "Success threshold    : 50 pixels"
)

print(
    f"Successful samples   : "
    f"{len(successful)}/{len(results)}"
)

print(
    f"Success rate         : "
    f"{100 * len(successful) / len(results):.2f}%"
)


csv_path = os.path.join(
    RESULT_DIR,
    "day4_template_results.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample",
        "gt_x",
        "gt_y",
        "pred_x",
        "pred_y",
        "error_pixels",
        "match_score",
        "processing_time_ms",
        "success"
    ])

    writer.writerows(results)


print("\nResults saved:")
print(csv_path)

print("=" * 70)
print("DAY 4 EVALUATION COMPLETE")
print("=" * 70)
