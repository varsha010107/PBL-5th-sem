import os
import csv
import time
import cv2
import numpy as np


BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TEST_DIR = os.path.join(
    BASE, "dataset", "day9_ai", "test"
)

REFERENCE_DIR = os.path.join(
    TEST_DIR, "reference"
)

SEARCH_DIR = os.path.join(
    TEST_DIR, "search"
)

LABEL_DIR = os.path.join(
    TEST_DIR, "labels"
)

RESULT_DIR = os.path.join(
    BASE, "results", "day12"
)

os.makedirs(RESULT_DIR, exist_ok=True)


print("=" * 70)
print("DAY 12 - TEMPLATE MATCHING ON AI TEST DATASET")
print("=" * 70)


results = []


for i in range(1, 101):

    ref_file = os.path.join(
        REFERENCE_DIR,
        f"reference_{i:04d}.png"
    )

    search_file = os.path.join(
        SEARCH_DIR,
        f"search_{i:04d}.png"
    )

    label_file = os.path.join(
        LABEL_DIR,
        f"label_{i:04d}.txt"
    )


    reference = cv2.imread(
        ref_file,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_file,
        cv2.IMREAD_GRAYSCALE
    )


    with open(label_file, "r") as f:
        gt_x, gt_y = map(
            float,
            f.readline().strip().split(",")
        )


    start = time.perf_counter()


    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )


    _, max_score, _, max_location = cv2.minMaxLoc(
        result
    )


    pred_x = max_location[0]
    pred_y = max_location[1]


    processing_time = (
        time.perf_counter() - start
    ) * 1000


    error = np.sqrt(
        (pred_x - gt_x) ** 2
        +
        (pred_y - gt_y) ** 2
    )


    success = error <= 50


    results.append([
        i,
        gt_x,
        gt_y,
        pred_x,
        pred_y,
        error,
        max_score,
        processing_time,
        success
    ])


    if i <= 10:
        print(
            f"{i:03d} | "
            f"GT=({gt_x:.0f},{gt_y:.0f}) | "
            f"Pred=({pred_x},{pred_y}) | "
            f"Error={error:.2f}px | "
            f"Score={max_score:.4f} | "
            f"Time={processing_time:.2f}ms"
        )


csv_file = os.path.join(
    RESULT_DIR,
    "template_results.csv"
)


with open(
    csv_file,
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


errors = [
    r[5] for r in results
]

times = [
    r[7] for r in results
]

successful = sum(
    r[8] for r in results
)


print()
print("=" * 70)
print("TEMPLATE MATCHING RESULTS")
print("=" * 70)

print("Samples              :", len(results))
print(f"Mean error           : {np.mean(errors):.2f} pixels")
print(f"Median error         : {np.median(errors):.2f} pixels")
print(f"Maximum error        : {np.max(errors):.2f} pixels")
print(f"Mean processing time : {np.mean(times):.2f} ms")
print(
    f"Successful samples   : "
    f"{successful}/{len(results)}"
)
print(
    f"Success rate         : "
    f"{successful / len(results) * 100:.2f}%"
)

print()
print("Results saved:")
print(csv_file)

print("=" * 70)
print("DAY 12 TEMPLATE TEST COMPLETE")
print("=" * 70)
