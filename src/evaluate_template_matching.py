import os
import cv2
import numpy as np
import csv
import time

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "corrected",
    "reference"
)

SEARCH_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "corrected",
    "search"
)

GT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "corrected",
    "ground_truth"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "day3_corrected"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

NUM_SAMPLES = 10

SUCCESS_THRESHOLD = 50.0


def read_ground_truth(path):

    with open(path) as f:
        lines = f.readlines()

    x = int(lines[0].split("=")[1])
    y = int(lines[1].split("=")[1])

    return x, y


results = []

errors = []
scores = []
times = []

successful = 0


print("=" * 70)
print("DAY 3 - TEMPLATE MATCHING EVALUATION")
print("=" * 70)


for i in range(1, NUM_SAMPLES + 1):

    sample = f"{i:03d}"

    reference_path = os.path.join(
        REF_DIR,
        f"reference_{sample}.png"
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
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    true_x, true_y = read_ground_truth(
        gt_path
    )

    start = time.perf_counter()

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, max_location = \
        cv2.minMaxLoc(result)

    elapsed = time.perf_counter() - start

    pred_x, pred_y = max_location

    error = np.sqrt(
        (pred_x - true_x) ** 2 +
        (pred_y - true_y) ** 2
    )

    errors.append(error)
    scores.append(max_score)
    times.append(elapsed)

    if error <= SUCCESS_THRESHOLD:
        successful += 1

    results.append([
        sample,
        true_x,
        true_y,
        pred_x,
        pred_y,
        error,
        max_score,
        elapsed
    ])

    print(
        f"{sample} | "
        f"GT=({true_x},{true_y}) | "
        f"Pred=({pred_x},{pred_y}) | "
        f"Error={error:.2f} px | "
        f"Score={max_score:.4f} | "
        f"Time={elapsed*1000:.2f} ms"
    )


# ============================================================
# Overall metrics
# ============================================================

mean_error = np.mean(errors)

median_error = np.median(errors)

max_error = np.max(errors)

mean_score = np.mean(scores)

mean_time = np.mean(times)

success_rate = (
    successful / NUM_SAMPLES
) * 100


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Samples              : {NUM_SAMPLES}"
)

print(
    f"Mean error           : {mean_error:.2f} pixels"
)

print(
    f"Median error         : {median_error:.2f} pixels"
)

print(
    f"Maximum error        : {max_error:.2f} pixels"
)

print(
    f"Mean match score     : {mean_score:.4f}"
)

print(
    f"Mean processing time : {mean_time*1000:.2f} ms"
)

print(
    f"Success threshold    : {SUCCESS_THRESHOLD:.0f} pixels"
)

print(
    f"Successful samples   : {successful}/{NUM_SAMPLES}"
)

print(
    f"Success rate         : {success_rate:.2f}%"
)


# ============================================================
# Save CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "template_matching_results.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Sample",
        "True_X",
        "True_Y",
        "Pred_X",
        "Pred_Y",
        "Error_pixels",
        "Match_score",
        "Time_seconds"
    ])

    writer.writerows(results)


print("\nResults saved:")
print(csv_path)

print("=" * 70)
print("DAY 3 EVALUATION COMPLETE")
print("=" * 70)
