import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def read_ground_truth(path):

    with open(path) as f:
        lines = f.readlines()

    x = int(
        lines[0].split("=")[1]
    )

    y = int(
        lines[1].split("=")[1]
    )

    return x, y


# ------------------------------------------------------------
# Sample
# ------------------------------------------------------------

sample = "001"

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


# ------------------------------------------------------------
# Read images
# ------------------------------------------------------------

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
        "Reference image could not be loaded."
    )

if search is None:
    raise RuntimeError(
        "Search image could not be loaded."
    )


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

true_x, true_y = read_ground_truth(
    gt_path
)


# ------------------------------------------------------------
# Template matching
# ------------------------------------------------------------

result = cv2.matchTemplate(
    search,
    reference,
    cv2.TM_CCOEFF_NORMED
)


min_value, max_value, min_location, max_location = \
    cv2.minMaxLoc(result)


pred_x, pred_y = max_location


# ------------------------------------------------------------
# Error
# ------------------------------------------------------------

error = np.sqrt(
    (pred_x - true_x) ** 2 +
    (pred_y - true_y) ** 2
)


# ------------------------------------------------------------
# Print result
# ------------------------------------------------------------

print("=" * 60)
print("DAY 3 - CORRECTED TEMPLATE MATCHING")
print("=" * 60)

print(
    f"Reference size : "
    f"{reference.shape[::-1]}"
)

print(
    f"Search size    : "
    f"{search.shape[::-1]}"
)

print(
    f"Ground truth   : "
    f"({true_x}, {true_y})"
)

print(
    f"Prediction     : "
    f"({pred_x}, {pred_y})"
)

print(
    f"Match score    : "
    f"{max_value:.4f}"
)

print(
    f"Error          : "
    f"{error:.2f} pixels"
)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 10)
)

plt.imshow(
    search,
    cmap="gray"
)

plt.plot(
    true_x,
    true_y,
    "r+",
    markersize=20,
    markeredgewidth=3,
    label="Ground Truth"
)

plt.plot(
    pred_x,
    pred_y,
    "bo",
    markersize=12,
    fillstyle="none",
    label="Prediction"
)

plt.title(
    "Corrected Template Matching"
)

plt.legend()

plt.axis("off")

output = os.path.join(
    RESULT_DIR,
    "template_matching_001.png"
)

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print()
print(
    f"Visualization saved:\n{output}"
)

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
