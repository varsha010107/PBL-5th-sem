import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DAY 3
# Classical Computer-Vision Baseline
# Template Matching
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(
    BASE_DIR, "dataset", "day2_reference"
)

SEARCH_DIR = os.path.join(
    BASE_DIR, "dataset", "day2_search"
)

GT_DIR = os.path.join(
    BASE_DIR, "dataset", "day2_ground_truth"
)

RESULT_DIR = os.path.join(
    BASE_DIR, "results", "day3"
)

os.makedirs(RESULT_DIR, exist_ok=True)


def read_ground_truth(path):

    with open(path, "r") as f:
        lines = f.readlines()

    x = int(lines[0].split("=")[1])
    y = int(lines[1].split("=")[1])

    return x, y


def calculate_error(pred_x, pred_y, true_x, true_y):

    error = np.sqrt(
        (pred_x - true_x) ** 2 +
        (pred_y - true_y) ** 2
    )

    return error


def localize(reference_path, search_path):

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
            f"Could not read reference image: {reference_path}"
        )

    if search is None:
        raise RuntimeError(
            f"Could not read search image: {search_path}"
        )

    # Template matching
    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    # Find best match
    _, max_value, _, max_location = cv2.minMaxLoc(result)

    predicted_x = max_location[0]
    predicted_y = max_location[1]

    return predicted_x, predicted_y, max_value


# ============================================================
# Test one sample
# ============================================================

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

true_x, true_y = read_ground_truth(gt_path)

pred_x, pred_y, confidence = localize(
    reference_path,
    search_path
)

error = calculate_error(
    pred_x,
    pred_y,
    true_x,
    true_y
)

print("=" * 60)
print("DAY 3 - TEMPLATE MATCHING")
print("=" * 60)

print(f"Sample              : {sample}")
print(f"Ground truth        : ({true_x}, {true_y})")
print(f"Predicted location  : ({pred_x}, {pred_y})")
print(f"Match score         : {confidence:.4f}")
print(f"Localization error  : {error:.2f} pixels")


# ============================================================
# Visualization
# ============================================================

search_image = cv2.imread(
    search_path,
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(8, 8))

plt.imshow(
    search_image,
    cmap="gray"
)

# Ground truth
plt.plot(
    true_x,
    true_y,
    "r+",
    markersize=20,
    markeredgewidth=3,
    label="Ground Truth"
)

# Prediction
plt.plot(
    pred_x,
    pred_y,
    "bo",
    markersize=10,
    fillstyle="none",
    label="Prediction"
)

plt.title(
    f"Template Matching\n"
    f"GT=({true_x},{true_y})  "
    f"Pred=({pred_x},{pred_y})"
)

plt.legend()
plt.axis("off")

output_path = os.path.join(
    RESULT_DIR,
    "template_matching_001.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print()
print(f"Visualization saved:")
print(output_path)

print("=" * 60)
print("DAY 3 TEST COMPLETE")
print("=" * 60)

