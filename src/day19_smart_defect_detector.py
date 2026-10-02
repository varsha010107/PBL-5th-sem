import cv2
import csv
import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE = "dataset/day18_defects"

REFERENCE_DIR = os.path.join(
    BASE,
    "reference"
)

FABRICATED_DIR = os.path.join(
    BASE,
    "fabricated"
)

LABEL_FILE = os.path.join(
    BASE,
    "labels",
    "defect_labels.csv"
)

RESULT_DIR = "results/day19"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Very small differences can indicate particles
PARTICLE_DIFF_PIXELS = 30

# Difference intensity
PARTICLE_MAX_DIFF = 5

# Minimum number of changed pixels for structural defects
STRUCTURAL_DIFF_PIXELS = 150

# Area threshold for large fabrication defects
STRUCTURAL_PERCENTAGE = 0.15


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABEL_FILE) as f:

    labels = {
        int(row["reference_id"]):
        row["defect_type"]

        for row in csv.DictReader(f)
    }


# ============================================================
# SMART DEFECT ANALYSIS
# ============================================================

def analyze_defect(reference, fabricated):

    # --------------------------------------------------------
    # Absolute pixel difference
    # --------------------------------------------------------

    diff = cv2.absdiff(
        reference,
        fabricated
    )

    changed_mask = (
        diff > 2
    ).astype(
        np.uint8
    ) * 255


    changed_pixels = int(
        np.count_nonzero(
            changed_mask
        )
    )


    total_pixels = (
        reference.shape[0] *
        reference.shape[1]
    )


    defect_percentage = (
        changed_pixels /
        total_pixels *
        100
    )


    max_difference = int(
        diff.max()
    )


    mean_difference = float(
        diff.mean()
    )


    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            changed_mask,
            8
        )
    )


    regions = []


    for i in range(1, num_labels):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < 2:
            continue

        x = int(
            stats[i, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[i, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[i, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[i, cv2.CC_STAT_HEIGHT]
        )

        regions.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area
            }
        )


    regions.sort(
        key=lambda r: r["area"],
        reverse=True
    )


    # --------------------------------------------------------
    # DECISION LOGIC
    # --------------------------------------------------------

    status = "PASS"

    defect_type = "GOOD"

    confidence = 0.0


    # ========================================================
    # SMALL PARTICLE DETECTION
    # ========================================================

    if changed_pixels >= PARTICLE_DIFF_PIXELS:

        # Small localized regions are characteristic
        # of particle contamination.

        if (
            len(regions) > 0
            and
            max_difference >= PARTICLE_MAX_DIFF
        ):

            largest_area = regions[0]["area"]

            if largest_area < 500:

                status = "FAIL"

                defect_type = "PARTICLE"

                confidence = min(
                    99.0,
                    60.0
                    + defect_percentage * 100
                    + max_difference * 0.10
                )


    # ========================================================
    # STRUCTURAL DEFECT DETECTION
    # ========================================================

    if status == "PASS":

        if (
            changed_pixels >= STRUCTURAL_DIFF_PIXELS
            and
            defect_percentage >= STRUCTURAL_PERCENTAGE
        ):

            status = "FAIL"

            defect_type = "FABRICATION DEFECT"

            confidence = min(
                99.0,
                70.0
                + defect_percentage * 40
            )


    # ========================================================
    # VERY STRONG DIFFERENCE
    # ========================================================

    if status == "PASS":

        if defect_percentage >= 0.30:

            status = "FAIL"

            defect_type = "FABRICATION DEFECT"

            confidence = min(
                99.0,
                80.0
                + defect_percentage * 20
            )


    return {
        "status": status,
        "defect_type": defect_type,
        "confidence": confidence,
        "changed_pixels": changed_pixels,
        "defect_percentage": defect_percentage,
        "max_difference": max_difference,
        "mean_difference": mean_difference,
        "regions": regions
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("DAY 19 - SMART FABRICATION DEFECT DETECTION")
print("=" * 70)

results = []


for i in range(1, 101):

    reference_file = os.path.join(
        REFERENCE_DIR,
        f"reference_{i:04d}.png"
    )

    fabricated_file = os.path.join(
        FABRICATED_DIR,
        f"fabricated_{i:04d}.png"
    )


    reference = cv2.imread(
        reference_file,
        cv2.IMREAD_GRAYSCALE
    )

    fabricated = cv2.imread(
        fabricated_file,
        cv2.IMREAD_GRAYSCALE
    )


    if reference is None:

        print(
            f"{i:03d} | Reference missing"
        )

        continue


    if fabricated is None:

        print(
            f"{i:03d} | Fabricated image missing"
        )

        continue


    analysis = analyze_defect(
        reference,
        fabricated
    )


    gt = labels[i]


    print(
        f"{i:03d} | "
        f"GT={gt:<18} | "
        f"Pred={analysis['status']:<4} | "
        f"Type={analysis['defect_type']:<20} | "
        f"Diff={analysis['defect_percentage']:.4f}% | "
        f"Regions={len(analysis['regions'])}"
    )


    results.append(
        {
            "reference_id": i,
            "ground_truth": gt,
            "predicted_status":
                analysis["status"],
            "predicted_type":
                analysis["defect_type"],
            "confidence":
                analysis["confidence"],
            "changed_pixels":
                analysis["changed_pixels"],
            "defect_percentage":
                analysis["defect_percentage"],
            "max_difference":
                analysis["max_difference"],
            "mean_difference":
                analysis["mean_difference"],
            "defect_regions":
                len(analysis["regions"])
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULT_DIR,
    "smart_defect_results.csv"
)


with open(
    output_file,
    "w",
    newline=""
) as f:

    fieldnames = [
        "reference_id",
        "ground_truth",
        "predicted_status",
        "predicted_type",
        "confidence",
        "changed_pixels",
        "defect_percentage",
        "max_difference",
        "mean_difference",
        "defect_regions"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

good_total = sum(
    r["ground_truth"] == "GOOD"
    for r in results
)

defective_total = sum(
    r["ground_truth"] != "GOOD"
    for r in results
)

correct_good = sum(
    r["ground_truth"] == "GOOD"
    and
    r["predicted_status"] == "PASS"
    for r in results
)

correct_defect = sum(
    r["ground_truth"] != "GOOD"
    and
    r["predicted_status"] == "FAIL"
    for r in results
)

total_correct = (
    correct_good +
    correct_defect
)

accuracy = (
    total_correct /
    len(results) *
    100
)


print()
print("=" * 70)
print("DAY 19 SMART DEFECT RESULTS")
print("=" * 70)

print(
    f"Samples tested       : {len(results)}"
)

print(
    f"Good patterns        : {good_total}"
)

print(
    f"Defective patterns   : {defective_total}"
)

print(
    f"Correct GOOD         : "
    f"{correct_good}/{good_total}"
)

print(
    f"Correct DEFECT       : "
    f"{correct_defect}/{defective_total}"
)

print(
    f"Overall accuracy     : "
    f"{accuracy:.2f}%"
)

print()
print(
    f"Results saved:"
)

print(
    output_file
)

print("=" * 70)
print(
    "DAY 19 SMART DEFECT DETECTION COMPLETE"
)
print("=" * 70)
