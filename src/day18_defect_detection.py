import os
import cv2
import csv
import numpy as np


# ============================================================
# DAY 18 - FABRICATION DEFECT DETECTION
# ============================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REFERENCE_DIR = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "reference"
)

FABRICATED_DIR = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "fabricated"
)

LABEL_FILE = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "labels",
    "defect_labels.csv"
)

RESULT_DIR = os.path.join(
    BASE,
    "results",
    "day18"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

DIFFERENCE_THRESHOLD = 25

MIN_DEFECT_AREA = 10

# Defect percentage above this value = FAIL
DEFECT_PERCENT_THRESHOLD = 0.50


# ============================================================
# DETECT DEFECTS
# ============================================================

def detect_defects(reference, fabricated):

    if reference.shape != fabricated.shape:

        raise RuntimeError(
            f"Image size mismatch: "
            f"{reference.shape} vs {fabricated.shape}"
        )

    # --------------------------------------------------------
    # Absolute pixel difference
    # --------------------------------------------------------

    difference = cv2.absdiff(
        reference,
        fabricated
    )

    # --------------------------------------------------------
    # Threshold difference
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        difference,
        DIFFERENCE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Remove tiny isolated noise
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(mask)

    defect_regions = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_DEFECT_AREA:
            continue

        x = stats[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        filtered_mask[
            labels == i
        ] = 255

        defect_regions.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area)
            }
        )

    # --------------------------------------------------------
    # Defect statistics
    # --------------------------------------------------------

    defect_pixels = int(
        np.count_nonzero(
            filtered_mask
        )
    )

    total_pixels = (
        reference.shape[0] *
        reference.shape[1]
    )

    defect_percentage = (
        defect_pixels /
        total_pixels *
        100.0
    )

    return (
        filtered_mask,
        defect_regions,
        defect_pixels,
        defect_percentage
    )


# ============================================================
# VISUALIZE
# ============================================================

def create_visualization(
    fabricated,
    defect_mask,
    defect_regions,
    passed
):

    # Convert grayscale → RGB

    output = cv2.cvtColor(
        fabricated,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Pattern boundary
    # --------------------------------------------------------

    if passed:

        pattern_color = (
            0,
            220,
            80
        )

    else:

        pattern_color = (
            0,
            0,
            255
        )

    cv2.rectangle(
        output,
        (0, 0),
        (
            output.shape[1] - 1,
            output.shape[0] - 1
        ),
        pattern_color,
        5
    )

    # --------------------------------------------------------
    # Draw defect regions
    # --------------------------------------------------------

    for region in defect_regions:

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            output,
            "DEFECT",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    return output


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("DAY 18 - FABRICATION DEFECT DETECTION")
print("=" * 70)

print()
print("Reference directory:")
print(REFERENCE_DIR)

print()
print("Fabricated directory:")
print(FABRICATED_DIR)

print()


# ============================================================
# LOAD LABELS
# ============================================================

ground_truth = {}

with open(
    LABEL_FILE,
    "r"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        ground_truth[
            int(row["reference_id"])
        ] = row["defect_type"]


# ============================================================
# PROCESS ALL PATTERNS
# ============================================================

results = []

visual_dir = os.path.join(
    RESULT_DIR,
    "visualizations"
)

os.makedirs(
    visual_dir,
    exist_ok=True
)


reference_files = sorted(
    [
        f
        for f in os.listdir(
            REFERENCE_DIR
        )
        if f.endswith(".png")
    ]
)


for filename in reference_files:

    number = int(
        filename.split("_")[1].split(".")[0]
    )

    fabricated_filename = (
        f"fabricated_{number:04d}.png"
    )

    reference_path = os.path.join(
        REFERENCE_DIR,
        filename
    )

    fabricated_path = os.path.join(
        FABRICATED_DIR,
        fabricated_filename
    )

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    fabricated = cv2.imread(
        fabricated_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:

        print(
            "ERROR reading:",
            reference_path
        )

        continue

    if fabricated is None:

        print(
            "ERROR reading:",
            fabricated_path
        )

        continue


    # --------------------------------------------------------
    # Detect
    # --------------------------------------------------------

    (
        defect_mask,
        defect_regions,
        defect_pixels,
        defect_percentage
    ) = detect_defects(
        reference,
        fabricated
    )


    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    if defect_percentage >= DEFECT_PERCENT_THRESHOLD:

        predicted_status = "FAIL"

        passed = False

    else:

        predicted_status = "PASS"

        passed = True


    true_type = ground_truth.get(
        number,
        "UNKNOWN"
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization = create_visualization(
        fabricated,
        defect_mask,
        defect_regions,
        passed
    )


    visual_path = os.path.join(
        visual_dir,
        f"inspection_{number:04d}.png"
    )

    cv2.imwrite(
        visual_path,
        visualization
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        [
            number,
            true_type,
            predicted_status,
            defect_pixels,
            defect_percentage,
            len(defect_regions)
        ]
    )


    print(
        f"{number:03d} | "
        f"GT={true_type:<18} | "
        f"Status={predicted_status:<4} | "
        f"Defect={defect_percentage:.3f}% | "
        f"Regions={len(defect_regions)}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(
    RESULT_DIR,
    "defect_detection_results.csv"
)


with open(
    result_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "reference_id",
            "ground_truth",
            "predicted_status",
            "defect_pixels",
            "defect_percentage",
            "defect_regions"
        ]
    )

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

total = len(results)

good_samples = sum(
    1
    for r in results
    if r[1] == "GOOD"
)

defect_samples = total - good_samples

predicted_pass = sum(
    1
    for r in results
    if r[2] == "PASS"
)

predicted_fail = total - predicted_pass


print()
print("=" * 70)
print("DAY 18 DEFECT DETECTION COMPLETE")
print("=" * 70)

print()
print("Samples tested       :", total)
print("Good patterns        :", good_samples)
print("Defective patterns   :", defect_samples)

print()
print("Predicted PASS       :", predicted_pass)
print("Predicted FAIL       :", predicted_fail)

print()
print("Difference threshold :", DIFFERENCE_THRESHOLD)
print(
    "PASS/FAIL threshold  :",
    DEFECT_PERCENT_THRESHOLD,
    "%"
)

print()
print("Results:")
print(result_file)

print()
print("Visualizations:")
print(visual_dir)

print()
print("=" * 70)
