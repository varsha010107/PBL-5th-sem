import os
import sys
import csv
import cv2
import numpy as np

# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from defect_engine import analyze_defect


# ============================================================
# DIRECTORIES
# ============================================================

REFERENCE_DIR = "dataset/day18_defects/reference"
FABRICATED_DIR = "dataset/day18_defects/fabricated"

RESULT_DIR = "results/day20"
VIS_DIR = os.path.join(RESULT_DIR, "visualizations")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

GRID_SIZE = 500
IMAGE_SIZE = 2500


# ============================================================
# LOCALIZE PATTERN USING REFERENCE TEMPLATE
# ============================================================

def localize_pattern(reference, search):

    result = cv2.matchTemplate(
        search,
        reference,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, location = cv2.minMaxLoc(result)

    x = int(location[0])
    y = int(location[1])

    return x, y, float(score)


# ============================================================
# EXTRACT FABRICATED PATTERN
# ============================================================

def extract_region(image, x, y):

    h, w = image.shape[:2]

    x2 = min(
        x + GRID_SIZE,
        w
    )

    y2 = min(
        y + GRID_SIZE,
        h
    )

    return image[
        y:y2,
        x:x2
    ]


# ============================================================
# DRAW DEFECT REGIONS
# ============================================================

def draw_defects(image, result):

    output = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    for region in result["regions"]:

        x = int(region["x"])
        y = int(region["y"])

        w = int(region["w"])
        h = int(region["h"])

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

    return output


# ============================================================
# MAIN INSPECTION
# ============================================================

def inspect_pattern(reference_id):

    reference_path = os.path.join(
        REFERENCE_DIR,
        f"reference_{reference_id:04d}.png"
    )

    fabricated_path = os.path.join(
        FABRICATED_DIR,
        f"fabricated_{reference_id:04d}.png"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(reference_path):

        print(
            f"Reference missing: {reference_path}"
        )

        return None

    if not os.path.exists(fabricated_path):

        print(
            f"Fabricated missing: {fabricated_path}"
        )

        return None


    # --------------------------------------------------------
    # Load reference
    # --------------------------------------------------------

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )


    # --------------------------------------------------------
    # Load fabricated
    # --------------------------------------------------------

    fabricated = cv2.imread(
        fabricated_path,
        cv2.IMREAD_GRAYSCALE
    )


    if reference is None or fabricated is None:

        print(
            f"Unable to load pattern {reference_id}"
        )

        return None


    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    # For a fabricated 500x500 pattern,
    # the pattern itself is inspected directly.

    x = 0
    y = 0
    localization_score = 1.0


    # --------------------------------------------------------
    # DEFECT ANALYSIS
    # --------------------------------------------------------

    defect_result = analyze_defect(
        reference,
        fabricated
    )


    status = defect_result["status"]

    defect_type = defect_result[
        "defect_type"
    ]

    confidence = defect_result[
        "confidence"
    ]

    difference = defect_result[
        "defect_percentage"
    ]

    changed_pixels = defect_result[
        "changed_pixels"
    ]

    regions = defect_result[
        "regions"
    ]


    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    visual = draw_defects(
        fabricated,
        defect_result
    )


    # Pattern boundary = GREEN

    cv2.rectangle(
        visual,
        (0, 0),
        (
            fabricated.shape[1] - 1,
            fabricated.shape[0] - 1
        ),
        (0, 255, 0),
        4
    )


    # --------------------------------------------------------
    # RESULT TEXT
    # --------------------------------------------------------

    color = (
        (0, 255, 0)
        if status == "PASS"
        else
        (0, 0, 255)
    )


    cv2.putText(
        visual,
        f"{status} - {defect_type}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


    cv2.putText(
        visual,
        f"Difference: {difference:.4f}%",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    output_path = os.path.join(
        VIS_DIR,
        f"inspection_{reference_id:04d}.png"
    )

    cv2.imwrite(
        output_path,
        visual
    )


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"{reference_id:03d} | "
        f"X={x:4d} Y={y:4d} | "
        f"{status:4s} | "
        f"{defect_type:20s} | "
        f"Diff={difference:.4f}% | "
        f"Regions={len(regions)}"
    )


    return {
        "reference_id": reference_id,
        "x": x,
        "y": y,
        "localization_score": localization_score,
        "status": status,
        "defect_type": defect_type,
        "confidence": confidence,
        "difference": difference,
        "changed_pixels": changed_pixels,
        "regions": len(regions),
        "visualization": output_path
    }


# ============================================================
# RUN ALL PATTERNS
# ============================================================

def main():

    print("=" * 75)
    print("DAY 20 - INTEGRATED SEMICONDUCTOR INSPECTION")
    print("=" * 75)
    print()

    results = []


    for reference_id in range(1, 101):

        result = inspect_pattern(
            reference_id
        )

        if result is not None:

            results.append(
                result
            )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        RESULT_DIR,
        "integrated_results.csv"
    )


    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reference_id",
                "x",
                "y",
                "localization_score",
                "status",
                "defect_type",
                "confidence",
                "difference",
                "changed_pixels",
                "regions",
                "visualization"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    total = len(results)

    passed = sum(
        r["status"] == "PASS"
        for r in results
    )

    failed = sum(
        r["status"] == "FAIL"
        for r in results
    )


    print()
    print("=" * 75)
    print("DAY 20 INTEGRATED RESULTS")
    print("=" * 75)

    print(
        f"Samples tested       : {total}"
    )

    print(
        f"PASS                 : {passed}"
    )

    print(
        f"FAIL                 : {failed}"
    )

    if total > 0:

        print(
            f"PASS rate            : "
            f"{passed / total * 100:.2f}%"
        )

        print(
            f"FAIL rate            : "
            f"{failed / total * 100:.2f}%"
        )

    print()
    print(
        f"CSV saved            : {csv_path}"
    )

    print(
        f"Visualizations       : {VIS_DIR}"
    )

    print("=" * 75)
    print("DAY 20 INTEGRATED INSPECTION COMPLETE")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
