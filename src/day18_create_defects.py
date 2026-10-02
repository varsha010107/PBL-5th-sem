import os
import cv2
import csv
import random
import numpy as np


# ============================================================
# DAY 18 - FABRICATION DEFECT DATASET GENERATOR
# ============================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SOURCE_DIR = os.path.join(
    BASE,
    "dataset",
    "day13_variation",
    "reference"
)

OUTPUT_REFERENCE = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "reference"
)

OUTPUT_FABRICATED = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "fabricated"
)

OUTPUT_LABELS = os.path.join(
    BASE,
    "dataset",
    "day18_defects",
    "labels"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 18

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_REFERENCE, exist_ok=True)
os.makedirs(OUTPUT_FABRICATED, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


# ============================================================
# DEFECT GENERATORS
# ============================================================

def add_missing_material(image):
    """
    Simulate missing/broken fabricated material.
    """

    result = image.copy()

    h, w = result.shape

    # Random rectangular region
    x = random.randint(
        int(w * 0.20),
        int(w * 0.65)
    )

    y = random.randint(
        int(h * 0.20),
        int(h * 0.65)
    )

    width = random.randint(
        25,
        70
    )

    height = random.randint(
        25,
        70
    )

    # Estimate background from nearby image
    border = int(np.median(result))

    cv2.rectangle(
        result,
        (x, y),
        (
            min(x + width, w - 1),
            min(y + height, h - 1)
        ),
        int(border),
        -1
    )

    return result


def add_extra_material(image):
    """
    Simulate unwanted deposited material.
    """

    result = image.copy()

    h, w = result.shape

    x = random.randint(
        int(w * 0.15),
        int(w * 0.75)
    )

    y = random.randint(
        int(h * 0.15),
        int(h * 0.75)
    )

    width = random.randint(
        20,
        60
    )

    height = random.randint(
        20,
        60
    )

    foreground = int(
        np.percentile(result, 90)
    )

    cv2.rectangle(
        result,
        (x, y),
        (
            min(x + width, w - 1),
            min(y + height, h - 1)
        ),
        foreground,
        -1
    )

    return result


def add_particle(image):
    """
    Simulate particle contamination.
    """

    result = image.copy()

    h, w = result.shape

    x = random.randint(
        20,
        w - 20
    )

    y = random.randint(
        20,
        h - 20
    )

    radius = random.randint(
        5,
        15
    )

    intensity = random.choice(
        [
            0,
            255
        ]
    )

    cv2.circle(
        result,
        (x, y),
        radius,
        intensity,
        -1
    )

    return result


def add_deformation(image):
    """
    Simulate a small local fabrication deformation.
    """

    result = image.copy()

    h, w = result.shape

    x1 = random.randint(
        int(w * 0.25),
        int(w * 0.60)
    )

    y1 = random.randint(
        int(h * 0.25),
        int(h * 0.60)
    )

    x2 = min(
        x1 + random.randint(30, 80),
        w - 1
    )

    y2 = min(
        y1 + random.randint(30, 80),
        h - 1
    )

    roi = result[
        y1:y2,
        x1:x2
    ]

    if roi.size > 0:

        # Slight local blur
        roi = cv2.GaussianBlur(
            roi,
            (9, 9),
            1.5
        )

        result[
            y1:y2,
            x1:x2
        ] = roi

    return result


# ============================================================
# GENERATE DATASET
# ============================================================

print("=" * 70)
print("DAY 18 - FABRICATION DEFECT DATASET")
print("=" * 70)

files = sorted(
    [
        f
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".png")
    ]
)

print()
print("Reference images found:", len(files))
print()


labels = []


for index, filename in enumerate(files, start=1):

    source_path = os.path.join(
        SOURCE_DIR,
        filename
    )

    reference = cv2.imread(
        source_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:

        print(
            "WARNING: Unable to read",
            filename
        )

        continue


    # --------------------------------------------------------
    # Copy reference
    # --------------------------------------------------------

    reference_name = (
        f"reference_{index:04d}.png"
    )

    fabricated_name = (
        f"fabricated_{index:04d}.png"
    )


    reference_output = os.path.join(
        OUTPUT_REFERENCE,
        reference_name
    )

    fabricated_output = os.path.join(
        OUTPUT_FABRICATED,
        fabricated_name
    )


    cv2.imwrite(
        reference_output,
        reference
    )


    # --------------------------------------------------------
    # Select defect type
    # --------------------------------------------------------

    # Every 5th pattern is GOOD.
    # Remaining patterns receive a synthetic defect.

    if index % 5 == 0:

        defect_type = "GOOD"

        fabricated = reference.copy()


    else:

        defect_type = random.choice(
            [
                "MISSING_MATERIAL",
                "EXTRA_MATERIAL",
                "PARTICLE",
                "DEFORMATION"
            ]
        )


        if defect_type == "MISSING_MATERIAL":

            fabricated = add_missing_material(
                reference
            )


        elif defect_type == "EXTRA_MATERIAL":

            fabricated = add_extra_material(
                reference
            )


        elif defect_type == "PARTICLE":

            fabricated = add_particle(
                reference
            )


        else:

            fabricated = add_deformation(
                reference
            )


    # --------------------------------------------------------
    # Save fabricated pattern
    # --------------------------------------------------------

    cv2.imwrite(
        fabricated_output,
        fabricated
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    labels.append(
        [
            index,
            reference_name,
            fabricated_name,
            defect_type
        ]
    )


    print(
        f"{index:03d} | "
        f"{reference_name} | "
        f"{defect_type}"
    )


# ============================================================
# SAVE LABEL CSV
# ============================================================

label_file = os.path.join(
    OUTPUT_LABELS,
    "defect_labels.csv"
)


with open(
    label_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "reference_id",
            "reference",
            "fabricated",
            "defect_type"
        ]
    )

    writer.writerows(
        labels
    )


# ============================================================
# SUMMARY
# ============================================================

counts = {}

for row in labels:

    defect = row[3]

    counts[defect] = (
        counts.get(defect, 0) + 1
    )


print()
print("=" * 70)
print("DAY 18 DATASET GENERATION COMPLETE")
print("=" * 70)

print()
print("Total patterns :", len(labels))

for defect, count in sorted(counts.items()):

    print(
        f"{defect:<20}: {count}"
    )

print()
print("Reference directory:")
print(OUTPUT_REFERENCE)

print()
print("Fabricated directory:")
print(OUTPUT_FABRICATED)

print()
print("Labels:")
print(label_file)

print()
print("=" * 70)
