import os
import csv
import cv2
import random


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

OUTPUT_DIR = os.path.join(
    BASE,
    "dataset",
    "day17_coordinate"
)

POSITIVE_DIR = os.path.join(
    OUTPUT_DIR,
    "positive"
)

NEGATIVE_DIR = os.path.join(
    OUTPUT_DIR,
    "negative"
)

os.makedirs(
    POSITIVE_DIR,
    exist_ok=True
)

os.makedirs(
    NEGATIVE_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

random.seed(17)

TILE_SIZE = 500

NUM_NEGATIVES = 2


# ============================================================
# LOAD REFERENCE FILES
# ============================================================

reference_files = sorted(
    [
        f
        for f in os.listdir(
            REFERENCE_DIR
        )
        if f.endswith(".png")
    ]
)

print("=" * 70)
print("DAY 17 - AI FEATURE MATCHING DATASET")
print("=" * 70)

print(
    f"Reference images : {len(reference_files)}"
)


# ============================================================
# PROCESS SEARCH IMAGES
# ============================================================

metadata = []

pair_id = 0

search_files = sorted(
    [
        f
        for f in os.listdir(
            SEARCH_DIR
        )
        if f.endswith(".png")
    ]
)

print(
    f"Search images    : {len(search_files)}"
)

print()


for search_file in search_files:

    search_path = os.path.join(
        SEARCH_DIR,
        search_file
    )

    search_image = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if search_image is None:

        print(
            "WARNING: Cannot read",
            search_file
        )

        continue


    label_file = (
        os.path.splitext(
            search_file
        )[0]
        + ".csv"
    )

    label_path = os.path.join(
        LABEL_DIR,
        label_file
    )

    with open(
        label_path,
        "r"
    ) as f:

        reader = csv.DictReader(f)

        labels = list(reader)


    # ========================================================
    # EACH TRUE PATTERN
    # ========================================================

    for row in labels:

        ref_id = int(
            row["reference"]
        )

        x = int(
            row["x"]
        )

        y = int(
            row["y"]
        )

        ref_name = (
            f"reference_{ref_id:04d}.png"
        )

        ref_path = os.path.join(
            REFERENCE_DIR,
            ref_name
        )

        reference = cv2.imread(
            ref_path,
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None:

            print(
                "WARNING: Missing",
                ref_name
            )

            continue


        # ====================================================
        # POSITIVE PAIR
        # ====================================================

        crop = search_image[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ]

        if crop.shape != (
            TILE_SIZE,
            TILE_SIZE
        ):

            continue


        pair_id += 1

        positive_name = (
            f"pair_{pair_id:05d}_positive.png"
        )

        positive_path = os.path.join(
            POSITIVE_DIR,
            positive_name
        )

        cv2.imwrite(
            positive_path,
            crop
        )

        metadata.append(
            [
                pair_id,
                ref_id,
                ref_name,
                positive_name,
                1,
                x,
                y,
                search_file
            ]
        )


        # ====================================================
        # NEGATIVE PAIRS
        # ====================================================

        other_ids = [
            i
            for i in range(
                1,
                101
            )
            if i != ref_id
        ]

        random.shuffle(
            other_ids
        )

        for negative_id in other_ids[
            :NUM_NEGATIVES
        ]:

            negative_ref_name = (
                f"reference_{negative_id:04d}.png"
            )

            negative_name = (
                f"pair_{pair_id:05d}_negative_"
                f"{negative_id:04d}.png"
            )

            negative_path = os.path.join(
                NEGATIVE_DIR,
                negative_name
            )

            cv2.imwrite(
                negative_path,
                crop
            )

            metadata.append(
                [
                    pair_id,
                    negative_id,
                    negative_ref_name,
                    negative_name,
                    0,
                    x,
                    y,
                    search_file
                ]
            )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.csv"
)

with open(
    metadata_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "pair_id",
            "reference_id",
            "reference",
            "sample",
            "label",
            "x",
            "y",
            "search_image"
        ]
    )

    writer.writerows(
        metadata
    )


# ============================================================
# SUMMARY
# ============================================================

positive_count = sum(
    1
    for row in metadata
    if row[4] == 1
)

negative_count = sum(
    1
    for row in metadata
    if row[4] == 0
)

print()
print("=" * 70)
print("DAY 17 DATASET GENERATION COMPLETE")
print("=" * 70)

print(
    f"Positive pairs : {positive_count}"
)

print(
    f"Negative pairs : {negative_count}"
)

print(
    f"Total pairs    : {len(metadata)}"
)

print()
print(
    "Positive directory:"
)

print(
    POSITIVE_DIR
)

print()
print(
    "Negative directory:"
)

print(
    NEGATIVE_DIR
)

print()
print(
    "Metadata:"
)

print(
    metadata_path
)

print("=" * 70)
