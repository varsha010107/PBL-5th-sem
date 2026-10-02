import os
import cv2
import numpy as np
import random
import csv


# =========================================================
# CONFIGURATION
# =========================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REFERENCE_DIR = os.path.join(
    BASE,
    "dataset",
    "day9_ai",
    "test",
    "reference"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "dataset",
    "day9_ai",
    "test",
    "search_combined"
)

# Each reference is 500 x 500
REFERENCE_SIZE = 500

# 5 x 5 patterns
GRID_SIZE = 5

# Therefore wafer/search image = 2500 x 2500
IMAGE_SIZE = REFERENCE_SIZE * GRID_SIZE

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# FIND REFERENCES
# =========================================================

reference_files = sorted(
    [
        f
        for f in os.listdir(REFERENCE_DIR)
        if f.lower().endswith(".png")
        and f.startswith("reference_")
    ]
)

print()
print("=" * 60)
print("FINLOC AI SEARCH IMAGE GENERATOR")
print("=" * 60)

print(
    "Reference directory:",
    REFERENCE_DIR
)

print(
    "References found:",
    len(reference_files)
)

print(
    "Reference size:",
    REFERENCE_SIZE,
    "x",
    REFERENCE_SIZE
)

print(
    "Search size:",
    IMAGE_SIZE,
    "x",
    IMAGE_SIZE
)

print(
    "Patterns per search:",
    GRID_SIZE * GRID_SIZE
)


if len(reference_files) == 0:

    raise RuntimeError(
        "No reference images found!"
    )


# =========================================================
# CREATE FABRICATION-LIKE WAFER
# =========================================================

def create_wafer():

    wafer = np.full(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        238,
        dtype=np.uint8
    )


    # -----------------------------------------------------
    # Fine process noise
    # -----------------------------------------------------

    noise = np.random.normal(
        0,
        3.5,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    wafer = np.clip(
        wafer.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)


    # -----------------------------------------------------
    # Random fabrication defects
    # -----------------------------------------------------

    for _ in range(350):

        x = random.randint(
            0,
            IMAGE_SIZE - 1
        )

        y = random.randint(
            0,
            IMAGE_SIZE - 1
        )

        w = random.randint(
            3,
            25
        )

        h = random.randint(
            3,
            25
        )

        value = random.randint(
            190,
            230
        )


        cv2.rectangle(
            wafer,
            (x, y),
            (
                min(x + w, IMAGE_SIZE - 1),
                min(y + h, IMAGE_SIZE - 1)
            ),
            value,
            -1
        )


    # -----------------------------------------------------
    # Illumination gradient
    # -----------------------------------------------------

    yy, xx = np.mgrid[
        0:IMAGE_SIZE,
        0:IMAGE_SIZE
    ]

    cx = IMAGE_SIZE / 2
    cy = IMAGE_SIZE / 2

    distance = np.sqrt(
        ((xx - cx) / IMAGE_SIZE) ** 2
        +
        ((yy - cy) / IMAGE_SIZE) ** 2
    )


    illumination = (
        1.0 -
        0.08 * distance
    )


    wafer = (
        wafer.astype(np.float32)
        *
        illumination
    )


    wafer = np.clip(
        wafer,
        0,
        255
    ).astype(np.uint8)


    return wafer


# =========================================================
# PLACE REFERENCE
# =========================================================

def place_reference(
    wafer,
    reference,
    row,
    col
):

    x = col * REFERENCE_SIZE
    y = row * REFERENCE_SIZE


    h, w = reference.shape


    if (
        h != REFERENCE_SIZE
        or
        w != REFERENCE_SIZE
    ):

        raise RuntimeError(
            "Unexpected reference size: "
            + str(reference.shape)
        )


    wafer[
        y:y + REFERENCE_SIZE,
        x:x + REFERENCE_SIZE
    ] = reference


    return x, y


# =========================================================
# GENERATE SEARCH IMAGES
# =========================================================

ground_truth = []


group_size = GRID_SIZE * GRID_SIZE


total_groups = (
    len(reference_files)
    +
    group_size
    -
    1
) // group_size


for group_index in range(total_groups):

    start = (
        group_index
        *
        group_size
    )

    end = min(
        start + group_size,
        len(reference_files)
    )


    group = reference_files[
        start:end
    ]


    search_number = (
        group_index + 1
    )


    search_name = (
        f"search_{search_number:04d}.png"
    )


    print()
    print("-" * 60)
    print(
        search_name
    )


    wafer = create_wafer()


    for local_index, filename in enumerate(group):

        reference_path = os.path.join(
            REFERENCE_DIR,
            filename
        )


        reference = cv2.imread(
            reference_path,
            cv2.IMREAD_GRAYSCALE
        )


        if reference is None:

            print(
                "WARNING: Could not read",
                filename
            )

            continue


        row = local_index // GRID_SIZE
        col = local_index % GRID_SIZE


        x, y = place_reference(
            wafer,
            reference,
            row,
            col
        )


        reference_number = (
            start
            +
            local_index
            +
            1
        )


        ground_truth.append(
            [
                search_name,
                filename,
                reference_number,
                x,
                y,
                REFERENCE_SIZE,
                REFERENCE_SIZE
            ]
        )


        print(
            f"{filename:20s} "
            f"X={x:4d} "
            f"Y={y:4d}"
        )


    # -----------------------------------------------------
    # Save search image
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        search_name
    )


    cv2.imwrite(
        output_path,
        wafer
    )


    print(
        "Saved:",
        output_path
    )


# =========================================================
# SAVE GROUND TRUTH
# =========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "ground_truth.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)


    writer.writerow(
        [
            "search_image",
            "reference_image",
            "reference_number",
            "x",
            "y",
            "width",
            "height"
        ]
    )


    writer.writerows(
        ground_truth
    )


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)

print(
    "Search images created:",
    total_groups
)

print(
    "Total references:",
    len(reference_files)
)

print(
    "Ground truth:",
    csv_path
)

print()
print(
    "Generated groups:"
)

for i in range(total_groups):

    first = (
        i * group_size + 1
    )

    last = min(
        (i + 1) * group_size,
        len(reference_files)
    )

    print(
        f"search_{i + 1:04d}.png : "
        f"reference_{first:04d} "
        f"-> reference_{last:04d}"
    )
