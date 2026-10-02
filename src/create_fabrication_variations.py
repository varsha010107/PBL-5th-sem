import os
import cv2
import csv
import random
import numpy as np


# =========================================================
# DAY 13 - FABRICATION VARIATION DATASET
# =========================================================

BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
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
    "day13_variation"
)


REFERENCE_OUT = os.path.join(
    OUTPUT_DIR,
    "reference"
)

SEARCH_OUT = os.path.join(
    OUTPUT_DIR,
    "search"
)

LABEL_OUT = os.path.join(
    OUTPUT_DIR,
    "labels"
)


os.makedirs(
    REFERENCE_OUT,
    exist_ok=True
)

os.makedirs(
    SEARCH_OUT,
    exist_ok=True
)

os.makedirs(
    LABEL_OUT,
    exist_ok=True
)


# =========================================================
# PARAMETERS
# =========================================================

SEARCH_SIZE = 2500

PATTERN_SIZE = 500

GRID_SIZE = 5

NUM_PATTERNS = 100


# Reproducible experiment

random.seed(42)
np.random.seed(42)


# =========================================================
# FABRICATION VARIATION
# =========================================================

def apply_fabrication_variation(pattern):

    result = pattern.copy()


    # -----------------------------------------------------
    # 1. Brightness variation
    # -----------------------------------------------------

    brightness = random.randint(
        -20,
        20
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=1.0,
        beta=brightness
    )


    # -----------------------------------------------------
    # 2. Contrast variation
    # -----------------------------------------------------

    contrast = random.uniform(
        0.85,
        1.15
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=contrast,
        beta=0
    )


    # -----------------------------------------------------
    # 3. Gaussian noise
    # -----------------------------------------------------

    noise_sigma = random.uniform(
        3.0,
        12.0
    )

    noise = np.random.normal(
        0,
        noise_sigma,
        result.shape
    )


    noisy = (
        result.astype(
            np.float32
        ) +
        noise
    )


    result = np.clip(
        noisy,
        0,
        255
    ).astype(
        np.uint8
    )


    # -----------------------------------------------------
    # 4. Slight blur
    # -----------------------------------------------------

    blur_probability = random.random()


    if blur_probability < 0.60:

        kernel_size = random.choice(
            [
                3,
                5
            ]
        )

        result = cv2.GaussianBlur(
            result,
            (
                kernel_size,
                kernel_size
            ),
            0
        )


    return result


# =========================================================
# CREATE LARGE SEARCH IMAGE
# =========================================================

def create_search_image(
    references,
    start_index,
    end_index,
    output_name
):

    # -----------------------------------------------------
    # Create large background
    # -----------------------------------------------------

    search = np.zeros(
        (
            SEARCH_SIZE,
            SEARCH_SIZE
        ),
        dtype=np.uint8
    )


    # Slight background variation

    background_value = random.randint(
        15,
        35
    )

    search[:, :] = background_value


    labels = []


    # -----------------------------------------------------
    # Place 25 patterns
    # -----------------------------------------------------

    for index in range(
        start_index,
        end_index
    ):

        reference_number = index + 1

        reference_path = references[
            index
        ]


        pattern = cv2.imread(
            reference_path,
            cv2.IMREAD_GRAYSCALE
        )


        if pattern is None:

            print(
                "WARNING: Cannot read",
                reference_path
            )

            continue


        # Make sure pattern is 500x500

        pattern = cv2.resize(
            pattern,
            (
                PATTERN_SIZE,
                PATTERN_SIZE
            )
        )


        # -------------------------------------------------
        # Apply fabrication variation
        # -------------------------------------------------

        varied_pattern = (
            apply_fabrication_variation(
                pattern
            )
        )


        # -------------------------------------------------
        # Grid position
        # -------------------------------------------------

        local_index = (
            index -
            start_index
        )


        row = (
            local_index //
            GRID_SIZE
        )

        col = (
            local_index %
            GRID_SIZE
        )


        x = (
            col *
            PATTERN_SIZE
        )

        y = (
            row *
            PATTERN_SIZE
        )


        # -------------------------------------------------
        # Insert pattern
        # -------------------------------------------------

        search[
            y:y + PATTERN_SIZE,
            x:x + PATTERN_SIZE
        ] = varied_pattern


        labels.append(
            [
                reference_number,
                x,
                y
            ]
        )


        print(
            f"reference_{reference_number:04d}.png"
            f"   X={x:4d} Y={y:4d}"
        )


    # -----------------------------------------------------
    # Add very small global sensor noise
    # -----------------------------------------------------

    global_noise = np.random.normal(
        0,
        2.0,
        search.shape
    )


    search = np.clip(
        search.astype(
            np.float32
        ) +
        global_noise,
        0,
        255
    ).astype(
        np.uint8
    )


    # -----------------------------------------------------
    # Save search image
    # -----------------------------------------------------

    output_path = os.path.join(
        SEARCH_OUT,
        output_name
    )


    cv2.imwrite(
        output_path,
        search
    )


    # -----------------------------------------------------
    # Save labels
    # -----------------------------------------------------

    label_path = os.path.join(
        LABEL_OUT,
        output_name.replace(
            ".png",
            ".csv"
        )
    )


    with open(
        label_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "reference",
                "x",
                "y"
            ]
        )

        writer.writerows(
            labels
        )


    print()
    print(
        "Saved:",
        output_path
    )

    print(
        "Labels:",
        label_path
    )


# =========================================================
# MAIN
# =========================================================

print("=" * 65)
print(
    "DAY 13 - FABRICATION VARIATION DATASET GENERATION"
)
print("=" * 65)


# ---------------------------------------------------------
# Find reference images
# ---------------------------------------------------------

references = []


for filename in os.listdir(
    REFERENCE_DIR
):

    if filename.lower().endswith(
        ".png"
    ):

        references.append(
            os.path.join(
                REFERENCE_DIR,
                filename
            )
        )


references.sort()


print(
    "Reference images found:",
    len(references)
)


if len(references) < NUM_PATTERNS:

    raise RuntimeError(
        f"Expected {NUM_PATTERNS} reference images, "
        f"found {len(references)}"
    )


# ---------------------------------------------------------
# Copy clean references
# ---------------------------------------------------------

for index in range(
    NUM_PATTERNS
):

    source = references[
        index
    ]

    destination = os.path.join(
        REFERENCE_OUT,
        f"reference_{index + 1:04d}.png"
    )


    pattern = cv2.imread(
        source,
        cv2.IMREAD_GRAYSCALE
    )


    if pattern is None:

        raise RuntimeError(
            f"Cannot read {source}"
        )


    pattern = cv2.resize(
        pattern,
        (
            PATTERN_SIZE,
            PATTERN_SIZE
        )
    )


    cv2.imwrite(
        destination,
        pattern
    )


# ---------------------------------------------------------
# Generate four large search images
# ---------------------------------------------------------

groups = [
    (
        0,
        25,
        "search_0001.png"
    ),
    (
        25,
        50,
        "search_0002.png"
    ),
    (
        50,
        75,
        "search_0003.png"
    ),
    (
        75,
        100,
        "search_0004.png"
    )
]


for start, end, name in groups:

    print()
    print(
        "-" * 65
    )

    print(
        f"Generating {name}"
    )

    print(
        f"References {start + 1} -> {end}"
    )

    print(
        "-" * 65
    )


    create_search_image(
        references,
        start,
        end,
        name
    )


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 65)
print(
    "DAY 13 GENERATION COMPLETE"
)
print("=" * 65)

print(
    "Output:",
    OUTPUT_DIR
)

print(
    "Search images:",
    len(groups)
)

print(
    "Total references:",
    NUM_PATTERNS
)

print()
print(
    "Variation types:"
)

print(
    "  - Brightness variation"
)

print(
    "  - Contrast variation"
)

print(
    "  - Gaussian noise"
)

print(
    "  - Slight Gaussian blur"
)

print(
    "  - Global sensor noise"
)

print("=" * 65)
