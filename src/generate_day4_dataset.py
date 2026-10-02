import os
import cv2
import numpy as np
import random

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(
    BASE, "dataset", "day4_reference"
)

SEARCH_DIR = os.path.join(
    BASE, "dataset", "day4_search"
)

GT_DIR = os.path.join(
    BASE, "dataset", "day4_ground_truth"
)

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

REFERENCE_SIZE = 500
SEARCH_SIZE = 2000

NUM_SAMPLES = 20


def create_reference():

    image = np.ones(
        (REFERENCE_SIZE, REFERENCE_SIZE),
        dtype=np.uint8
    ) * 255

    # Semiconductor-style FinFET pattern
    for x in range(80, 450, 45):

        cv2.rectangle(
            image,
            (x, 80),
            (x + 12, 420),
            40,
            -1
        )

    # Horizontal gate structures
    for y in range(120, 420, 70):

        cv2.rectangle(
            image,
            (60, y),
            (450, y + 10),
            80,
            -1
        )

    # Add small features
    cv2.circle(
        image,
        (250, 250),
        35,
        20,
        3
    )

    return image


def transform_target(image):

    result = image.copy()

    # Slight rotation
    angle = random.uniform(-5, 5)

    matrix = cv2.getRotationMatrix2D(
        (REFERENCE_SIZE // 2, REFERENCE_SIZE // 2),
        angle,
        1.0
    )

    result = cv2.warpAffine(
        result,
        matrix,
        (REFERENCE_SIZE, REFERENCE_SIZE),
        borderValue=255
    )

    # Slight blur
    blur_size = random.choice([3, 5, 7])

    result = cv2.GaussianBlur(
        result,
        (blur_size, blur_size),
        0
    )

    # Contrast and brightness
    alpha = random.uniform(0.75, 1.15)
    beta = random.randint(-25, 25)

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=beta
    )

    # Gaussian noise
    noise = np.random.normal(
        0,
        random.uniform(3, 12),
        result.shape
    )

    result = result.astype(np.float32) + noise

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result


def create_search(target, x, y):

    # Background with semiconductor-like structures
    search = np.ones(
        (SEARCH_SIZE, SEARCH_SIZE),
        dtype=np.uint8
    ) * 255

    # Background patterns
    for _ in range(80):

        px = random.randint(
            0,
            SEARCH_SIZE - 30
        )

        py = random.randint(
            0,
            SEARCH_SIZE - 30
        )

        width = random.randint(10, 40)
        height = random.randint(10, 80)

        intensity = random.randint(
            150,
            240
        )

        cv2.rectangle(
            search,
            (px, py),
            (
                min(px + width, SEARCH_SIZE - 1),
                min(py + height, SEARCH_SIZE - 1)
            ),
            intensity,
            -1
        )

    # Place target
    h, w = target.shape

    search[
        y:y+h,
        x:x+w
    ] = target

    # Global noise
    noise = np.random.normal(
        0,
        2,
        search.shape
    )

    search = search.astype(
        np.float32
    ) + noise

    search = np.clip(
        search,
        0,
        255
    ).astype(np.uint8)

    return search


print("=" * 60)
print("DAY 4 - HARD DATASET GENERATION")
print("=" * 60)

for i in range(1, NUM_SAMPLES + 1):

    sample = f"{i:03d}"

    reference = create_reference()

    target = transform_target(
        reference
    )

    max_x = SEARCH_SIZE - REFERENCE_SIZE
    max_y = SEARCH_SIZE - REFERENCE_SIZE

    x = random.randint(
        0,
        max_x
    )

    y = random.randint(
        0,
        max_y
    )

    search = create_search(
        target,
        x,
        y
    )

    cv2.imwrite(
        os.path.join(
            REF_DIR,
            f"reference_{sample}.png"
        ),
        reference
    )

    cv2.imwrite(
        os.path.join(
            SEARCH_DIR,
            f"search_{sample}.png"
        ),
        search
    )

    with open(
        os.path.join(
            GT_DIR,
            f"ground_truth_{sample}.txt"
        ),
        "w"
    ) as f:

        f.write(
            f"target_x={x}\n"
        )

        f.write(
            f"target_y={y}\n"
        )

    print(
        f"Sample {i:02d}: "
        f"Target=({x},{y})"
    )


print("\n" + "=" * 60)
print("DAY 4 DATASET COMPLETE")
print("=" * 60)

print(
    f"Reference images : {NUM_SAMPLES}"
)

print(
    f"Search images    : {NUM_SAMPLES}"
)

print(
    f"Ground truths    : {NUM_SAMPLES}"
)
