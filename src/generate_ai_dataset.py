import os
import cv2
import random
import numpy as np

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET = os.path.join(BASE, "dataset", "ai")

SEARCH_SIZE = 2000
REFERENCE_SIZE = 500

SPLITS = {
    "train": 800,
    "val": 100,
    "test": 100
}

random.seed(123)
np.random.seed(123)


def create_reference():

    image = np.ones(
        (REFERENCE_SIZE, REFERENCE_SIZE),
        dtype=np.uint8
    ) * 255

    # Fin-like vertical structures
    for x in range(80, 450, 45):

        cv2.rectangle(
            image,
            (x, 80),
            (x + 12, 420),
            40,
            -1
        )

    # Gate structures
    for y in range(120, 420, 70):

        cv2.rectangle(
            image,
            (60, y),
            (450, y + 10),
            80,
            -1
        )

    # Central feature
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

    # Rotation
    angle = random.uniform(-8, 8)

    matrix = cv2.getRotationMatrix2D(
        (REFERENCE_SIZE // 2,
         REFERENCE_SIZE // 2),
        angle,
        1.0
    )

    result = cv2.warpAffine(
        result,
        matrix,
        (REFERENCE_SIZE, REFERENCE_SIZE),
        borderValue=255
    )

    # Blur
    blur = random.choice([3, 5, 7])

    result = cv2.GaussianBlur(
        result,
        (blur, blur),
        0
    )

    # Brightness / contrast
    alpha = random.uniform(
        0.70,
        1.20
    )

    beta = random.randint(
        -30,
        30
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=beta
    )

    # Noise
    noise_level = random.uniform(
        2,
        15
    )

    noise = np.random.normal(
        0,
        noise_level,
        result.shape
    )

    result = (
        result.astype(np.float32)
        + noise
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result


def create_background():

    image = np.ones(
        (SEARCH_SIZE, SEARCH_SIZE),
        dtype=np.uint8
    ) * 255

    # Random semiconductor-like structures
    for _ in range(150):

        x = random.randint(
            0,
            SEARCH_SIZE - 40
        )

        y = random.randint(
            0,
            SEARCH_SIZE - 40
        )

        width = random.randint(
            5,
            50
        )

        height = random.randint(
            5,
            100
        )

        intensity = random.randint(
            150,
            245
        )

        cv2.rectangle(
            image,
            (x, y),
            (
                min(x + width,
                    SEARCH_SIZE - 1),

                min(y + height,
                    SEARCH_SIZE - 1)
            ),
            intensity,
            -1
        )

    return image


def create_search(target, x, y):

    search = create_background()

    h, w = target.shape

    search[
        y:y + h,
        x:x + w
    ] = target

    return search


print("=" * 65)
print("DAY 5 - AI/ML DATASET GENERATION")
print("=" * 65)

total = 0

for split, count in SPLITS.items():

    print(
        f"\nGenerating {split}: {count} samples"
    )

    ref_dir = os.path.join(
        DATASET,
        split,
        "reference"
    )

    search_dir = os.path.join(
        DATASET,
        split,
        "search"
    )

    label_dir = os.path.join(
        DATASET,
        split,
        "labels"
    )

    for i in range(count):

        sample_id = (
            f"{i + 1:04d}"
        )

        reference = create_reference()

        target = transform_target(
            reference
        )

        x = random.randint(
            0,
            SEARCH_SIZE - REFERENCE_SIZE
        )

        y = random.randint(
            0,
            SEARCH_SIZE - REFERENCE_SIZE
        )

        search = create_search(
            target,
            x,
            y
        )

        cv2.imwrite(
            os.path.join(
                ref_dir,
                f"reference_{sample_id}.png"
            ),
            reference
        )

        cv2.imwrite(
            os.path.join(
                search_dir,
                f"search_{sample_id}.png"
            ),
            search
        )

        with open(
            os.path.join(
                label_dir,
                f"label_{sample_id}.txt"
            ),
            "w"
        ) as f:

            f.write(
                f"{x},{y}\n"
            )

        total += 1

        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{count}"
            )

print("\n" + "=" * 65)
print("AI DATASET GENERATION COMPLETE")
print("=" * 65)

print(
    f"Total samples : {total}"
)

print(
    "Train         : 800"
)

print(
    "Validation    : 100"
)

print(
    "Test          : 100"
)
