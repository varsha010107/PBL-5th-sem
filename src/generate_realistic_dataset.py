import os
import cv2
import numpy as np
import random
import csv

# ============================================================
# CONFIGURATION
# ============================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(BASE, "dataset", "references")
SEARCH_DIR = os.path.join(BASE, "dataset", "search")
FAB_DIR = os.path.join(BASE, "dataset", "fabricated")

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(FAB_DIR, exist_ok=True)

random.seed(21)
np.random.seed(21)

REF_SIZE = 180
SEARCH_SIZE = 1800

# 5 x 5 = 25 patterns per search image
GRID = 5

CELL_SIZE = 300
MARGIN = 50


# ============================================================
# REALISTIC SEMICONDUCTOR PATTERN
# ============================================================

def create_reference(index):

    img = np.zeros(
        (REF_SIZE, REF_SIZE),
        dtype=np.uint8
    )

    # Dark silicon background
    img[:] = random.randint(18, 35)

    # Fine process texture
    noise = np.random.normal(
        0,
        3,
        img.shape
    ).astype(np.int16)

    img = np.clip(
        img.astype(np.int16) + noise,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Main chip structures
    # --------------------------------------------------------

    margin = 15

    # Outer rectangular structures
    cv2.rectangle(
        img,
        (margin, margin),
        (REF_SIZE - margin, REF_SIZE - margin),
        random.randint(150, 220),
        2
    )

    # Random transistor-like structures
    for _ in range(12):

        x = random.randint(20, 145)
        y = random.randint(20, 145)

        w = random.randint(8, 30)
        h = random.randint(8, 25)

        intensity = random.randint(
            120,
            220
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            intensity,
            1
        )

    # --------------------------------------------------------
    # Metal interconnects
    # --------------------------------------------------------

    for _ in range(10):

        x1 = random.randint(
            15,
            REF_SIZE - 20
        )

        y1 = random.randint(
            15,
            REF_SIZE - 20
        )

        if random.random() < 0.5:

            x2 = random.randint(
                x1,
                REF_SIZE - 10
            )

            y2 = y1

        else:

            x2 = x1

            y2 = random.randint(
                y1,
                REF_SIZE - 10
            )

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            random.randint(170, 235),
            random.choice([1, 2])
        )

    # --------------------------------------------------------
    # Via/contact points
    # --------------------------------------------------------

    for _ in range(25):

        x = random.randint(
            20,
            REF_SIZE - 20
        )

        y = random.randint(
            20,
            REF_SIZE - 20
        )

        cv2.circle(
            img,
            (x, y),
            random.choice([1, 2]),
            random.randint(190, 255),
            -1
        )

    # --------------------------------------------------------
    # Fine lithography-like grid
    # --------------------------------------------------------

    for x in range(25, 160, 20):

        cv2.line(
            img,
            (x, 10),
            (x, 170),
            random.randint(45, 80),
            1
        )

    for y in range(25, 160, 20):

        cv2.line(
            img,
            (10, y),
            (170, y),
            random.randint(45, 80),
            1
        )

    # Slight blur like microscope acquisition
    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0.4
    )

    return img


# ============================================================
# CREATE 100 REFERENCES
# ============================================================

print("\nGenerating 100 reference patterns...\n")

references = {}

for i in range(1, 101):

    ref = create_reference(i)

    path = os.path.join(
        REF_DIR,
        f"reference_{i:03d}.png"
    )

    cv2.imwrite(
        path,
        ref
    )

    references[i] = ref

    print(
        f"Reference {i:03d}/100 created"
    )


# ============================================================
# FABRICATION DEFECTS
# ============================================================

def add_defects(image):

    defective = image.copy()

    h, w = defective.shape

    defect_count = random.randint(
        4,
        12
    )

    for _ in range(defect_count):

        defect_type = random.choice([
            "particle",
            "scratch",
            "missing",
            "bridge"
        ])

        x = random.randint(
            20,
            w - 20
        )

        y = random.randint(
            20,
            h - 20
        )

        # ----------------------------------------------------
        # PARTICLE
        # ----------------------------------------------------

        if defect_type == "particle":

            radius = random.randint(
                2,
                6
            )

            cv2.circle(
                defective,
                (x, y),
                radius,
                random.randint(180, 255),
                -1
            )

        # ----------------------------------------------------
        # SCRATCH
        # ----------------------------------------------------

        elif defect_type == "scratch":

            length = random.randint(
                20,
                80
            )

            cv2.line(
                defective,
                (x, y),
                (
                    min(w - 1, x + length),
                    min(h - 1, y + random.randint(-5, 5))
                ),
                random.randint(80, 130),
                random.randint(1, 3)
            )

        # ----------------------------------------------------
        # MISSING STRUCTURE
        # ----------------------------------------------------

        elif defect_type == "missing":

            size = random.randint(
                5,
                20
            )

            cv2.rectangle(
                defective,
                (
                    max(0, x - size),
                    max(0, y - size)
                ),
                (
                    min(w - 1, x + size),
                    min(h - 1, y + size)
                ),
                random.randint(10, 25),
                -1
            )

        # ----------------------------------------------------
        # BRIDGE
        # ----------------------------------------------------

        elif defect_type == "bridge":

            cv2.line(
                defective,
                (x, y),
                (
                    min(w - 1, x + random.randint(15, 40)),
                    y
                ),
                random.randint(180, 230),
                random.randint(2, 4)
            )

    return defective


# ============================================================
# CREATE SEARCH IMAGE
# ============================================================

def create_search_image(
    pattern_ids,
    search_number
):

    canvas = np.zeros(
        (SEARCH_SIZE, SEARCH_SIZE),
        dtype=np.uint8
    )

    canvas[:] = random.randint(
        15,
        30
    )

    # Background wafer-like texture
    noise = np.random.normal(
        0,
        3,
        canvas.shape
    ).astype(np.int16)

    canvas = np.clip(
        canvas.astype(np.int16) + noise,
        0,
        255
    ).astype(np.uint8)

    metadata = []

    # Randomize order
    shuffled = pattern_ids.copy()

    random.shuffle(
        shuffled
    )

    for position, pattern_id in enumerate(
        shuffled
    ):

        row = position // GRID
        col = position % GRID

        x = (
            MARGIN
            + col * CELL_SIZE
            + 60
        )

        y = (
            MARGIN
            + row * CELL_SIZE
            + 60
        )

        reference = references[
            pattern_id
        ]

        # Slightly enlarge
        pattern = cv2.resize(
            reference,
            (REF_SIZE, REF_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # Put pattern into wide-field image
        # ----------------------------------------------------

        canvas[
            y:y + REF_SIZE,
            x:x + REF_SIZE
        ] = pattern

        metadata.append({

            "reference_id": pattern_id,

            "x": x,

            "y": y,

            "w": REF_SIZE,

            "h": REF_SIZE

        })

    # Add slight acquisition blur
    canvas = cv2.GaussianBlur(
        canvas,
        (3, 3),
        0.35
    )

    search_path = os.path.join(
        SEARCH_DIR,
        f"search_{search_number}.png"
    )

    cv2.imwrite(
        search_path,
        canvas
    )

    return canvas, metadata


# ============================================================
# GENERATE SEARCH + FABRICATED DATA
# ============================================================

all_metadata = []

print(
    "\nGenerating four wide-field search images...\n"
)

for search_number in range(1, 5):

    start = (
        (search_number - 1) * 25
        + 1
    )

    end = (
        search_number * 25
    )

    ids = list(
        range(start, end + 1)
    )

    search_image, metadata = create_search_image(
        ids,
        search_number
    )

    # --------------------------------------------------------
    # Fabricated image
    # --------------------------------------------------------

    fabricated = add_defects(
        search_image
    )

    fab_path = os.path.join(
        FAB_DIR,
        f"fabricated_{search_number}.png"
    )

    cv2.imwrite(
        fab_path,
        fabricated
    )

    # Save metadata
    for item in metadata:

        item["search_image"] = (
            f"search_{search_number}.png"
        )

        item["fabricated_image"] = (
            f"fabricated_{search_number}.png"
        )

        all_metadata.append(
            item
        )

    print(
        f"Search {search_number}: "
        f"references {start}-{end}"
    )


# ============================================================
# SAVE GROUND TRUTH CSV
# ============================================================

csv_path = os.path.join(
    BASE,
    "dataset",
    "ground_truth.csv"
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
            "search_image",
            "fabricated_image",
            "x",
            "y",
            "w",
            "h"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_metadata
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==============================================")
print("REALISTIC SEMICONDUCTOR DATASET COMPLETE")
print("==============================================")

print(
    f"References : {REF_DIR}"
)

print(
    f"Search     : {SEARCH_DIR}"
)

print(
    f"Fabricated : {FAB_DIR}"
)

print(
    f"Ground truth : {csv_path}"
)

print("==============================================")
