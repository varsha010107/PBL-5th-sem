import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ============================================================
# DAY 3
# Corrected Synthetic Semiconductor Localization Dataset
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(
    BASE_DIR, "dataset", "corrected", "reference"
)

SEARCH_DIR = os.path.join(
    BASE_DIR, "dataset", "corrected", "search"
)

GT_DIR = os.path.join(
    BASE_DIR, "dataset", "corrected", "ground_truth"
)

NUM_SAMPLES = 10

REFERENCE_SIZE = 500
SEARCH_SIZE = 2000


os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)


# ============================================================
# Create semiconductor pattern
# ============================================================

def create_pattern(size):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    # Background
    image[:] = 35

    # Fin parameters
    fin_spacing = random.randint(35, 55)
    fin_width = random.randint(8, 14)

    start_x = random.randint(30, 60)

    # Vertical fins
    for x in range(
        start_x,
        size - 30,
        fin_spacing
    ):

        width = random.randint(
            max(5, fin_width - 2),
            fin_width + 2
        )

        intensity = random.randint(
            150,
            210
        )

        image[
            30:size - 30,
            x:x + width
        ] = intensity

    # Gate structures
    gate_spacing = random.randint(
        120,
        180
    )

    gate_width = random.randint(
        12,
        20
    )

    start_y = random.randint(
        50,
        100
    )

    for y in range(
        start_y,
        size - 50,
        gate_spacing
    ):

        intensity = random.randint(
            190,
            235
        )

        image[
            y:y + gate_width,
            30:size - 30
        ] = intensity

    return image


# ============================================================
# Add realistic variations
# ============================================================

def add_variations(image):

    # Contrast
    pil = Image.fromarray(
        np.uint8(
            np.clip(image, 0, 255)
        )
    )

    contrast = random.uniform(
        0.85,
        1.20
    )

    pil = ImageEnhance.Contrast(
        pil
    ).enhance(contrast)

    # Blur
    blur_radius = random.uniform(
        0.2,
        0.8
    )

    pil = pil.filter(
        ImageFilter.GaussianBlur(
            blur_radius
        )
    )

    image = np.array(
        pil,
        dtype=np.float32
    )

    # Gaussian noise
    noise_level = random.uniform(
        2,
        8
    )

    noise = np.random.normal(
        0,
        noise_level,
        image.shape
    )

    image = image + noise

    return np.clip(
        image,
        0,
        255
    )


# ============================================================
# Create search field
# ============================================================

def create_search_field(
    reference,
    target_x,
    target_y
):

    search = np.zeros(
        (SEARCH_SIZE, SEARCH_SIZE),
        dtype=np.float32
    )

    # Background
    search[:] = random.randint(
        25,
        45
    )

    # --------------------------------------------------------
    # Add distractor patterns
    # --------------------------------------------------------

    for _ in range(15):

        size = random.randint(
            250,
            500
        )

        pattern = create_pattern(
            size
        )

        x = random.randint(
            0,
            SEARCH_SIZE - size
        )

        y = random.randint(
            0,
            SEARCH_SIZE - size
        )

        search[
            y:y + size,
            x:x + size
        ] = pattern

    # --------------------------------------------------------
    # Insert actual reference target
    # --------------------------------------------------------

    h, w = reference.shape

    search[
        target_y:target_y + h,
        target_x:target_x + w
    ] = reference

    return search


# ============================================================
# Save image
# ============================================================

def save_image(
    image,
    path
):

    image = Image.fromarray(
        np.uint8(
            np.clip(
                image,
                0,
                255
            )
        )
    )

    image.save(path)


# ============================================================
# Main
# ============================================================

print("=" * 60)
print("CORRECTED DATASET GENERATION")
print("=" * 60)

random.seed(123)
np.random.seed(123)


for i in range(
    1,
    NUM_SAMPLES + 1
):

    print(
        f"\nGenerating sample "
        f"{i}/{NUM_SAMPLES}"
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = create_pattern(
        REFERENCE_SIZE
    )

    reference = add_variations(
        reference
    )

    # --------------------------------------------------------
    # Target location
    #
    # Keep entire 500x500 reference inside
    # 2000x2000 search image.
    # --------------------------------------------------------

    target_x = random.randint(
        0,
        SEARCH_SIZE - REFERENCE_SIZE
    )

    target_y = random.randint(
        0,
        SEARCH_SIZE - REFERENCE_SIZE
    )

    # --------------------------------------------------------
    # Search image
    # --------------------------------------------------------

    search = create_search_field(
        reference,
        target_x,
        target_y
    )

    search = add_variations(
        search
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    reference_path = os.path.join(
        REF_DIR,
        f"reference_{i:03d}.png"
    )

    search_path = os.path.join(
        SEARCH_DIR,
        f"search_{i:03d}.png"
    )

    ground_truth_path = os.path.join(
        GT_DIR,
        f"ground_truth_{i:03d}.txt"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        reference,
        reference_path
    )

    save_image(
        search,
        search_path
    )

    with open(
        ground_truth_path,
        "w"
    ) as f:

        f.write(
            f"target_x={target_x}\n"
        )

        f.write(
            f"target_y={target_y}\n"
        )

    print(
        f"Target = "
        f"({target_x}, {target_y})"
    )


print("\n" + "=" * 60)
print("CORRECTED DATASET COMPLETE")
print("=" * 60)

print(
    f"Reference size : "
    f"{REFERENCE_SIZE} x {REFERENCE_SIZE}"
)

print(
    f"Search size    : "
    f"{SEARCH_SIZE} x {SEARCH_SIZE}"
)

print(
    f"Samples        : "
    f"{NUM_SAMPLES}"
)
