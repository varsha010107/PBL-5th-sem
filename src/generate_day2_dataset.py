import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

# ============================================================
# DAY 2
# Realistic Synthetic Semiconductor Dataset Generator
# ============================================================

IMAGE_SIZE = 1000
NUM_SAMPLES = 10

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(BASE_DIR, "dataset", "day2_reference")
SEARCH_DIR = os.path.join(BASE_DIR, "dataset", "day2_search")
GT_DIR = os.path.join(BASE_DIR, "dataset", "day2_ground_truth")
RESULT_DIR = os.path.join(BASE_DIR, "results", "day2")

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def create_finfet_pattern(size=1000):
    """
    Create a synthetic FinFET-like semiconductor pattern.
    """

    image = np.zeros((size, size), dtype=np.float32)

    # Background
    image[:] = 35

    # Random process variation
    fin_spacing = random.randint(45, 65)
    fin_width = random.randint(10, 16)

    # Vertical fins
    start_x = random.randint(80, 130)

    for x in range(start_x, size - 80, fin_spacing):

        width = random.randint(
            max(6, fin_width - 2),
            fin_width + 2
        )

        image[
            80:size - 80,
            x:x + width
        ] = random.randint(150, 210)

    # Gate structures
    gate_spacing = random.randint(180, 240)
    gate_width = random.randint(15, 25)

    start_y = random.randint(100, 160)

    for y in range(start_y, size - 100, gate_spacing):

        image[
            y:y + gate_width,
            60:size - 60
        ] = random.randint(190, 235)

    return image


def add_noise(image):

    # Gaussian sensor noise
    noise_strength = random.uniform(3, 12)

    noise = np.random.normal(
        0,
        noise_strength,
        image.shape
    )

    noisy = image + noise

    # Salt-and-pepper noise
    probability = random.uniform(0.001, 0.005)

    mask = np.random.random(image.shape)

    noisy[mask < probability] = 0
    noisy[mask > 1 - probability] = 255

    return np.clip(noisy, 0, 255)


def random_contrast(image):

    pil = Image.fromarray(
        np.uint8(np.clip(image, 0, 255))
    )

    factor = random.uniform(0.8, 1.25)

    enhancer = ImageEnhance.Contrast(pil)

    pil = enhancer.enhance(factor)

    return np.array(pil, dtype=np.float32)


def blur_image(image):

    pil = Image.fromarray(
        np.uint8(np.clip(image, 0, 255))
    )

    radius = random.uniform(0.3, 1.2)

    pil = pil.filter(
        ImageFilter.GaussianBlur(radius)
    )

    return np.array(pil, dtype=np.float32)


def create_search_image(reference_pattern, target_x, target_y):

    # Create larger physical region
    large_size = 10000

    large = np.zeros(
        (large_size, large_size),
        dtype=np.float32
    )

    large[:] = random.randint(25, 45)

    # Create repeated semiconductor structures
    for i in range(25):

        x = random.randint(0, large_size - 1000)
        y = random.randint(0, large_size - 1000)

        pattern_size = random.randint(700, 1200)

        pattern = create_finfet_pattern(pattern_size)

        h, w = pattern.shape

        if y + h >= large_size:
            h = large_size - y

        if x + w >= large_size:
            w = large_size - x

        large[
            y:y+h,
            x:x+w
        ] = pattern[:h, :w]

    # Convert target coordinates from 1000x1000
    # search coordinates to 10000x10000 physical space
    physical_x = target_x * 10
    physical_y = target_y * 10

    # Insert reference pattern at target position
    h, w = reference_pattern.shape

    y1 = physical_y
    x1 = physical_x

    y2 = min(y1 + h, large_size)
    x2 = min(x1 + w, large_size)

    large[
        y1:y2,
        x1:x2
    ] = reference_pattern[
        :y2-y1,
        :x2-x1
    ]

    # Crop 1000x1000 physical search region
    crop_size = 1000

    crop = large[
        y1:y1 + crop_size,
        x1:x1 + crop_size
    ]

    # Resize back to 1000x1000
    search = Image.fromarray(
        np.uint8(np.clip(crop, 0, 255))
    )

    search = search.resize(
        (1000, 1000),
        Image.Resampling.BILINEAR
    )

    return np.array(search, dtype=np.float32)


def save_image(array, path):

    image = Image.fromarray(
        np.uint8(np.clip(array, 0, 255))
    )

    image.save(path)


print("=" * 60)
print("DAY 2 - REALISTIC SYNTHETIC DATASET GENERATION")
print("=" * 60)

random.seed(42)
np.random.seed(42)

for i in range(1, NUM_SAMPLES + 1):

    print(f"\nGenerating sample {i}/{NUM_SAMPLES}")

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = create_finfet_pattern()

    reference = random_contrast(reference)

    reference = add_noise(reference)

    reference = blur_image(reference)

    # --------------------------------------------------------
    # Target position
    # --------------------------------------------------------

    target_x = random.randint(100, 800)
    target_y = random.randint(100, 800)

    # --------------------------------------------------------
    # Search image
    # --------------------------------------------------------

    search = create_search_image(
        reference,
        target_x,
        target_y
    )

    search = random_contrast(search)

    search = add_noise(search)

    search = blur_image(search)

    # --------------------------------------------------------
    # File names
    # --------------------------------------------------------

    ref_path = os.path.join(
        REF_DIR,
        f"reference_{i:03d}.png"
    )

    search_path = os.path.join(
        SEARCH_DIR,
        f"search_{i:03d}.png"
    )

    gt_path = os.path.join(
        GT_DIR,
        f"ground_truth_{i:03d}.txt"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(reference, ref_path)
    save_image(search, search_path)

    with open(gt_path, "w") as f:
        f.write(f"target_x={target_x}\n")
        f.write(f"target_y={target_y}\n")

    print(
        f"Target = ({target_x}, {target_y})"
    )

print("\n" + "=" * 60)
print("DAY 2 DATASET GENERATION COMPLETE")
print("=" * 60)

print(f"Reference images : {NUM_SAMPLES}")
print(f"Search images    : {NUM_SAMPLES}")
print(f"Ground truths    : {NUM_SAMPLES}")

print("\nOutput folders:")
print(REF_DIR)
print(SEARCH_DIR)
print(GT_DIR)
