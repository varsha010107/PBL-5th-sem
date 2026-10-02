import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ============================================================
# Applied Materials PBL
# Day 1 - Synthetic FinFET Image Generator
# ============================================================

IMAGE_SIZE = 1000

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REFERENCE_DIR = os.path.join(BASE_DIR, "dataset", "reference")
SEARCH_DIR = os.path.join(BASE_DIR, "dataset", "search")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "dataset", "ground_truth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_finfet_pattern(size):
    """
    Generate a simple synthetic FinFET-style pattern.

    Vertical bright lines represent fins.
    Horizontal bright bars represent gates.
    """

    image = np.full((size, size), 40, dtype=np.uint8)

    # Vertical fins
    fin_spacing = 35
    fin_width = 8

    for x in range(80, size - 80, fin_spacing):
        image[:, x:x + fin_width] = 200

    # Horizontal gates
    gate_height = 12

    gate_y1 = size // 3
    gate_y2 = 2 * size // 3

    image[
        gate_y1:gate_y1 + gate_height,
        50:size - 50
    ] = 240

    image[
        gate_y2:gate_y2 + gate_height,
        50:size - 50
    ] = 240

    return image


def main():

    print("=" * 60)
    print("Applied Materials PBL - Day 1")
    print("Synthetic FinFET Image Generator")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Generate reference image
    # --------------------------------------------------------

    print("\n[1/5] Generating reference image...")

    reference = generate_finfet_pattern(IMAGE_SIZE)

    reference_path = os.path.join(
        REFERENCE_DIR,
        "reference_001.png"
    )

    Image.fromarray(reference).save(reference_path)

    print("Reference:", reference_path)
    print("Size:", reference.shape)

    # --------------------------------------------------------
    # 2. Generate large continuous physical region
    # --------------------------------------------------------

    print("\n[2/5] Generating large search region...")

    large_size = IMAGE_SIZE * 10

    large_image = generate_finfet_pattern(large_size)

    print("Large physical image:", large_image.shape)

    # --------------------------------------------------------
    # 3. Select target location
    # --------------------------------------------------------

    target_x = 650
    target_y = 400

    print("\n[3/5] Target location:")
    print("X =", target_x)
    print("Y =", target_y)

    # --------------------------------------------------------
    # 4. Create wide-search image
    # --------------------------------------------------------

    print("\n[4/5] Creating 10x wide-search image...")

    # Physical coordinates in the 10x larger image
    physical_x = target_x * 10
    physical_y = target_y * 10

    # Crop a 1000 x 1000 physical region
    crop_size = IMAGE_SIZE

    x1 = physical_x
    y1 = physical_y
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Keep crop inside large image
    x1 = min(x1, large_size - crop_size)
    y1 = min(y1, large_size - crop_size)

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = large_image[y1:y2, x1:x2]

    # Downsample to 1000 x 1000 pixels
    search = Image.fromarray(crop).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    search = np.array(search)

    search_path = os.path.join(
        SEARCH_DIR,
        "search_001.png"
    )

    Image.fromarray(search).save(search_path)

    print("Search:", search_path)
    print("Size:", search.shape)

    # --------------------------------------------------------
    # 5. Save ground truth
    # --------------------------------------------------------

    print("\n[5/5] Saving ground truth...")

    ground_truth_path = os.path.join(
        GROUND_TRUTH_DIR,
        "ground_truth_001.txt"
    )

    with open(ground_truth_path, "w") as file:
        file.write(f"target_x={target_x}\n")
        file.write(f"target_y={target_y}\n")

    print("Ground truth:", ground_truth_path)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nCreating visualization...")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(reference, cmap="gray")
    plt.title("Reference Image - 100x")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(search, cmap="gray")
    plt.title("Wide Search Image - 10x")
    plt.axis("off")

    plt.tight_layout()

    visualization_path = os.path.join(
        RESULTS_DIR,
        "day1_first_pair.png"
    )

    plt.savefig(
        visualization_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\n" + "=" * 60)
    print("DAY 1 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Reference : {reference_path}")
    print(f"Search    : {search_path}")
    print(f"Ground truth: ({target_x}, {target_y})")
    print(f"Visualization: {visualization_path}")


if __name__ == "__main__":
    main()
