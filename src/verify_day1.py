from PIL import Image
import os

# ============================================================
# Applied Materials PBL
# Day 1 - Geometry Verification
# ============================================================

IMAGE_SIZE = 1000

REFERENCE_PIXEL_SIZE_NM = 1
SEARCH_PIXEL_SIZE_NM = 10

REFERENCE_PHYSICAL_SIZE_NM = IMAGE_SIZE * REFERENCE_PIXEL_SIZE_NM
SEARCH_PHYSICAL_SIZE_NM = IMAGE_SIZE * SEARCH_PIXEL_SIZE_NM

TARGET_X = 650
TARGET_Y = 400

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REFERENCE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "reference",
    "reference_001.png"
)

SEARCH_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "search",
    "search_001.png"
)

GROUND_TRUTH_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "ground_truth",
    "ground_truth_001.txt"
)


print("=" * 60)
print("DAY 1 - 10X GEOMETRY VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check image dimensions
# ------------------------------------------------------------

reference_size = Image.open(REFERENCE_PATH).size
search_size = Image.open(SEARCH_PATH).size

print("\n[1] IMAGE DIMENSIONS")

print("Reference:", reference_size)
print("Search   :", search_size)

assert reference_size == (1000, 1000)
assert search_size == (1000, 1000)

print("PASS")


# ------------------------------------------------------------
# 2. Check physical dimensions
# ------------------------------------------------------------

print("\n[2] PHYSICAL SCALE")

print(
    "Reference pixel size:",
    REFERENCE_PIXEL_SIZE_NM,
    "nm/pixel"
)

print(
    "Search pixel size   :",
    SEARCH_PIXEL_SIZE_NM,
    "nm/pixel"
)

print(
    "Reference physical field:",
    REFERENCE_PHYSICAL_SIZE_NM,
    "nm x",
    REFERENCE_PHYSICAL_SIZE_NM,
    "nm"
)

print(
    "Search physical field:",
    SEARCH_PHYSICAL_SIZE_NM,
    "nm x",
    SEARCH_PHYSICAL_SIZE_NM,
    "nm"
)

scale_ratio = SEARCH_PIXEL_SIZE_NM / REFERENCE_PIXEL_SIZE_NM

print("Physical scale ratio:", scale_ratio, "x")

assert scale_ratio == 10

print("PASS")


# ------------------------------------------------------------
# 3. Check target coordinates
# ------------------------------------------------------------

print("\n[3] GROUND TRUTH")

print("Target X:", TARGET_X)
print("Target Y:", TARGET_Y)

assert 0 <= TARGET_X < IMAGE_SIZE
assert 0 <= TARGET_Y < IMAGE_SIZE

print("Target coordinate is valid")
print("PASS")


# ------------------------------------------------------------
# 4. Convert target coordinate into physical coordinates
# ------------------------------------------------------------

print("\n[4] PHYSICAL COORDINATES")

physical_x_nm = TARGET_X * SEARCH_PIXEL_SIZE_NM
physical_y_nm = TARGET_Y * SEARCH_PIXEL_SIZE_NM

print(
    "Target physical X:",
    physical_x_nm,
    "nm"
)

print(
    "Target physical Y:",
    physical_y_nm,
    "nm"
)


# ------------------------------------------------------------
# 5. Calculate theoretical target size
# ------------------------------------------------------------

print("\n[5] TARGET SCALE")

reference_physical_width = (
    IMAGE_SIZE * REFERENCE_PIXEL_SIZE_NM
)

target_width_in_search_pixels = (
    reference_physical_width / SEARCH_PIXEL_SIZE_NM
)

print(
    "Reference physical width:",
    reference_physical_width,
    "nm"
)

print(
    "Expected target width in search:",
    target_width_in_search_pixels,
    "pixels"
)

assert target_width_in_search_pixels == 100

print("PASS")


# ------------------------------------------------------------
# 6. Read ground-truth file
# ------------------------------------------------------------

print("\n[6] GROUND-TRUTH FILE")

with open(GROUND_TRUTH_PATH, "r") as file:
    content = file.read()

print(content)

assert "target_x=650" in content
assert "target_y=400" in content

print("PASS")


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DAY 1 GEOMETRY VERIFICATION COMPLETE")
print("=" * 60)

print("\nExpected relationship:")
print("Reference : 1 nm/pixel")
print("Search    : 10 nm/pixel")
print("Ratio     : 10x")

print("\nExpected target scale:")
print("Reference pattern: 1000 pixels")
print("Search equivalent: 100 pixels")

print("\nAll verification checks passed!")
