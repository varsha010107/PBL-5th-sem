import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

REF_DIR = os.path.join(BASE_DIR, "dataset", "day2_reference")
SEARCH_DIR = os.path.join(BASE_DIR, "dataset", "day2_search")
GT_DIR = os.path.join(BASE_DIR, "dataset", "day2_ground_truth")
RESULT_DIR = os.path.join(BASE_DIR, "results", "day2")

os.makedirs(RESULT_DIR, exist_ok=True)

sample = "001"

ref_path = os.path.join(
    REF_DIR, f"reference_{sample}.png"
)

search_path = os.path.join(
    SEARCH_DIR, f"search_{sample}.png"
)

gt_path = os.path.join(
    GT_DIR, f"ground_truth_{sample}.txt"
)

reference = np.array(
    Image.open(ref_path).convert("L")
)

search = np.array(
    Image.open(search_path).convert("L")
)

with open(gt_path) as f:
    lines = f.readlines()

target_x = int(lines[0].split("=")[1])
target_y = int(lines[1].split("=")[1])

print("=" * 60)
print("DAY 2 DATASET VERIFICATION")
print("=" * 60)

print(f"Reference size : {reference.shape}")
print(f"Search size    : {search.shape}")
print(f"Ground truth   : ({target_x}, {target_y})")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(reference, cmap="gray")
axes[0].set_title("Reference Pattern")
axes[0].axis("off")

axes[1].imshow(search, cmap="gray")

# Mark ground-truth position
axes[1].plot(
    target_x,
    target_y,
    "r+",
    markersize=20,
    markeredgewidth=3
)

axes[1].set_title(
    f"Search Image\nGround Truth = ({target_x}, {target_y})"
)

axes[1].axis("off")

output = os.path.join(
    RESULT_DIR,
    "day2_verification_001.png"
)

plt.tight_layout()
plt.savefig(output, dpi=150)
plt.close()

print(f"\nVerification image saved:")
print(output)

print("\nDAY 2 VERIFICATION COMPLETE")
