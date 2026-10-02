import torch
import matplotlib.pyplot as plt

from heatmap_dataset import HeatmapDataset

# Load Day 9 training dataset
dataset = HeatmapDataset("dataset/day9_ai/train")

# Get first sample
reference, search, heatmap, position = dataset[0]

# Find heatmap maximum
max_index = torch.argmax(heatmap[0])
hy, hx = divmod(max_index.item(), heatmap.shape[2])

# Convert heatmap coordinates back to 256x256
pred_x = hx * 4
pred_y = hy * 4

print("=" * 60)
print("DAY 10 - HEATMAP VERIFICATION")
print("=" * 60)

print(f"Dataset samples : {len(dataset)}")
print(f"Ground truth    : ({position[0]:.2f}, {position[1]:.2f})")
print(f"Heatmap maximum : ({hx}, {hy})")
print(f"Approx position : ({pred_x}, {pred_y})")

error = (
    (pred_x - position[0]) ** 2
    +
    (pred_y - position[1]) ** 2
) ** 0.5

print(f"Approx error    : {error:.2f} pixels")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(reference[0], cmap="gray")
axes[0].set_title("Reference Pattern")
axes[0].axis("off")

axes[1].imshow(search[0], cmap="gray")
axes[1].scatter(
    position[0],
    position[1],
    marker="x",
    s=100
)
axes[1].set_title("Search + Ground Truth")
axes[1].axis("off")

axes[2].imshow(heatmap[0], cmap="hot")
axes[2].scatter(
    hx,
    hy,
    marker="x",
    s=100
)
axes[2].set_title("Target Heatmap")
axes[2].axis("off")

plt.tight_layout()

output = "results/day10/heatmap_verification.png"
plt.savefig(output, dpi=150)
plt.close()

print()
print("Visualization saved:")
print(output)
print("=" * 60)
