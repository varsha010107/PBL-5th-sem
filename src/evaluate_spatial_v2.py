import os
import math
import torch
import numpy as np

from heatmap_dataset import HeatmapDataset
from heatmap_model import SpatialLocalizer


BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TEST_DIR = os.path.join(
    BASE, "dataset", "day9_ai", "test"
)

MODEL_PATH = os.path.join(
    BASE, "models", "day10",
    "spatial_localizer_v2.pth"
)


device = torch.device("cpu")

print("=" * 70)
print("SPATIAL AI V2 - TEST EVALUATION")
print("=" * 70)

dataset = HeatmapDataset(TEST_DIR)

print("Test samples:", len(dataset))
print("Model:", MODEL_PATH)
print("Device:", device)


model = SpatialLocalizer()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


errors = []


with torch.no_grad():

    for i in range(len(dataset)):

        reference, search, target_heatmap, position = dataset[i]

        reference = reference.unsqueeze(0)
        search = search.unsqueeze(0)

        prediction = model(
            reference,
            search
        )

        prediction = prediction[0, 0]

        # Find maximum heatmap location
        max_index = torch.argmax(
            prediction
        )

        hy, hx = divmod(
            max_index.item(),
            prediction.shape[1]
        )

        # Heatmap -> 256x256
        pred_x = hx * 4
        pred_y = hy * 4

        true_x = float(position[0])
        true_y = float(position[1])

        error = math.sqrt(
            (pred_x - true_x) ** 2 +
            (pred_y - true_y) ** 2
        )

        errors.append(error)


errors = np.array(errors)


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Mean error   : {errors.mean():.2f} pixels")
print(f"Median error : {np.median(errors):.2f} pixels")
print(f"Minimum error: {errors.min():.2f} pixels")
print(f"Maximum error: {errors.max():.2f} pixels")

print()
print("LOCALIZATION ACCURACY")
print("-" * 40)

for threshold in [5, 10, 25, 50, 100]:

    accuracy = (
        np.sum(errors <= threshold)
        / len(errors)
        * 100
    )

    print(
        f"Within {threshold:3d} px : "
        f"{accuracy:6.2f}%"
    )

print("=" * 70)
