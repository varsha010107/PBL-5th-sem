import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from heatmap_dataset import HeatmapDataset
from heatmap_model import SpatialLocalizer


BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TEST_DIR = os.path.join(
    BASE, "dataset", "day9_ai", "test"
)

MODEL_PATH = os.path.join(
    BASE, "models", "day10", "spatial_localizer.pth"
)

RESULT_DIR = os.path.join(
    BASE, "results", "day11"
)

os.makedirs(RESULT_DIR, exist_ok=True)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("DAY 11 - AI TESTING AND VISUAL LOCALIZATION")
print("=" * 70)

print("Device:", device)
print("Test directory:", TEST_DIR)
print("Model:", MODEL_PATH)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = HeatmapDataset(TEST_DIR)

print("Test samples:", len(dataset))


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = SpatialLocalizer().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


results = []


# ---------------------------------------------------------
# Test samples
# ---------------------------------------------------------

for index in range(len(dataset)):

    reference, search, heatmap, position = dataset[index]

    search_input = search.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(search_input)

    predicted_heatmap = prediction[0, 0].cpu()

    # Find maximum activation
    max_index = torch.argmax(predicted_heatmap)

    hy, hx = divmod(
        max_index.item(),
        predicted_heatmap.shape[1]
    )

    # Convert 64x64 heatmap coordinates
    # back to 256x256 search coordinates
    pred_x_256 = hx * 4 + 2
    pred_y_256 = hy * 4 + 2

    # Ground truth is already represented
    # in 256x256 coordinates by the dataset
    gt_x = position[0].item()
    gt_y = position[1].item()

    # Localization error
    error = np.sqrt(
        (pred_x_256 - gt_x) ** 2
        +
        (pred_y_256 - gt_y) ** 2
    )

    # Convert prediction back to original 2000x2000
    pred_x_original = pred_x_256 * 2000.0 / 256.0
    pred_y_original = pred_y_256 * 2000.0 / 256.0

    # Ground truth original coordinates
    gt_x_original = gt_x * 2000.0 / 256.0
    gt_y_original = gt_y * 2000.0 / 256.0


    results.append([
        index + 1,
        gt_x_original,
        gt_y_original,
        pred_x_original,
        pred_y_original,
        error
    ])


    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    search_image = search[0].numpy()

    plt.figure(figsize=(9, 8))

    plt.imshow(
        search_image,
        cmap="gray"
    )

    # Ground truth
    plt.scatter(
        gt_x,
        gt_y,
        marker="x",
        s=150,
        linewidths=3,
        label="Ground Truth"
    )

    # AI prediction
    plt.scatter(
        pred_x_256,
        pred_y_256,
        marker="o",
        s=120,
        facecolors="none",
        linewidths=3,
        label="AI Prediction"
    )

    plt.title(
        f"AI FinFET Localization - Sample {index + 1}\n"
        f"Error = {error:.2f} pixels"
    )

    plt.legend()

    plt.xlim(0, 256)
    plt.ylim(256, 0)

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.tight_layout()

    output_file = os.path.join(
        RESULT_DIR,
        f"prediction_{index + 1:03d}.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()


    if index < 10:

        print(
            f"{index + 1:03d} | "
            f"GT=({gt_x_original:.1f},{gt_y_original:.1f}) | "
            f"Pred=({pred_x_original:.1f},{pred_y_original:.1f}) | "
            f"Error={error:.2f}px"
        )


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

errors = [
    row[5]
    for row in results
]

mean_error = np.mean(errors)
median_error = np.median(errors)
max_error = np.max(errors)

threshold = 50.0

successful = sum(
    error <= threshold
    for error in errors
)

success_rate = (
    successful / len(errors)
) * 100


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

csv_path = os.path.join(
    RESULT_DIR,
    "day11_ai_results.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample",
        "ground_truth_x",
        "ground_truth_y",
        "predicted_x",
        "predicted_y",
        "error_pixels"
    ])

    writer.writerows(results)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("=" * 70)
print("DAY 11 OVERALL RESULTS")
print("=" * 70)

print("Samples              :", len(results))
print(f"Mean error           : {mean_error:.2f} pixels")
print(f"Median error         : {median_error:.2f} pixels")
print(f"Maximum error        : {max_error:.2f} pixels")
print("Success threshold    :", threshold, "pixels")
print(
    f"Successful samples   : "
    f"{successful}/{len(results)}"
)
print(f"Success rate         : {success_rate:.2f}%")

print()
print("Visual predictions:")
print(RESULT_DIR)

print()
print("CSV results:")
print(csv_path)

print("=" * 70)
print("DAY 11 TEST COMPLETE")
print("=" * 70)
