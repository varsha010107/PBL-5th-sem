import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from heatmap_dataset import HeatmapDataset
from heatmap_model import SpatialLocalizer


BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


TRAIN_DIR = os.path.join(
    BASE,
    "dataset",
    "day9_ai",
    "train"
)

VAL_DIR = os.path.join(
    BASE,
    "dataset",
    "day9_ai",
    "val"
)

MODEL_DIR = os.path.join(
    BASE,
    "models",
    "day10"
)

RESULT_DIR = os.path.join(
    BASE,
    "results",
    "day10"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("DAY 10 V2 - REFERENCE-CONDITIONED SPATIAL AI")
print("=" * 70)

print("Device:", device)


# =========================================================
# DATASET
# =========================================================

train_dataset = HeatmapDataset(
    TRAIN_DIR
)

val_dataset = HeatmapDataset(
    VAL_DIR
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# =========================================================
# MODEL
# =========================================================

model = SpatialLocalizer().to(
    device
)


criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


EPOCHS = 30


best_val_loss = float("inf")

history = []


# =========================================================
# TRAINING
# =========================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0


    for (
        reference,
        search,
        target_heatmap,
        position
    ) in train_loader:

        reference = reference.to(
            device
        )

        search = search.to(
            device
        )

        target_heatmap = (
            target_heatmap.to(device)
        )


        optimizer.zero_grad()


        prediction = model(
            reference,
            search
        )


        loss = criterion(
            prediction,
            target_heatmap
        )


        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
        )


    train_loss /= len(
        train_loader
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss = 0.0


    with torch.no_grad():

        for (
            reference,
            search,
            target_heatmap,
            position
        ) in val_loader:

            reference = reference.to(
                device
            )

            search = search.to(
                device
            )

            target_heatmap = (
                target_heatmap.to(device)
            )


            prediction = model(
                reference,
                search
            )


            loss = criterion(
                prediction,
                target_heatmap
            )


            val_loss += (
                loss.item()
            )


    val_loss /= len(
        val_loader
    )


    history.append(
        [
            epoch + 1,
            train_loss,
            val_loss
        ]
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_path = os.path.join(
            MODEL_DIR,
            "spatial_localizer_v2.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "  -> Best model saved"
        )


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_path = os.path.join(
    RESULT_DIR,
    "training_history_v2.csv"
)


with open(
    history_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "validation_loss"
        ]
    )

    writer.writerows(
        history
    )


print()
print("=" * 70)
print("SPATIAL AI V2 TRAINING COMPLETE")
print("=" * 70)

print(
    "Best model:",
    best_model_path
)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "History:",
    history_path
)

print("=" * 70)
