import os
import csv

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ai_dataset import FinFETDataset
from localization_model import FinFETLocalizer


BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


TRAIN_DIR = os.path.join(
    BASE,
    "dataset",
    "ai",
    "train"
)

VAL_DIR = os.path.join(
    BASE,
    "dataset",
    "ai",
    "val"
)

MODEL_DIR = os.path.join(
    BASE,
    "models",
    "day6"
)

RESULT_DIR = os.path.join(
    BASE,
    "results",
    "day6"
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

print("=" * 65)
print("DAY 6 - AI MODEL TRAINING")
print("=" * 65)

print(
    "Device:",
    device
)


train_dataset = FinFETDataset(
    TRAIN_DIR
)

val_dataset = FinFETDataset(
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


model = FinFETLocalizer().to(
    device
)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


EPOCHS = 15

history = []


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0


    for reference, search, target in train_loader:

        reference = reference.to(
            device
        )

        search = search.to(
            device
        )

        target = target.to(
            device
        )


        optimizer.zero_grad()


        prediction = model(
            reference,
            search
        )


        loss = criterion(
            prediction,
            target
        )


        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
        )


    train_loss /= len(
        train_loader
    )


    model.eval()

    val_loss = 0.0


    with torch.no_grad():

        for reference, search, target in val_loader:

            reference = reference.to(
                device
            )

            search = search.to(
                device
            )

            target = target.to(
                device
            )


            prediction = model(
                reference,
                search
            )


            loss = criterion(
                prediction,
                target
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


model_path = os.path.join(
    MODEL_DIR,
    "finfet_localizer.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


csv_path = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

with open(
    csv_path,
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


print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(
    "Model saved:",
    model_path
)

print(
    "History saved:",
    csv_path
)

