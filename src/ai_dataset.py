import os
import cv2
import torch

from torch.utils.data import Dataset


class FinFETDataset(Dataset):

    def __init__(self, base_dir):

        self.base_dir = base_dir

        self.reference_dir = os.path.join(
            base_dir,
            "reference"
        )

        self.search_dir = os.path.join(
            base_dir,
            "search"
        )

        self.label_dir = os.path.join(
            base_dir,
            "labels"
        )

        self.samples = []

        label_files = sorted(
            os.listdir(self.label_dir)
        )

        for label_file in label_files:

            if not label_file.endswith(".txt"):
                continue

            sample_id = label_file.replace(
                "label_",
                ""
            ).replace(
                ".txt",
                ""
            )

            reference_file = os.path.join(
                self.reference_dir,
                f"reference_{sample_id}.png"
            )

            search_file = os.path.join(
                self.search_dir,
                f"search_{sample_id}.png"
            )

            label_path = os.path.join(
                self.label_dir,
                label_file
            )

            if (
                os.path.exists(reference_file)
                and
                os.path.exists(search_file)
            ):

                self.samples.append(
                    (
                        reference_file,
                        search_file,
                        label_path
                    )
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        reference_file, search_file, label_file = (
            self.samples[index]
        )

        reference = cv2.imread(
            reference_file,
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            search_file,
            cv2.IMREAD_GRAYSCALE
        )

        # Resize for CPU-friendly training
        reference = cv2.resize(
            reference,
            (128, 128)
        )

        search = cv2.resize(
            search,
            (256, 256)
        )

        reference = (
            reference.astype("float32")
            / 255.0
        )

        search = (
            search.astype("float32")
            / 255.0
        )

        # Add channel dimension
        reference = torch.tensor(
            reference
        ).unsqueeze(0)

        search = torch.tensor(
            search
        ).unsqueeze(0)

        with open(label_file, "r") as f:

            line = f.readline().strip()

        x, y = map(
            float,
            line.split(",")
        )

        # Normalize coordinates to 0-1
        x = x / 1500.0
        y = y / 1500.0

        target = torch.tensor(
            [x, y],
            dtype=torch.float32
        )

        return reference, search, target
