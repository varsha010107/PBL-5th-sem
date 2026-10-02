import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class HeatmapDataset(Dataset):

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

        for filename in sorted(
            os.listdir(self.label_dir)
        ):

            if not filename.endswith(".txt"):
                continue

            sample_id = (
                filename
                .replace("label_", "")
                .replace(".txt", "")
            )

            reference_file = os.path.join(
                self.reference_dir,
                f"reference_{sample_id}.png"
            )

            search_file = os.path.join(
                self.search_dir,
                f"search_{sample_id}.png"
            )

            label_file = os.path.join(
                self.label_dir,
                filename
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
                        label_file
                    )
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        (
            reference_file,
            search_file,
            label_file
        ) = self.samples[index]


        # =================================================
        # LOAD IMAGES
        # =================================================

        reference = cv2.imread(
            reference_file,
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            search_file,
            cv2.IMREAD_GRAYSCALE
        )


        if reference is None:

            raise RuntimeError(
                f"Unable to read {reference_file}"
            )

        if search is None:

            raise RuntimeError(
                f"Unable to read {search_file}"
            )


        # =================================================
        # RESIZE
        # =================================================

        reference = cv2.resize(
            reference,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )

        search = cv2.resize(
            search,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )


        # =================================================
        # NORMALIZE
        # =================================================

        reference = (
            reference.astype(np.float32)
            / 255.0
        )

        search = (
            search.astype(np.float32)
            / 255.0
        )


        # =================================================
        # CONVERT TO TORCH
        # =================================================

        reference = torch.tensor(
            reference,
            dtype=torch.float32
        ).unsqueeze(0)

        search = torch.tensor(
            search,
            dtype=torch.float32
        ).unsqueeze(0)


        # =================================================
        # READ ORIGINAL COORDINATES
        # =================================================

        with open(
            label_file,
            "r"
        ) as f:

            x, y = map(
                float,
                f.readline()
                .strip()
                .split(",")
            )


        # =================================================
        # ORIGINAL DATASET:
        #
        # SEARCH = 2000 × 2000
        #
        # AI SEARCH = 256 × 256
        # =================================================

        x = x * 256.0 / 2000.0
        y = y * 256.0 / 2000.0


        # =================================================
        # HEATMAP
        # =================================================

        heatmap = torch.zeros(
            (1, 64, 64),
            dtype=torch.float32
        )


        hx = int(
            np.clip(
                x / 4.0,
                0,
                63
            )
        )

        hy = int(
            np.clip(
                y / 4.0,
                0,
                63
            )
        )


        # =================================================
        # GAUSSIAN TARGET
        # =================================================

        sigma = 2.0

        for yy in range(
            max(0, hy - 8),
            min(64, hy + 9)
        ):

            for xx in range(
                max(0, hx - 8),
                min(64, hx + 9)
            ):

                distance = (
                    (xx - hx) ** 2
                    +
                    (yy - hy) ** 2
                )

                heatmap[
                    0,
                    yy,
                    xx
                ] = np.exp(
                    -distance /
                    (
                        2 *
                        sigma *
                        sigma
                    )
                )


        position = torch.tensor(
            [x, y],
            dtype=torch.float32
        )


        return (
            reference,
            search,
            heatmap,
            position
        )
