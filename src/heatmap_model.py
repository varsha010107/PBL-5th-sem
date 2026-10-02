import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLocalizer(nn.Module):

    def __init__(self):

        super().__init__()

        # =================================================
        # REFERENCE ENCODER
        # =================================================

        self.reference_encoder = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # =================================================
        # SEARCH ENCODER
        # =================================================

        self.search_encoder = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # =================================================
        # MATCHING HEAD
        # =================================================

        self.matching_head = nn.Sequential(

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )


    def forward(
        self,
        reference,
        search
    ):

        # =================================================
        # ENCODE REFERENCE
        # =================================================

        ref_features = self.reference_encoder(
            reference
        )


        # =================================================
        # ENCODE SEARCH
        # =================================================

        search_features = self.search_encoder(
            search
        )


        # =================================================
        # GLOBAL REFERENCE REPRESENTATION
        # =================================================

        ref_features = F.adaptive_avg_pool2d(
            ref_features,
            (1, 1)
        )


        # Expand reference feature across search map

        ref_features = ref_features.expand(
            -1,
            -1,
            search_features.shape[2],
            search_features.shape[3]
        )


        # =================================================
        # COMBINE REFERENCE + SEARCH
        # =================================================

        combined = torch.cat(
            [
                search_features,
                ref_features
            ],
            dim=1
        )


        # =================================================
        # PREDICT HEATMAP
        # =================================================

        heatmap = self.matching_head(
            combined
        )


        # =================================================
        # STANDARD OUTPUT SIZE
        # =================================================

        heatmap = F.interpolate(
            heatmap,
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        )


        return heatmap
