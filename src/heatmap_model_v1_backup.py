import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLocalizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

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

        self.head = nn.Sequential(

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


    def forward(self, search):

        features = self.encoder(search)

        features = F.interpolate(
            features,
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        )

        heatmap = self.head(features)

        return heatmap
