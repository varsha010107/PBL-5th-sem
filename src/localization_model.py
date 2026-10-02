import torch
import torch.nn as nn


class FinFETLocalizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.reference_encoder = nn.Sequential(

            nn.Conv2d(
                1, 16, 3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, 3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.search_encoder = nn.Sequential(

            nn.Conv2d(
                1, 16, 3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, 3, padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.localization_head = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        reference,
        search
    ):

        reference_features = (
            self.reference_encoder(
                reference
            )
        )

        search_features = (
            self.search_encoder(
                search
            )
        )

        reference_features = (
            reference_features
            .view(
                reference_features.size(0),
                -1
            )
        )

        search_features = (
            search_features
            .view(
                search_features.size(0),
                -1
            )
        )

        features = torch.cat(
            [
                reference_features,
                search_features
            ],
            dim=1
        )

        coordinates = (
            self.localization_head(
                features
            )
        )

        return coordinates
