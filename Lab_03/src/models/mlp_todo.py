"""Multitask MLP strategy — E2."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """Fully connected multitask network that predicts gender and age.

    Flattens the input image to a 1-D vector and processes it through shared
    hidden layers before splitting into two task-specific heads. The enormous
    input dimension (image_size² × 3 = 150 528 for 224 px) means the first
    weight matrix dominates parameter count and makes the contrast with the
    CNN's parameter efficiency visible in the final comparison table.

    Args:
        image_size: Side length of the square input images in pixels.
        hidden_sizes: Number of units in each shared hidden layer.
        dropout: Dropout probability applied after every hidden activation.
    """

    def __init__(
        self,
        image_size: int = 224,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")

        self.dropout = dropout
        self.image_size = image_size
        input_dim = image_size * image_size * 3

        layers: list[nn.Module] = [nn.Flatten()]
        in_features = input_dim
        for out_features in hidden_sizes:
            layers += [
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_features = out_features

        self.shared = nn.Sequential(*layers)
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
