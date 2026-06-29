"""ResNet18 transfer learning strategies — E4 (frozen backbone) and E5 (fine-tuning)."""

from __future__ import annotations

import torch
import torchvision.models as tv_models
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """ResNet18 backbone with separate gender classification and age regression heads.

    The backbone is always initialised with ImageNet weights. The ``unfreeze_blocks``
    argument controls how many of the four residual layers (counted from the last)
    have their parameters set to ``requires_grad=True``:

    - 0  → all backbone frozen         (E4)
    - 1  → only ``layer4`` trainable   (E5 base)
    - 2  → ``layer3`` + ``layer4``     (E5 ablation: unfreeze_more)
    - 4  → full backbone fine-tuning   (E5 ablation: full)

    The heads always have ``requires_grad=True`` regardless of this setting.

    ResNet18 children order when iterated:
        0: conv1  1: bn1  2: relu  3: maxpool
        4: layer1  5: layer2  6: layer3  7: layer4
        8: avgpool  9: fc  (fc is dropped)

    Output of backbone (after avgpool): [B, 512, 1, 1].
    """

    _RESNET_LAYER_INDICES = [4, 5, 6, 7]

    def __init__(self, unfreeze_blocks: int = 0, dropout: float = 0.4) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")
        if not 0 <= unfreeze_blocks <= 4:
            raise ValueError("unfreeze_blocks debe estar entre 0 y 4.")

        self.unfreeze_blocks = unfreeze_blocks
        self.dropout = dropout

        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_blocks > 0:
            for idx in self._RESNET_LAYER_INDICES[-unfreeze_blocks:]:
                for param in self.backbone[idx].parameters():
                    param.requires_grad = True

        self.gender_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 2),
        )
        self.age_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)        # [B, 512, 1, 1]
        gender_logits = self.gender_head(features)
        age_predictions = self.age_head(features).squeeze(1)
        return gender_logits, age_predictions
