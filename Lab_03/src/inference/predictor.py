"""Load any multitask checkpoint and run PyTorch inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from src.data.transforms import TransformFactory
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet


@dataclass(frozen=True)
class Prediction:
    """Human-readable model output for one detected face."""

    gender_index: int
    gender_label: str
    gender_confidence: float
    estimated_age: float


def _build_model(model_name: str, model_kwargs: dict) -> nn.Module:
    if model_name == "cnn":
        return MultiTaskCNN(**model_kwargs)
    if model_name == "mlp":
        return MultiTaskMLP(**model_kwargs)
    if model_name in ("resnet_frozen", "resnet_finetuning"):
        return MultiTaskResNet(**model_kwargs)
    raise ValueError(
        f"Tipo de modelo desconocido en el checkpoint: {model_name!r}. "
        "Tipos soportados: cnn, mlp, resnet_frozen, resnet_finetuning."
    )


class CNNPredictor:
    """Apply exactly the same deterministic preprocessing used during testing.

    Supports any checkpoint produced by the experiment runner: CNN, MLP and
    ResNet variants are all handled transparently via the model_name metadata
    field saved in the checkpoint.
    """

    GENDER_LABELS = {0: "Masculino", 1: "Femenino"}

    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        device: torch.device,
        age_scale: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.image_size = image_size
        self.device = device
        self.age_scale = age_scale
        self.transform = TransformFactory.inference(image_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> "CNNPredictor":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint {path}. "
                "Entrena el modelo antes de usar Streamlit."
            )

        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        model_name = checkpoint.get("model_name", "cnn")
        model_kwargs = checkpoint.get("model_kwargs", {})
        model = _build_model(model_name, model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])

        image_size = int(checkpoint.get("image_size", 224))
        normalize_age = bool(checkpoint.get("normalize_age", False))
        age_scale = 100.0 if normalize_age else 1.0

        return cls(model=model, image_size=image_size, device=device, age_scale=age_scale)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        gender_logits, age_prediction = self.model(image_tensor)
        probabilities = torch.softmax(gender_logits, dim=1)
        confidence, gender_index = probabilities.max(dim=1)
        index = int(gender_index.item())
        estimated_age = float(age_prediction.item()) * self.age_scale
        return Prediction(
            gender_index=index,
            gender_label=self.GENDER_LABELS.get(index, str(index)),
            gender_confidence=float(confidence.item()),
            estimated_age=estimated_age,
        )
