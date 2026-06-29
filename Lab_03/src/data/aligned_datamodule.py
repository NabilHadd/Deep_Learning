"""DataModule that applies automatic face alignment before the standard transforms.

Used exclusively by E6 to simulate the preprocessing pipeline of the Streamlit
application (face detection → rotation alignment → crop) so that the model is
trained with the same domain distribution it will encounter at inference time.
"""

from __future__ import annotations

import dataclasses
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.data.dataset import UTKFaceDataset
from src.data.face_aligner import FaceAligner
from src.data.parser import UTKFaceRecord
from src.data.transforms import TransformFactory


class AlignedUTKFaceDataset(UTKFaceDataset):
    """UTKFaceDataset variant that aligns and crops each face before transforms."""

    def __init__(
        self,
        records: list[UTKFaceRecord],
        transform=None,
        aligner: FaceAligner | None = None,
        normalize_age: bool = False,
    ) -> None:
        super().__init__(records, transform, normalize_age=normalize_age)
        self.aligner = aligner or FaceAligner()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as f:
            image = f.convert("RGB")

        # Apply automatic alignment; fall back to original image if no face found.
        image, _ = self.aligner.align_and_crop(image)

        if self.transform is not None:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError("La transformacion debe convertir la imagen a torch.Tensor.")

        return (
            image,
            torch.tensor(record.gender, dtype=torch.long),
            torch.tensor(record.age, dtype=torch.float32),
        )


class AlignedUTKFaceDataModule(UTKFaceDataModule):
    """UTKFaceDataModule variant for raw UTKFace images processed with FaceAligner.

    Reads images from ``config.utkface_raw_dir`` instead of ``config.dataset_dir``
    and wraps every dataset instance with ``AlignedUTKFaceDataset`` so that the
    automatic alignment pipeline runs during data loading.

    This mirrors exactly what the Streamlit application does when a user uploads
    a photo, validating or rejecting the hypothesis that training on pre-aligned
    images causes a domain gap at inference time.
    """

    def __init__(
        self,
        config: AppConfig,
        use_augmentation: bool = True,
        normalize_age: bool = False,
    ) -> None:
        if config.utkface_raw_dir is None:
            raise ValueError(
                "UTKFACE_RAW_DIR no está configurado en .env. "
                "Define la ruta al dataset UTKFace original (no alineado) "
                "para poder ejecutar el experimento E6."
            )
        raw_config = dataclasses.replace(config, dataset_dir=config.utkface_raw_dir)
        super().__init__(raw_config, use_augmentation, normalize_age=normalize_age)
        self._aligner = FaceAligner()

    def setup(self) -> None:
        """Discover raw images, split and create AlignedUTKFaceDataset instances."""
        records = self._discover_records()
        train_records, val_records, test_records = self._split_records(records)

        train_transform = TransformFactory.training(
            self.config.image_size, use_augmentation=self.use_augmentation
        )
        eval_transform = TransformFactory.evaluation(self.config.image_size)

        self.train_dataset = AlignedUTKFaceDataset(
            train_records, transform=train_transform,
            aligner=self._aligner, normalize_age=self.normalize_age,
        )
        self.val_dataset = AlignedUTKFaceDataset(
            val_records, transform=eval_transform,
            aligner=self._aligner, normalize_age=self.normalize_age,
        )
        self.test_dataset = AlignedUTKFaceDataset(
            test_records, transform=eval_transform,
            aligner=self._aligner, normalize_age=self.normalize_age,
        )
        self._save_raw_split_manifest(train_records, val_records, test_records)

    def _save_raw_split_manifest(
        self,
        train_records: list[UTKFaceRecord],
        val_records: list[UTKFaceRecord],
        test_records: list[UTKFaceRecord],
    ) -> None:
        self.config.splits_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed": self.config.seed,
            "dataset_dir": str(self.config.dataset_dir),
            "train": [r.path.name for r in train_records],
            "validation": [r.path.name for r in val_records],
            "test": [r.path.name for r in test_records],
        }
        output_path = self.config.splits_dir / "utkface_raw_split.json"
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
