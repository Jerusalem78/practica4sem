from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from dataset.yolo_detection_dataset import _parse_label_file, _resolve_image_path


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def get_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=image_size, width=image_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )


def get_valid_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0,
            min_visibility=0,
            label_fields=["labels"],
        ),
    )


@dataclass(frozen=True)
class KittiEfficientDetAdaptor:
    data_root: Path
    split: str
    class_names: Dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        if self.class_names is None:
            object.__setattr__(self, "class_names", dict(CLASS_NAMES))

    @property
    def images_dir(self) -> Path:
        return self.data_root / self.split / "images"

    @property
    def labels_dir(self) -> Path:
        return self.data_root / self.split / "labels"

    def __len__(self) -> int:
        return len(self.image_stems)

    @property
    def image_stems(self) -> List[str]:
        stems = []
        for path in sorted(self.images_dir.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                stems.append(path.stem)
        return stems

    def get_image_and_labels_by_idx(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        stem = self.image_stems[index]
        image_path = _resolve_image_path(self.images_dir, stem)
        label_path = self.labels_dir / f"{stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        pascal_bboxes: List[List[float]] = []
        class_labels: List[float] = []
        for box in _parse_label_file(label_path):
            if box.class_id not in self.class_names:
                continue
            x1 = (box.x_center - box.width / 2.0) * width
            y1 = (box.y_center - box.height / 2.0) * height
            x2 = (box.x_center + box.width / 2.0) * width
            y2 = (box.y_center + box.height / 2.0) * height
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(width), x2)
            y2 = min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            pascal_bboxes.append([x1, y1, x2, y2])
            class_labels.append(float(box.class_id))

        if not pascal_bboxes:
            pascal_bboxes = [[0.0, 0.0, 1.0, 1.0]]
            class_labels = [0.0]

        return (
            image,
            np.asarray(pascal_bboxes, dtype=np.float32),
            np.asarray(class_labels, dtype=np.float32),
            index,
        )


class EfficientDetDataset(Dataset):
    def __init__(self, adaptor: KittiEfficientDetAdaptor, transforms: A.Compose):
        self.adaptor = adaptor
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.adaptor)

    def __getitem__(self, index: int):
        image, pascal_bboxes, class_labels, image_id = self.adaptor.get_image_and_labels_by_idx(index)
        sample = self.transforms(
            image=image,
            bboxes=pascal_bboxes.tolist(),
            labels=class_labels.tolist(),
        )
        image_tensor = sample["image"]
        bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
        labels = np.asarray(sample["labels"], dtype=np.float32)

        if len(bboxes) == 0:
            bboxes = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            labels = np.asarray([0.0], dtype=np.float32)

        # Pascal VOC (xmin, ymin, xmax, ymax) -> EfficientDet (ymin, xmin, ymax, xmax)
        bboxes = bboxes[:, [1, 0, 3, 2]]
        _, height, width = image_tensor.shape

        target = {
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.float32),
            "image_id": torch.tensor([image_id]),
            "img_size": (height, width),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
        }
        return image_tensor, target, image_id


def efficientdet_collate(batch):
    images, targets, image_ids = zip(*batch)
    images = torch.stack(images).float()
    boxes = [target["bboxes"].float() for target in targets]
    labels = [target["labels"].float() for target in targets]
    img_size = torch.tensor([target["img_size"] for target in targets]).float()
    img_scale = torch.stack([target["img_scale"] for target in targets]).float()
    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": img_size,
        "img_scale": img_scale,
    }
    return images, annotations, list(targets), list(image_ids)
