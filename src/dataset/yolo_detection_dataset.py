from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def _resolve_image_path(images_dir: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for stem {stem} in {images_dir}")


def _parse_label_file(path: Path) -> List[YoloBox]:
    boxes: List[YoloBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append(
            YoloBox(
                class_id=int(class_id),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )
    return boxes


class YoloDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: Dict[int, str], input_size: Tuple[int, int] | None = None):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.allowed_class_ids = set(class_names.keys())
        self.input_size = input_size
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        target_width = width
        target_height = height
        if self.input_size is not None:
            target_height, target_width = self.input_size
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        for box in _parse_label_file(label_path):
            if box.class_id not in self.allowed_class_ids:
                continue
            x1 = (box.x_center - box.width / 2.0) * width
            y1 = (box.y_center - box.height / 2.0) * height
            x2 = (box.x_center + box.width / 2.0) * width
            y2 = (box.y_center + box.height / 2.0) * height
            if x2 <= x1 or y2 <= y1:
                continue
            if self.input_size is not None:
                x1 = x1 * (target_width / width)
                y1 = y1 * (target_height / height)
                x2 = x2 * (target_width / width)
                y2 = y2 * (target_height / height)
            x1 = max(0.0, min(float(x1), float(target_width)))
            y1 = max(0.0, min(float(y1), float(target_height)))
            x2 = max(0.0, min(float(x2), float(target_width)))
            y2 = max(0.0, min(float(y2), float(target_height)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(box.class_id + 1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(
                [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes_xyxy], dtype=torch.float32
            ),
            "iscrowd": torch.zeros((len(boxes_xyxy),), dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
