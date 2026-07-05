from pathlib import Path

import cv2
import torch

from src.dataset.yolo_detection_dataset import YoloDetectionDataset


def test_dataset_resizes_and_clips_boxes_for_ssd(tmp_path):
    images_dir = tmp_path / "train" / "images"
    labels_dir = tmp_path / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / "sample.png"
    cv2.imwrite(str(image_path), 255 * torch.ones(32, 32, 3, dtype=torch.uint8).numpy())
    (labels_dir / "sample.txt").write_text("0 1.2 0.5 0.4 0.2\n", encoding="utf-8")

    dataset = YoloDetectionDataset(tmp_path, "train", {0: "Car"}, input_size=(64, 64))
    image_tensor, target = dataset[0]

    assert image_tensor.shape == (3, 64, 64)
    assert target["boxes"].shape[0] == 1
    assert torch.all(target["boxes"][:, 0] >= 0)
    assert torch.all(target["boxes"][:, 2] <= 64)
    assert torch.all(target["boxes"][:, 1] >= 0)
    assert torch.all(target["boxes"][:, 3] <= 64)
