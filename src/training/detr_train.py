from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from transformers import DetrConfig, DetrForObjectDetection


def _resize_boxes(boxes, original_size, target_size):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    original_height, original_width = original_size
    target_height, target_width = target_size
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    boxes = boxes.clone()
    boxes[:, 0] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 2] *= scale_x
    boxes[:, 3] *= scale_y
    boxes[:, 0] = boxes[:, 0].clamp(min=0.0, max=float(target_width))
    boxes[:, 1] = boxes[:, 1].clamp(min=0.0, max=float(target_height))
    boxes[:, 2] = boxes[:, 2].clamp(min=0.0, max=float(target_width))
    boxes[:, 3] = boxes[:, 3].clamp(min=0.0, max=float(target_height))
    return boxes


def _to_detr_labels(targets, device, image_size=(800, 800)):
    labels = []
    for t in targets:
        boxes = t["boxes"].to(device)
        labels_tensor = t["labels"].to(device)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_tensor = torch.zeros((0,), dtype=torch.long, device=device)
        labels.append({"class_labels": labels_tensor, "boxes": boxes})
    return labels

# Add src directory to path for imports
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def build_model(num_classes: int):
    config = DetrConfig(
        num_labels=num_classes + 1,
        hidden_size=256,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_queries=100,
        d_model=256,
        encoder_layers=3,
        decoder_layers=3,
    )
    model = DetrForObjectDetection(config)
    return model


def train_one_epoch(model, loader, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    print(json.dumps({"status": "epoch_start", "epoch": epoch, "batches": len(loader)}, ensure_ascii=False))
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        resized_images = []
        resized_targets = []
        for img, target in zip(images, targets):
            img = img.to(device)
            original_height, original_width = img.shape[1], img.shape[2]
            resized_img = T.Resize((800, 800))(img)
            resized_images.append(resized_img)

            boxes = target["boxes"].to(device)
            if boxes.numel() > 0:
                boxes = _resize_boxes(boxes, (original_height, original_width), (800, 800))
                boxes = boxes[:, [0, 1, 2, 3]]
                boxes[:, 2] = boxes[:, 2].clamp(min=boxes[:, 0].max().item() + 1e-6, max=800.0)
                boxes[:, 3] = boxes[:, 3].clamp(min=boxes[:, 1].max().item() + 1e-6, max=800.0)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            resized_targets.append({"boxes": boxes, "labels": target["labels"].to(device)})

        images = torch.stack(resized_images, dim=0)

        # Build labels in the format expected by HF Detr: boxes as
        # normalized [center_x, center_y, width, height] in range [0, 1]
        labels = []
        for t in resized_targets:
            boxes = t["boxes"]
            cls = t["labels"] if "labels" in t else None
            if boxes.numel() == 0:
                boxes_norm = boxes.new_zeros((0, 4))
                class_labels = boxes.new_zeros((0,), dtype=torch.long)
            else:
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                cx = (x1 + x2) / 2.0 / 800.0
                cy = (y1 + y2) / 2.0 / 800.0
                w = (x2 - x1) / 800.0
                h = (y2 - y1) / 800.0
                boxes_norm = torch.stack([cx, cy, w, h], dim=1)
                # dataset uses class_id+1; convert to zero-based contiguous labels
                # DETR expects labels in [0, num_labels-1] where background is included
                class_labels = (cls - 1).to(dtype=torch.long)
            labels.append({"class_labels": class_labels, "boxes": boxes_norm})

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            total_loss += 0.0
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        if batch_idx % 50 == 0:
            print(json.dumps({"status": "batch", "epoch": epoch, "batch": batch_idx, "loss": float(loss.item())}, ensure_ascii=False))
    return total_loss / max(1, len(loader))


def main() -> None:
    default_config = {}
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            default_config = yaml.safe_load(fh) or {}
    project_cfg = default_config.get("project", {})

    parser = argparse.ArgumentParser(description="Train DETR on KITTI YOLO labels.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--epochs", type=int, default=int(project_cfg.get("epochs", 1)))
    parser.add_argument("--batch-size", type=int, default=int(project_cfg.get("batch_size", 2)))
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--output-dir", default="results/detr")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(data_root, "train", CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate)

    # Detr expects num_labels == number of object classes (no background index)
    model = build_model(num_classes=len(CLASS_NAMES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch)
        history.append({"epoch": epoch, "loss": loss})
        print(json.dumps({"epoch": epoch, "loss": loss}, ensure_ascii=False))

    torch.save(model.state_dict(), output_dir / "detr_kitti.pt")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "detr_kitti.pt")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
