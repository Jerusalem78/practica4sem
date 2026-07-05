from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Add src directory to path for imports
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def build_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    print(json.dumps({"status": "epoch_start", "epoch": epoch, "batches": len(loader)}, ensure_ascii=False))
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
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

    parser = argparse.ArgumentParser(description="Train Faster R-CNN on KITTI YOLO labels.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--epochs", type=int, default=int(project_cfg.get("epochs", 1)))
    parser.add_argument("--batch-size", type=int, default=int(project_cfg.get("batch_size", 2)))
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output-dir", default="results/faster_rcnn")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(data_root, "train", CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate)

    model = build_model(num_classes=len(CLASS_NAMES) + 1)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    history = []
    print(json.dumps({"status": "setup", "device": str(device), "batch_size": args.batch_size, "epochs": args.epochs, "dataset_size": len(dataset)}, ensure_ascii=False))
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch)
        history.append({"epoch": epoch, "loss": loss})
        print(json.dumps({"epoch": epoch, "loss": loss}, ensure_ascii=False))

    torch.save(model.state_dict(), output_dir / "faster_rcnn_kitti.pt")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output_dir / "faster_rcnn_kitti.pt")}, ensure_ascii=False))

 
if __name__ == "__main__":
    main()
