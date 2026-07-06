from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from effdet import create_model
from torch.utils.data import DataLoader

# Add src directory to path for imports
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dataset.efficientdet_dataset import (
    CLASS_NAMES,
    EfficientDetDataset,
    KittiEfficientDetAdaptor,
    efficientdet_collate,
    get_train_transforms,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_experiment_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_model(architecture: str, num_classes: int, pretrained: bool):
    return create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        bench_task="train",
        bench_labeler=True,
    )


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> dict:
    model.train()
    total_loss = 0.0
    total_class_loss = 0.0
    total_box_loss = 0.0
    print(json.dumps({"status": "epoch_start", "epoch": epoch, "batches": len(loader)}, ensure_ascii=False))

    for batch_idx, (images, annotations, _, _) in enumerate(loader, start=1):
        images = images.to(device)
        annotations = {
            "bbox": [bbox.to(device) for bbox in annotations["bbox"]],
            "cls": [cls.to(device) for cls in annotations["cls"]],
            "img_size": annotations["img_size"].to(device),
            "img_scale": annotations["img_scale"].to(device),
        }

        outputs = model(images, annotations)
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            print(json.dumps({"warning": "non-finite loss", "epoch": epoch, "batch": batch_idx}, ensure_ascii=False))
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_class_loss += float(outputs["class_loss"].item())
        total_box_loss += float(outputs["box_loss"].item())
        if batch_idx % 50 == 0:
            print(
                json.dumps(
                    {
                        "status": "batch",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": float(loss.item()),
                        "class_loss": float(outputs["class_loss"].item()),
                        "box_loss": float(outputs["box_loss"].item()),
                    },
                    ensure_ascii=False,
                )
            )

    batches = max(1, len(loader))
    return {
        "loss": total_loss / batches,
        "class_loss": total_class_loss / batches,
        "box_loss": total_box_loss / batches,
    }


def main() -> None:
    default_config = {}
    default_config_path = Path("configs/default.yaml")
    if default_config_path.exists():
        with default_config_path.open("r", encoding="utf-8") as fh:
            default_config = yaml.safe_load(fh) or {}
    project_cfg = default_config.get("project", {})
    experiment_cfg = load_experiment_config(Path("configs/experiment_efficientdet.yaml"))
    hyperparams = experiment_cfg.get("hyperparameters", {})

    parser = argparse.ArgumentParser(description="Train EfficientDet on KITTI YOLO labels.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--architecture", default=experiment_cfg.get("arch", "tf_efficientdet_d0"))
    parser.add_argument("--epochs", type=int, default=int(hyperparams.get("epochs", project_cfg.get("epochs", 7))))
    parser.add_argument("--batch-size", type=int, default=int(hyperparams.get("batch_size", project_cfg.get("batch_size", 4))))
    parser.add_argument("--img-size", type=int, default=int(hyperparams.get("imgsz", 512)))
    parser.add_argument("--lr", type=float, default=float(hyperparams.get("lr", 0.0004)))
    parser.add_argument("--weight-decay", type=float, default=float(hyperparams.get("weight_decay", 1e-5)))
    parser.add_argument("--seed", type=int, default=int(hyperparams.get("seed", 42)))
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=experiment_cfg.get("pretrained", True))
    parser.add_argument("--output-dir", default="results/efficientdet")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptor = KittiEfficientDetAdaptor(data_root=data_root, split="train", class_names=CLASS_NAMES)
    dataset = EfficientDetDataset(adaptor, transforms=get_train_transforms(args.img_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=efficientdet_collate,
    )

    model = build_model(args.architecture, num_classes=len(CLASS_NAMES), pretrained=args.pretrained)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    print(
        json.dumps(
            {
                "status": "setup",
                "device": str(device),
                "architecture": args.architecture,
                "num_classes": len(CLASS_NAMES),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "img_size": args.img_size,
                "dataset_size": len(dataset),
                "pretrained": args.pretrained,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, loader, optimizer, device, epoch)
        record = {"epoch": epoch, **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    weights_path = output_dir / "efficientdet_kitti.pt"
    torch.save(model.state_dict(), weights_path)
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    run_config = {
        "architecture": args.architecture,
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES.values()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "device": str(device),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "saved": str(weights_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
