from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def load_training_config(config_path: Path | None = None) -> dict:
    candidates = []
    if config_path is not None:
        candidates.append(config_path)
    candidates.extend([
        Path("configs/experiment_yolo.yaml"),
        Path("configs/default.yaml"),
    ])

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on KITTI YOLO labels.")
    parser.add_argument("--config", default="configs/experiment_yolo.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", default="results/yolo")
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    training_config = load_training_config(Path(args.config))
    hyperparameters = training_config.get("hyperparameters", {})
    if args.epochs is None:
        args.epochs = int(hyperparameters.get("epochs", 7))
    if args.batch_size is None:
        args.batch_size = int(hyperparameters.get("batch_size", 8))
    if args.img_size is None:
        args.img_size = int(hyperparameters.get("imgsz", 640))
    if args.lr is None:
        args.lr = float(hyperparameters.get("lr0", hyperparameters.get("lr", 0.01)))
    if args.seed is None:
        args.seed = int(hyperparameters.get("seed", 42))
    if args.model == "yolov8n.pt" and training_config.get("pretrained_weights"):
        args.model = training_config.get("pretrained_weights")

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr,
        project=str(output_dir.parent),
        name=output_dir.name,
        device=device,
        workers=0,
        seed=args.seed,
        exist_ok=True,
        verbose=False,
    )

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "config": args.config}, ensure_ascii=False))
    print(results)


if __name__ == "__main__":
    main()
