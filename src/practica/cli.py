from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.kitti_to_yolo import convert_kitti_to_yolo
from training.faster_rcnn_train import main as train_faster_rcnn_main
from training.ssd_train import main as train_ssd_main
from training.detr_train import main as train_detr_main
from training.yolo_train import main as train_yolo_main
from training.efficientdet_train import main as train_efficientdet_main
from evaluation.faster_rcnn_eval import main as eval_faster_rcnn_main
from evaluation.ssd_eval import main as eval_ssd_main
from evaluation.efficientdet_eval import main as eval_efficientdet_main
from evaluation.detr_eval import main as eval_detr_main
from evaluation.yolo_eval import main as eval_yolo_main
from .config_loader import load_default_config
from .pipeline import ProjectConfig, run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="practica",
        description="Utilities for the computer-vision practice project.",
    )
    default_config = load_default_config(Path("."))
    project_cfg = default_config.get("project", {})
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "scaffold",
            "summary",
            "convert-kitti",
            "train-faster-rcnn",
            "train-ssd",
            "train-detr",
            "train-yolo",
            "train-efficientdet",
            "eval-faster-rcnn",
            "eval-ssd",
            "eval-efficientdet",
            "eval-detr",
            "eval-yolo",
        ],
        default="summary",
        help="Print project summary, create folders, or convert KITTI to YOLO.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="KITTI raw data root for conversion.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed",
        help="Output root for converted data.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for conversion.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(project_cfg.get("epochs", 1)),
        help="Training epochs for Faster R-CNN mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(project_cfg.get("batch_size", 2)),
        help="Batch size for Faster R-CNN mode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(project_cfg.get("lr", 0.005)),
        help="Learning rate for training modes.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/faster_rcnn",
        help="Output directory for Faster R-CNN mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    default_config = load_default_config(root)
    project_cfg = default_config.get("project", {})
    config = ProjectConfig(root=root)

    if args.mode == "scaffold":
        result = run_project(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "convert-kitti":
        result = convert_kitti_to_yolo(
            raw_root=Path(args.raw_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            val_ratio=args.val_ratio,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "train-faster-rcnn":
        import sys

        sys.argv = [
            sys.argv[0],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--output-dir",
            args.output_dir,
        ]
        train_faster_rcnn_main()
    elif args.mode == "train-ssd":
        import sys

        sys.argv = [
            sys.argv[0],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--output-dir",
            "results/ssd",
        ]
        train_ssd_main()
    elif args.mode == "train-detr":
        import sys

        sys.argv = [
            sys.argv[0],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--output-dir",
            "results/detr",
        ]
        train_detr_main()
    elif args.mode == "train-yolo":
        import sys

        sys.argv = [
            sys.argv[0],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--img-size",
            str(project_cfg.get("image_size", 640)),
            "--lr",
            str(args.lr),
            "--output-dir",
            "results/yolo",
        ]
        train_yolo_main()
    elif args.mode == "train-efficientdet":
        import sys

        sys.argv = [
            sys.argv[0],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--output-dir",
            "results/efficientdet",
        ]
        train_efficientdet_main()
    elif args.mode == "eval-faster-rcnn":
        import sys

        sys.argv = [
            sys.argv[0],
        ]
        eval_faster_rcnn_main()
    elif args.mode == "eval-ssd":
        import sys

        sys.argv = [
            sys.argv[0],
        ]
        eval_ssd_main()
    elif args.mode == "eval-efficientdet":
        import sys

        sys.argv = [
            sys.argv[0],
        ]
        eval_efficientdet_main()
    elif args.mode == "eval-detr":
        import sys

        sys.argv = [
            sys.argv[0],
        ]
        eval_detr_main()
    elif args.mode == "eval-yolo":
        import sys

        sys.argv = [
            sys.argv[0],
        ]
        eval_yolo_main()
    else:
        print(config.summary_text())
