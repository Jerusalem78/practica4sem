from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16

from evaluation.faster_rcnn_eval import evaluate_model
from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def build_model(num_classes: int):
    return ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSD on KITTI validation split.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--weights", default="results/ssd/ssd_kitti.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="results/metrics/ssd_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(Path(args.data_root), "val", CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate)

    model = build_model(num_classes=len(CLASS_NAMES) + 1)
    state_dict = torch.load(Path(args.weights), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_model(model, loader, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# SSD evaluation on KITTI val",
        "",
        f"- Precision: `{metrics['precision']}`",
        f"- Recall: `{metrics['recall']}`",
        f"- F1: `{metrics['f1']}`",
        f"- mAP@0.5: `{metrics['map50']}`",
        f"- mAP@0.5:0.95: `{metrics['map50_95']}`",
        "",
        "Per-class:",
    ]
    for class_name, values in metrics["per_class"].items():
        lines.append(
            f"- {class_name}: P `{values['precision']}`, R `{values['recall']}`, mAP50 `{values['map50']}`, mAP50-95 `{values['map50_95']}`"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    comparison_path = Path("results/metrics/comparison_table.md")
    if comparison_path.exists():
        table_lines = comparison_path.read_text(encoding="utf-8").splitlines()
        updated_lines = []
        for line in table_lines:
            if line.startswith("| SSD |"):
                updated_lines.append(
                    f"| SSD | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | evaluated on KITTI val after SSD training |"
                )
            else:
                updated_lines.append(line)
        comparison_path.write_text("\n".join(updated_lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
