from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate
from evaluation.faster_rcnn_eval import IOU_THRESHOLDS, box_iou, evaluate_class


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def build_model(weights_path: Path):
    return YOLO(str(weights_path))


def prepare_images(images: List[torch.Tensor]) -> List[np.ndarray]:
    prepared: List[np.ndarray] = []
    for image in images:
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
            image_np = np.clip(image_np, 0.0, 1.0)
            image_np = (image_np * 255.0).round().astype(np.uint8)
            image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = np.asarray(image)
        prepared.append(image_np)
    return prepared


def parse_yolo_results(result, score_threshold: float) -> List[Tuple[int, float, np.ndarray]]:
    if result is None or not hasattr(result, "boxes"):
        return []

    boxes = result.boxes
    if boxes is None:
        return []

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            xyxy = np.asarray(boxes.xyxy)
            scores = np.asarray(boxes.conf)
            labels = np.asarray(boxes.cls).astype(int)
        except Exception:
            return []

    predictions: List[Tuple[int, float, np.ndarray]] = []
    for box, score, class_idx in zip(xyxy, scores, labels):
        if float(score) < score_threshold:
            continue
        if int(class_idx) not in CLASS_NAMES:
            continue
        predictions.append((int(class_idx), float(score), box.astype(np.float32)))
    return predictions


def evaluate_yolo(model, loader, device, img_size: int, score_threshold: float = 0.25) -> dict:
    gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = defaultdict(list)
    global_image_id = 0

    model.to(device)
    model.overrides = {"conf": score_threshold}

    with torch.no_grad():
        for images, targets in loader:
            image_inputs = prepare_images(images)
            results = model.predict(
                source=image_inputs,
                imgsz=img_size,
                device=str(device),
                conf=score_threshold,
                verbose=False,
            )

            for result, target in zip(results, targets):
                for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
                    gt_by_class[int(label) - 1][global_image_id].append(box.astype(np.float32))

                detections = parse_yolo_results(result, score_threshold=score_threshold)
                for class_idx, score, box in detections:
                    pred_by_class[class_idx].append((global_image_id, score, box))
                global_image_id += 1

    per_class = {}
    ap50_values = []
    ap5095_values = []
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for class_id, class_name in CLASS_NAMES.items():
        class_gts = gt_by_class.get(class_id, {})
        class_preds = pred_by_class.get(class_id, [])
        ap50 = evaluate_class(class_preds, class_gts, 0.5)
        ap50_95 = float(np.mean([evaluate_class(class_preds, class_gts, thr) for thr in IOU_THRESHOLDS]))
        ap50_values.append(ap50)
        ap5095_values.append(ap50_95)

        total_gts = sum(len(v) for v in class_gts.values())
        matched = {image_id: [False] * len(boxes) for image_id, boxes in class_gts.items()}
        preds_sorted = sorted(class_preds, key=lambda x: x[1], reverse=True)
        tp = 0
        fp = 0
        for image_id, score, box in preds_sorted:
            gt_boxes = class_gts.get(image_id, [])
            best_iou = 0.0
            best_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                current_iou = box_iou(box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = j
            if best_iou >= 0.5 and best_idx >= 0 and not matched[image_id][best_idx]:
                tp += 1
                matched[image_id][best_idx] = True
            else:
                fp += 1
        fn = total_gts - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, total_gts)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": ap50,
            "map50_95": ap50_95,
            "gt": total_gts,
            "pred": len(class_preds),
        }

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(np.mean(ap50_values)),
        "map50_95": float(np.mean(ap5095_values)),
        "per_class": per_class,
    }


def update_comparison_table(metrics: dict) -> None:
    comparison_path = Path("results/metrics/comparison_table.md")
    if not comparison_path.exists():
        return
    table_lines = comparison_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    for line in table_lines:
        if line.startswith("| YOLO |"):
            updated_lines.append(
                f"| YOLO | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | evaluated on KITTI val after YOLO training |"
            )
        else:
            updated_lines.append(line)
    comparison_path.write_text("\n".join(updated_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on KITTI validation split.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--weights", default="results/yolo/best.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="results/metrics/yolo_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(Path(args.data_root), "val", CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate)

    model = build_model(Path(args.weights))
    metrics = evaluate_yolo(model, loader, device, img_size=args.img_size, score_threshold=args.score_threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# YOLO evaluation on KITTI val",
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
    update_comparison_table(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
