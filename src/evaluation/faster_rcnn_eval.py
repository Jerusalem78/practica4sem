from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


def build_model(num_classes: int):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_class(
    predictions: List[Tuple[int, float, np.ndarray]],
    ground_truths: Dict[int, List[np.ndarray]],
    iou_threshold: float,
) -> float:
    preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    total_gts = sum(len(v) for v in ground_truths.values())
    if total_gts == 0:
        return 0.0

    matched = {image_id: [False] * len(boxes) for image_id, boxes in ground_truths.items()}
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    for i, (image_id, score, box) in enumerate(preds):
        gt_boxes = ground_truths.get(image_id, [])
        best_iou = 0.0
        best_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            current_iou = box_iou(box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx >= 0 and not matched[image_id][best_idx]:
            tp[i] = 1.0
            matched[image_id][best_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, total_gts)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return average_precision(recalls, precisions)


def evaluate_model(model, loader, device, score_threshold: float = 0.25) -> dict:
    model.eval()
    gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = defaultdict(list)
    global_image_id = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
                    gt_by_class[int(label) - 1][global_image_id].append(box)
                scores = output["scores"].detach().cpu().numpy()
                boxes = output["boxes"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if score < score_threshold or int(label) == 0:
                        continue
                    pred_by_class[int(label) - 1].append((global_image_id, float(score), box))
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


def load_model(weights_path: Path, device):
    model = build_model(num_classes=len(CLASS_NAMES) + 1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on KITTI validation split.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--weights", default="results/faster_rcnn/faster_rcnn_kitti.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="results/metrics/faster_rcnn_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(Path(args.data_root), "val", CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate)
    model = load_model(Path(args.weights), device)
    metrics = evaluate_model(model, loader, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
