from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import DetrConfig, DetrForObjectDetection
from torch.utils.data import DataLoader

from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate
from evaluation.faster_rcnn_eval import IOU_THRESHOLDS, box_iou, evaluate_class

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


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess_detr_outputs(
    outputs,
    image_sizes: List[Tuple[int, int]],
    score_threshold: float = 0.25,
) -> List[List[Tuple[int, float, np.ndarray]]]:
    logits = getattr(outputs, "logits", None)
    pred_boxes = getattr(outputs, "pred_boxes", None)
    if logits is None or pred_boxes is None:
        raise AttributeError("DETR outputs do not contain logits/pred_boxes")

    probs = logits.softmax(-1)
    object_probs = probs[..., :-1]
    no_object_probs = probs[..., -1]
    scores, labels = object_probs.max(-1)

    detections: List[List[Tuple[int, float, np.ndarray]]] = []
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    for image_size, score_row, label_row, box_row, noobj_row in zip(
        image_sizes, scores, labels, boxes_xyxy, no_object_probs
    ):
        height, width = image_size
        boxes_scaled = box_row.clone()
        boxes_scaled[:, 0] = boxes_scaled[:, 0].clamp(0.0, 1.0) * width
        boxes_scaled[:, 1] = boxes_scaled[:, 1].clamp(0.0, 1.0) * height
        boxes_scaled[:, 2] = boxes_scaled[:, 2].clamp(0.0, 1.0) * width
        boxes_scaled[:, 3] = boxes_scaled[:, 3].clamp(0.0, 1.0) * height

        row_detections: List[Tuple[int, float, np.ndarray]] = []
        for score, label, box, noobj_score in zip(score_row, label_row, boxes_scaled, noobj_row):
            if float(score) < score_threshold:
                continue
            if float(score) <= float(noobj_score):
                continue
            class_idx = int(label)
            if class_idx not in CLASS_NAMES:
                continue
            box_np = box.detach().cpu().numpy().astype(np.float32)
            row_detections.append((class_idx, float(score), box_np))
        detections.append(row_detections)
    return detections


def evaluate_detr(model, loader, device, score_threshold: float = 0.25) -> dict:
    model.eval()
    gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = defaultdict(list)
    global_image_id = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = torch.stack(images).to(device)
            image_sizes = [(img.shape[1], img.shape[2]) for img in images]
            print(f"[DETR eval] batch {batch_idx}/{len(loader)}: processing {len(images)} images")
            outputs = model(pixel_values=images)
            detections = postprocess_detr_outputs(
                outputs, image_sizes, score_threshold=score_threshold
            )

            for dets, target in zip(detections, targets):
                for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
                    gt_by_class[int(label) - 1][global_image_id].append(box)
                for class_idx, score, box in dets:
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
        if line.startswith("| DETR |"):
            updated_lines.append(
                f"| DETR | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | evaluated on KITTI val after DETR training |"
            )
        else:
            updated_lines.append(line)
    comparison_path.write_text("\n".join(updated_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DETR on KITTI validation split.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--weights", default="results/detr/detr_kitti.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="results/metrics/detr_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(Path(args.data_root), "val", CLASS_NAMES, input_size=(args.image_size, args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate)

    model = build_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(Path(args.weights), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_detr(model, loader, device, score_threshold=args.score_threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = output_path.with_suffix(".md")
    lines = [
        "# DETR evaluation on KITTI val",
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
