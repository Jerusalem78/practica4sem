from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from effdet import create_model
from torch.utils.data import DataLoader

from dataset.efficientdet_dataset import (
    CLASS_NAMES,
    EfficientDetDataset,
    KittiEfficientDetAdaptor,
    efficientdet_collate,
    get_valid_transforms,
)
from evaluation.faster_rcnn_eval import IOU_THRESHOLDS, box_iou, evaluate_class


def build_model(architecture: str, num_classes: int):
    return create_model(
        architecture,
        pretrained=False,
        num_classes=num_classes,
        bench_task="predict",
        bench_labeler=True,
    )


def align_state_dict_prefix(state_dict: dict, model: torch.nn.Module, prefix: str = "model.") -> dict:
    if not state_dict:
        return state_dict
    model_keys = next(iter(model.state_dict().keys()))
    state_keys = next(iter(state_dict.keys()))
    if model_keys.startswith(prefix) and state_keys.startswith(prefix):
        return state_dict
    if state_keys.startswith(prefix) and not model_keys.startswith(prefix):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    if not state_keys.startswith(prefix) and model_keys.startswith(prefix):
        return {f"{prefix}{key}": value for key, value in state_dict.items()}
    return state_dict


def load_run_config(output_dir: Path) -> dict:
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def detections_to_predictions(
    detections: torch.Tensor,
    score_threshold: float,
) -> List[Tuple[int, float, np.ndarray]]:
    predictions: List[Tuple[int, float, np.ndarray]] = []
    det = detections.detach().cpu().numpy()
    for row in det:
        if row.shape[0] < 6:
            continue
        x1, y1, x2, y2, score, class_id = row[:6]
        if score < score_threshold:
            continue
        class_idx = int(class_id) - 1
        if class_idx not in CLASS_NAMES:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        predictions.append((class_idx, float(score), box))
    return predictions


def evaluate_efficientdet(
    model,
    loader,
    device,
    image_size: int,
    score_threshold: float = 0.05,
) -> dict:
    model.eval()
    gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]] = defaultdict(list)
    global_image_id = 0

    with torch.no_grad():
        for images, annotations, targets, _ in loader:
            images = images.to(device)
            img_info = {
                "img_scale": annotations["img_scale"].to(device),
                "img_size": annotations["img_size"].to(device),
            }

            detections = model(images, img_info=img_info)
            if isinstance(detections, tuple) and len(detections) == 1:
                detections = detections[0]

            for det, target in zip(detections, targets):
                boxes = target["bboxes"].cpu().numpy()
                labels = target["labels"].cpu().numpy().astype(int)
                for box, label in zip(boxes, labels):
                    if int(label) not in CLASS_NAMES:
                        continue
                    # yxyx -> xyxy
                    ymin, xmin, ymax, xmax = box
                    gt_by_class[int(label)][global_image_id].append(
                        np.asarray([xmin, ymin, xmax, ymax], dtype=np.float32)
                    )

                for class_idx, score, box in detections_to_predictions(
                    det,
                    score_threshold=score_threshold,
                ):
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
        if line.startswith("| EfficientDet |"):
            updated_lines.append(
                f"| EfficientDet | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | evaluated on KITTI val after EfficientDet training |"
            )
        else:
            updated_lines.append(line)
    comparison_path.write_text("\n".join(updated_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EfficientDet on KITTI validation split.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--weights", default="results/efficientdet/efficientdet_kitti.pt")
    parser.add_argument("--architecture", default="tf_efficientdet_d0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="results/metrics/efficientdet_eval.json")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    run_config = load_run_config(weights_path.parent)
    architecture = run_config.get("architecture", args.architecture)
    image_size = int(run_config.get("img_size", args.img_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptor = KittiEfficientDetAdaptor(data_root=Path(args.data_root), split="val", class_names=CLASS_NAMES)
    dataset = EfficientDetDataset(adaptor, transforms=get_valid_transforms(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=efficientdet_collate,
    )

    model = build_model(architecture, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = align_state_dict_prefix(state_dict, model, prefix="model.")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    metrics = evaluate_efficientdet(
        model,
        loader,
        device,
        image_size=image_size,
        score_threshold=args.score_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# EfficientDet evaluation on KITTI val",
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
