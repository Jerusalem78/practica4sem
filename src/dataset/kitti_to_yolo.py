from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


KITTI_CLASS_TO_ID: Dict[str, int] = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
    "Tram": 3,
    "Misc": 4,
}


@dataclass(frozen=True)
class KittiObject:
    class_name: str
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float


def parse_kitti_label_file(path: Path) -> List[KittiObject]:
    objects: List[KittiObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        class_name = parts[0]
        if class_name == "DontCare":
            continue
        bbox_left, bbox_top, bbox_right, bbox_bottom = map(float, parts[4:8])
        objects.append(
            KittiObject(
                class_name=class_name,
                bbox_left=bbox_left,
                bbox_top=bbox_top,
                bbox_right=bbox_right,
                bbox_bottom=bbox_bottom,
            )
        )
    return objects


def image_size_from_path(image_path: Path) -> Tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for KITTI conversion") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def to_yolo_line(obj: KittiObject, image_width: int, image_height: int) -> str | None:
    class_id = KITTI_CLASS_TO_ID.get(obj.class_name)
    if class_id is None:
        return None

    x_center = ((obj.bbox_left + obj.bbox_right) / 2.0) / image_width
    y_center = ((obj.bbox_top + obj.bbox_bottom) / 2.0) / image_height
    bbox_width = (obj.bbox_right - obj.bbox_left) / image_width
    bbox_height = (obj.bbox_bottom - obj.bbox_top) / image_height

    if bbox_width <= 0 or bbox_height <= 0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def collect_training_samples(raw_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = raw_root / "images" / "training" / "image_2"
    label_dir = raw_root / "labels" / "training" / "label_2"

    image_paths = sorted(image_dir.glob("*.png"))
    samples: List[Tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append((image_path, label_path))
    return samples


def split_samples(
    samples: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def prepare_output_dirs(output_root: Path) -> None:
    for split in ("train", "val"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_sample(
    image_path: Path,
    label_path: Path,
    output_root: Path,
    split: str,
) -> dict:
    target_image = output_root / split / "images" / image_path.name
    target_label = output_root / split / "labels" / f"{image_path.stem}.txt"

    shutil.copy2(image_path, target_image)
    image_width, image_height = image_size_from_path(image_path)

    yolo_lines: List[str] = []
    for obj in parse_kitti_label_file(label_path):
        yolo_line = to_yolo_line(obj, image_width, image_height)
        if yolo_line is not None:
            yolo_lines.append(yolo_line)

    target_label.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
    return {
        "image": str(target_image),
        "label": str(target_label),
        "objects": len(yolo_lines),
    }


def convert_kitti_to_yolo(
    raw_root: Path,
    output_root: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    samples = collect_training_samples(raw_root)
    if not samples:
        raise FileNotFoundError(
            f"No KITTI training samples found under {raw_root / 'images' / 'training' / 'image_2'}"
        )

    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)
    prepare_output_dirs(output_root)

    train_records = [write_sample(img, lbl, output_root, "train") for img, lbl in train_samples]
    val_records = [write_sample(img, lbl, output_root, "val") for img, lbl in val_samples]

    return {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_objects": sum(item["objects"] for item in train_records),
        "val_objects": sum(item["objects"] for item in val_records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert KITTI object detection data to YOLO format.")
    parser.add_argument("--raw-root", default="data/raw", help="Path to KITTI raw data root.")
    parser.add_argument("--output-root", default="data/processed", help="Where to write YOLO data.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = convert_kitti_to_yolo(
        raw_root=Path(args.raw_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
