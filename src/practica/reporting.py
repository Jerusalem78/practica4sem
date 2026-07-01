from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class MetricRow:
    model: str
    precision: float
    recall: float
    f1: float
    map50: float


def format_metric_table(rows: Iterable[MetricRow]) -> str:
    header = "| Model | Precision | Recall | F1 | mAP@0.5 |"
    separator = "|---|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            f"| {row.model} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.map50:.3f} |"
        )
    return "\n".join(lines)


def mock_benchmark_rows() -> List[MetricRow]:
    return [
        MetricRow("YOLO", 0.71, 0.66, 0.69, 0.62),
        MetricRow("Faster R-CNN", 0.75, 0.68, 0.71, 0.65),
        MetricRow("SSD", 0.63, 0.60, 0.61, 0.54),
        MetricRow("EfficientDet", 0.73, 0.69, 0.71, 0.66),
        MetricRow("DETR", 0.74, 0.70, 0.72, 0.67),
    ]
