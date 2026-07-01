from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .project_files import ensure_project_structure


@dataclass
class ProjectConfig:
    root: Path
    dataset_name: str = "COCO-like detection benchmark"
    task: str = "object_detection"
    models: List[str] = field(
        default_factory=lambda: [
            "YOLO",
            "Faster R-CNN",
            "SSD",
            "EfficientDet",
            "DETR",
        ]
    )

    def summary_text(self) -> str:
        lines = [
            f"Project root: {self.root}",
            f"Task: {self.task}",
            f"Dataset: {self.dataset_name}",
            "Models:",
        ]
        lines.extend(f"  - {model}" for model in self.models)
        lines.append("")
        lines.append("Run `practica --mode scaffold` to create the project structure.")
        return "\n".join(lines)


def run_project(config: ProjectConfig) -> Dict[str, object]:
    structure = ensure_project_structure(config.root)
    return {
        "status": "scaffolded",
        "root": str(config.root),
        "created_paths": [str(path) for path in structure],
        "next_steps": [
            "download a dataset into data/raw",
            "add annotations converter",
            "plug in 5 model wrappers",
            "run training and export metrics to results/",
        ],
    }
