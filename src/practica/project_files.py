from __future__ import annotations

from pathlib import Path
from typing import List


PROJECT_DIRS = [
    "data/raw",
    "data/processed",
    "src/dataset",
    "src/models",
    "src/training",
    "src/evaluation",
    "notebooks",
    "configs",
    "results",
]


def ensure_project_structure(root: Path) -> List[Path]:
    created: List[Path] = []
    for rel_path in PROJECT_DIRS:
        path = root / rel_path
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
