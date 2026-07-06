from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_default_config(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or ".").resolve()
    config_path = root / "configs" / "default.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
