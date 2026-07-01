from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ProjectConfig, run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="practica",
        description="Utilities for the computer-vision practice project.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["scaffold", "summary"],
        default="summary",
        help="Print project summary or create folders and sample files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    config = ProjectConfig(root=root)

    if args.mode == "scaffold":
        result = run_project(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(config.summary_text())
