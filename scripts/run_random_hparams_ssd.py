#!/usr/bin/env python
"""Run random SSD hyperparameter experiments using experiment_ssd.yaml as the base config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict:
    config_path = config_path or ROOT / "configs" / "experiment_ssd.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_run_params(config: dict, run_index: int) -> dict:
    hyperparameters = config.get("hyperparameters", {})

    presets = hyperparameters.get("presets")
    if isinstance(presets, list) and presets:
        selected = presets[run_index - 1]
    else:
        selected = {
            "epochs": hyperparameters.get("epochs", 10),
            "batch_size": hyperparameters.get("batch_size", 16),
            "lr": hyperparameters.get("lr", 0.001),
        }

    fixed_params = {
        "seed": int(hyperparameters.get("seed", 42)),
    }

    return {**fixed_params, **selected}


def run_once(config_path: Path, run_index: int, output_root: Path, dry_run: bool = False) -> dict:
    config = load_config(config_path)
    params = build_run_params(config, run_index=run_index)
    run_name = f"run_{run_index:02d}_e{params['epochs']}_b{params['batch_size']}_lr{params['lr']}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = os.environ.get("PYTHON_EXE") or sys.executable
    cmd = [
        python_exe,
        str(ROOT / "src" / "training" / "ssd_train.py"),
        "--data-root",
        str(ROOT / "data" / "processed"),
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--lr",
        str(params["lr"]),
        "--output-dir",
        str(output_dir),
    ]

    meta = {
        "run_name": run_name,
        "config_path": str(config_path),
        "params": params,
        "command": cmd,
    }
    (output_dir / "params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Starting run {run_index}: {run_name}")
    print("Parameters:", json.dumps(params, ensure_ascii=False))

    if dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return meta

    print(f"Starting run {run_index}: {run_name}")
    log_path = output_dir / "train.log"

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )

    # Stream output ONLY to console, not to log
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    returncode = process.wait()

    # Write only results to log file
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Exit code: {returncode}\n\n")

        # Find and write results from SSD's output
        history_file = output_dir / "history.json"
        if history_file.exists():
            log_file.write("=== Training History ===\n")
            log_file.write(history_file.read_text())

    print(f"Finished run {run_index} with exit code {returncode}")
    print(f"Logs saved to {log_path}")
    print(f"Artifacts saved to {output_dir}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random SSD hyperparameter experiments using experiment_ssd.yaml as the base config.")
    parser.add_argument("--config", default="configs/experiment_ssd.yaml", help="Path to the experiment YAML file.")
    parser.add_argument("--runs", type=int, default=4, help="How many random runs to execute.")
    parser.add_argument("--output-root", default="results/random_runs_ssd", help="Folder where run artifacts will be saved.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing training.")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output_root = (ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for run_index in range(1, args.runs + 1):
        run_once(config_path=config_path, run_index=run_index, output_root=output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
