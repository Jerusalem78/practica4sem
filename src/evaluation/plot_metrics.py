from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_FILES = [
    "faster_rcnn_eval.json",
    "yolo_eval.json",
    "ssd_eval.json",
    "efficientdet_eval.json",
    "detr_eval.json",
]


def load_metrics(metrics_dir: Path) -> Dict[str, dict]:
    metrics = {}
    for filename in METRIC_FILES:
        path = metrics_dir / filename
        if not path.exists():
            continue
        model_name = filename.replace("_eval.json", "").replace("_", " ").upper()
        if model_name == "YOLO":
            model_name = "YOLOv8n"
        metrics[model_name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def plot_overall_metrics(metrics: Dict[str, dict], output_dir: Path) -> None:
    if not metrics:
        return

    categories = ["precision", "recall", "f1", "map50", "map50_95"]
    models = list(metrics.keys())
    values = [[metrics[model].get(category, 0.0) for model in models] for category in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    total_width = 0.8
    width = total_width / len(categories)

    for idx, (category, vals) in enumerate(zip(categories, values)):
        ax.bar(
            [pos + idx * width for pos in x],
            vals,
            width=width,
            label=category,
            edgecolor="black",
        )

    ax.set_xticks([pos + total_width / 2 - width / 2 for pos in x])
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Value")
    ax.set_title("Overall Detection Metrics by Model")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path = output_dir / "overall_metrics.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(metrics: Dict[str, dict], output_dir: Path) -> None:
    if not metrics:
        return

    categories = ["precision", "recall", "f1", "map50", "map50_95"]
    class_names = ["Car", "Pedestrian", "Cyclist"]
    for class_name in class_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(metrics))
        models = list(metrics.keys())
        total_width = 0.8
        width = total_width / len(categories)

        values = []
        for category in categories:
            vals = [metrics[model].get("per_class", {}).get(class_name, {}).get(category, 0.0) for model in models]
            values.append(vals)

        for idx, (category, vals) in enumerate(zip(categories, values)):
            ax.bar(
                [pos + idx * width for pos in x],
                vals,
                width=width,
                label=category,
                edgecolor="black",
            )

        ax.set_xticks([pos + total_width / 2 - width / 2 for pos in x])
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Value")
        ax.set_title(f"Per-class metrics for {class_name}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        output_path = output_dir / f"per_class_metrics_{class_name.lower()}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    metrics_dir = repo_root / "results" / "metrics"
    output_dir = metrics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_dir)
    if not metrics:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir}")

    plot_overall_metrics(metrics, output_dir)
    plot_per_class_metrics(metrics, output_dir)
    print(f"Saved overall_metrics.png and per_class_metrics_*.png in {output_dir}")


if __name__ == "__main__":
    main()
