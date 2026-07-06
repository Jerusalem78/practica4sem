from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "results" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_HISTORY_FILES = {
    "Faster R-CNN": RESULTS_DIR / "faster_rcnn" / "history.json",
    "SSD": RESULTS_DIR / "ssd" / "history.json",
    "EfficientDet": RESULTS_DIR / "efficientdet" / "history.json",
    "DETR": RESULTS_DIR / "detr" / "history.json",
}
YOLO_RESULTS_CSV = ROOT / "runs" / "detect" / "train" / "results.csv"


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_yolo_history(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            values = line.strip().split(",")
            row = dict(zip(header, values))
            try:
                box_loss = float(row.get("train/box_loss", 0.0))
                cls_loss = float(row.get("train/cls_loss", 0.0))
                dfl_loss = float(row.get("train/dfl_loss", 0.0))
                rows.append({
                    "epoch": int(row["epoch"]),
                    "loss": (box_loss + cls_loss + dfl_loss) / 3.0,
                })
            except (KeyError, ValueError):
                continue
    return rows


def limit_history(history: list[dict], max_epochs: int = 7) -> list[dict]:
    return history[:max_epochs]


def load_series(name: str, path: Path) -> list[dict]:
    if name == "YOLO":
        return load_yolo_history(path)
    return load_history(path)


def plot_overall_loss() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, path in MODEL_HISTORY_FILES.items():
        history = limit_history(load_series(name, path))
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        losses = [entry.get("loss", float("nan")) for entry in history]
        ax.plot(epochs, losses, marker="o", linewidth=1.8, label=name)

    yolo_history = limit_history(load_series("YOLO", YOLO_RESULTS_CSV))
    if yolo_history:
        epochs = [entry["epoch"] for entry in yolo_history]
        losses = [entry.get("loss", float("nan")) for entry in yolo_history]
        ax.plot(epochs, losses, marker="o", linewidth=1.8, label="YOLO")

    ax.set_title("Training Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss_overall.png", dpi=200)
    plt.close(fig)


def plot_individual_losses() -> None:
    for name, path in MODEL_HISTORY_FILES.items():
        history = limit_history(load_series(name, path))
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        losses = [entry.get("loss", float("nan")) for entry in history]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, losses, marker="o", color="tab:blue", linewidth=1.8)
        ax.set_title(f"{name} Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"training_loss_{name.lower().replace(' ', '_')}.png", dpi=200)
        plt.close(fig)

    yolo_history = limit_history(load_series("YOLO", YOLO_RESULTS_CSV))
    if yolo_history:
        epochs = [entry["epoch"] for entry in yolo_history]
        losses = [entry.get("loss", float("nan")) for entry in yolo_history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, losses, marker="o", color="tab:blue", linewidth=1.8)
        ax.set_title("YOLO Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "training_loss_yolo.png", dpi=200)
        plt.close(fig)


def main() -> None:
    plot_overall_loss()
    plot_individual_losses()
    print(f"Saved training loss plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
