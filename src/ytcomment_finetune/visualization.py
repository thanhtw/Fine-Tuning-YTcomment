from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_FORMATS = ("png", "pdf")
METHOD_COLORS = {
    "Corpus Lexicon Baseline": "#C44E52",
    "bert-base-chinese": "#4C72B0",
    "hfl/chinese-roberta-wwm-ext": "#55A868",
}


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_dataset_distribution(output_dir: Path) -> None:
    summary_path = output_dir / "dataset_summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    label_distribution = summary["label_distribution"]
    labels = list(label_distribution["all"].keys())
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [label_distribution["all"][label] for label in labels], width, label="All", color="#4C72B0")
    ax.bar(x, [label_distribution["train"][label] for label in labels], width, label="Train", color="#55A868")
    ax.bar(x + width, [label_distribution["test"][label] for label in labels], width, label="Test", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Label Distribution")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, output_dir / "figures" / "dataset_label_distribution")

    length_summary = summary["text_length_characters"]
    splits = ["all", "train", "test"]
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [length_summary[split]["mean"] for split in splits]
    medians = [length_summary[split]["median"] for split in splits]
    x = np.arange(len(splits))
    ax.bar(x - 0.18, means, 0.36, label="Mean", color="#8172B2")
    ax.bar(x + 0.18, medians, 0.36, label="Median", color="#CCB974")
    ax.set_xticks(x)
    ax.set_xticklabels([split.capitalize() for split in splits])
    ax.set_ylabel("Characters per Comment")
    ax.set_title("Comment Length Summary")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, output_dir / "figures" / "comment_length_summary")


def plot_model_comparison(output_dir: Path) -> None:
    comparison_path = output_dir / "model_comparison.csv"
    if not comparison_path.exists():
        return

    rows = _read_csv_rows(comparison_path)
    methods = [row["method"] for row in rows]
    colors = [METHOD_COLORS.get(method, "#4C72B0") for method in methods]

    metric_names = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    x = np.arange(len(methods))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, (metric_key, metric_label) in zip((-width, 0, width), metric_names):
        values = [float(row[metric_key]) for row in rows]
        ax.bar(x + offset, values, width, label=metric_label, alpha=0.9)
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, output_dir / "figures" / "model_comparison_overall")

    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, color, row in zip(methods, colors, rows):
        values = [float(row[name]) for name in metric_names]
        ax.plot(metric_labels, values, marker="o", linewidth=2, label=method, color=color)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Metric Profile by Method")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _save_figure(fig, output_dir / "figures" / "model_metric_profile")


def plot_per_class_comparison(output_dir: Path, results: list[dict[str, object]]) -> None:
    if not results:
        return

    method_tables = []
    all_labels: list[str] = []
    for result in results:
        metrics_path = Path(result["output_dir"]) / "per_class_metrics.csv"
        if not metrics_path.exists():
            continue
        rows = _read_csv_rows(metrics_path)
        label_to_f1 = {row["label"]: float(row["f1-score"]) for row in rows}
        all_labels = [row["label"] for row in rows]
        method_tables.append((result["display_name"], label_to_f1))

    if not method_tables or not all_labels:
        return

    x = np.arange(len(all_labels))
    width = 0.8 / len(method_tables)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, (method, label_to_f1) in enumerate(method_tables):
        offset = (index - (len(method_tables) - 1) / 2) * width
        values = [label_to_f1[label] for label in all_labels]
        ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=METHOD_COLORS.get(method, "#4C72B0"),
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.legend(frameon=False)
    _style_axes(ax)
    _save_figure(fig, output_dir / "figures" / "per_class_f1_comparison")


def plot_confusion_matrix_figure(model_output_dir: Path) -> None:
    matrix_path = model_output_dir / "confusion_matrix.csv"
    if not matrix_path.exists():
        return

    rows = _read_csv_rows(matrix_path)
    labels = [key for key in rows[0].keys() if key != "true/pred"]
    matrix = np.array([[float(row[label]) for label in labels] for row in rows], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Normalized Confusion Matrix\n{model_output_dir.name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, model_output_dir / "figures" / "confusion_matrix")


def plot_training_curves(model_output_dir: Path) -> None:
    history_path = model_output_dir / "training_history.csv"
    if not history_path.exists() or not history_path.read_text(encoding="utf-8").strip():
        return

    rows = _read_csv_rows(history_path)
    epochs_train = []
    train_loss = []
    epochs_eval = []
    eval_loss = []
    eval_f1 = []
    for row in rows:
        epoch = row.get("epoch")
        if not epoch:
            continue
        epoch_value = float(epoch)
        if row.get("loss"):
            epochs_train.append(epoch_value)
            train_loss.append(float(row["loss"]))
        if row.get("eval_loss"):
            epochs_eval.append(epoch_value)
            eval_loss.append(float(row["eval_loss"]))
        if row.get("eval_f1_weighted"):
            eval_f1.append(float(row["eval_f1_weighted"]))

    if not epochs_train and not epochs_eval:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if epochs_train:
        axes[0].plot(epochs_train, train_loss, marker="o", color="#4C72B0", label="Train loss")
    if epochs_eval:
        axes[0].plot(epochs_eval, eval_loss, marker="s", color="#C44E52", label="Eval loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Evaluation Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if epochs_eval and eval_f1:
        usable_epochs = epochs_eval[: len(eval_f1)]
        axes[1].plot(usable_epochs, eval_f1, marker="o", color="#55A868", label="Eval weighted F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Evaluation F1 by Epoch")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    _save_figure(fig, model_output_dir / "figures" / "training_curves")


def generate_visualizations(output_dir: Path, results: list[dict[str, object]]) -> None:
    plot_dataset_distribution(output_dir)
    plot_model_comparison(output_dir)
    plot_per_class_comparison(output_dir, results)
    for result in results:
        model_output_dir = Path(result["output_dir"])
        plot_confusion_matrix_figure(model_output_dir)
        if result["method_type"] == "transformer":
            plot_training_curves(model_output_dir)
