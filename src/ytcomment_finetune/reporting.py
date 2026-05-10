from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def summarize_numeric(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 0:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    else:
        median = ordered[mid]

    return {
        "count": count,
        "min": min(ordered),
        "max": max(ordered),
        "mean": round(sum(ordered) / count, 4),
        "median": round(median, 4),
    }


def count_label_ids(label_ids: list[int], id_to_label: dict[int, str]) -> dict[str, int]:
    counts = {label: 0 for label in id_to_label.values()}
    for label_id in label_ids:
        counts[id_to_label[label_id]] += 1
    return counts


def build_dataset_summary(
    all_texts: list[str],
    train_texts: list[str],
    test_texts: list[str],
    all_label_ids: list[int],
    train_label_ids: list[int],
    test_label_ids: list[int],
    id_to_label: dict[int, str],
) -> dict[str, object]:
    return {
        "total_examples": len(all_texts),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "split_ratio": {
            "train": round(len(train_texts) / len(all_texts), 4) if all_texts else 0.0,
            "test": round(len(test_texts) / len(all_texts), 4) if all_texts else 0.0,
        },
        "label_distribution": {
            "all": count_label_ids(all_label_ids, id_to_label),
            "train": count_label_ids(train_label_ids, id_to_label),
            "test": count_label_ids(test_label_ids, id_to_label),
        },
        "text_length_characters": {
            "all": summarize_numeric([len(text) for text in all_texts]),
            "train": summarize_numeric([len(text) for text in train_texts]),
            "test": summarize_numeric([len(text) for text in test_texts]),
        },
    }


def save_predictions(
    output_dir: Path,
    texts: list[str],
    true_ids: list[int],
    pred_ids: list[int],
    confidences: list[float],
    id_to_label: dict[int, str],
) -> None:
    predictions_path = output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["text", "true_label", "predicted_label", "prediction_confidence"],
        )
        writer.writeheader()
        for text, true_id, pred_id, confidence in zip(texts, true_ids, pred_ids, confidences):
            writer.writerow(
                {
                    "text": text,
                    "true_label": id_to_label[true_id],
                    "predicted_label": id_to_label[pred_id],
                    "prediction_confidence": round(confidence, 6),
                }
            )


def save_training_history(output_dir: Path, training_history: list[dict[str, object]]) -> None:
    history_path = output_dir / "training_history.csv"
    if not training_history:
        history_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in training_history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in training_history:
            writer.writerow(row)


def save_confusion_matrix_csv(
    output_dir: Path,
    true_ids: list[int],
    pred_ids: list[int],
    ordered_labels: list[str],
) -> list[list[int]]:
    matrix = confusion_matrix(
        true_ids,
        pred_ids,
        labels=list(range(len(ordered_labels))),
    ).tolist()

    matrix_path = output_dir / "confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + ordered_labels)
        for label_name, row in zip(ordered_labels, matrix):
            writer.writerow([label_name] + row)

    return matrix


def save_per_class_metrics_csv(
    output_dir: Path,
    ordered_labels: list[str],
    report_dict: dict[str, object],
) -> None:
    metrics_path = output_dir / "per_class_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["label", "precision", "recall", "f1-score", "support"],
        )
        writer.writeheader()
        for label in ordered_labels:
            row = report_dict[label]
            writer.writerow(
                {
                    "label": label,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1-score": row["f1-score"],
                    "support": row["support"],
                }
            )


def save_misclassified_examples(
    output_dir: Path,
    texts: list[str],
    true_ids: list[int],
    pred_ids: list[int],
    confidences: list[float],
    id_to_label: dict[int, str],
    limit: int = 50,
) -> list[dict[str, object]]:
    mistakes = []
    for text, true_id, pred_id, confidence in zip(texts, true_ids, pred_ids, confidences):
        if true_id == pred_id:
            continue
        mistakes.append(
            {
                "text": text,
                "true_label": id_to_label[true_id],
                "predicted_label": id_to_label[pred_id],
                "prediction_confidence": round(confidence, 6),
            }
        )

    mistakes.sort(key=lambda item: item["prediction_confidence"], reverse=True)
    selected = mistakes[:limit]

    mistakes_path = output_dir / "misclassified_examples.csv"
    with mistakes_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["text", "true_label", "predicted_label", "prediction_confidence"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    return selected


def _format_markdown_table(rows: list[list[object]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(str(cell) for cell in rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body)


def save_markdown_report(
    output_dir: Path,
    model_name: str,
    metrics: dict[str, float],
    dataset_summary: dict[str, object],
    config_summary: dict[str, object],
    token_length_summary: dict[str, object],
    ordered_labels: list[str],
    report_dict: dict[str, object],
    confusion: list[list[int]],
    best_checkpoint: str | None,
    best_metric: float | None,
    mistake_examples: list[dict[str, object]],
) -> None:
    overall_rows = [
        ["Metric", "Value"],
        ["Accuracy", round(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0)), 4)],
        ["Macro Precision", round(metrics.get("eval_precision_macro", metrics.get("precision_macro", 0.0)), 4)],
        ["Macro Recall", round(metrics.get("eval_recall_macro", metrics.get("recall_macro", 0.0)), 4)],
        ["Macro F1", round(metrics.get("eval_f1_macro", metrics.get("f1_macro", 0.0)), 4)],
        ["Weighted Precision", round(metrics.get("eval_precision_weighted", metrics.get("precision_weighted", 0.0)), 4)],
        ["Weighted Recall", round(metrics.get("eval_recall_weighted", metrics.get("recall_weighted", 0.0)), 4)],
        ["Weighted F1", round(metrics.get("eval_f1_weighted", metrics.get("f1_weighted", 0.0)), 4)],
    ]

    per_class_rows = [["Label", "Precision", "Recall", "F1", "Support"]]
    for label in ordered_labels:
        row = report_dict[label]
        per_class_rows.append(
            [
                label,
                round(row["precision"], 4),
                round(row["recall"], 4),
                round(row["f1-score"], 4),
                int(row["support"]),
            ]
        )

    confusion_rows = [["True / Pred"] + ordered_labels]
    for label, row in zip(ordered_labels, confusion):
        confusion_rows.append([label] + row)

    error_preview = mistake_examples[:5]
    error_lines = []
    for index, item in enumerate(error_preview, start=1):
        error_lines.append(
            f"{index}. True: `{item['true_label']}`, Pred: `{item['predicted_label']}`, "
            f"Confidence: {item['prediction_confidence']}\n"
            f"   Text: {item['text']}"
        )
    error_section = "\n\n".join(error_lines) if error_lines else "No misclassified examples in the saved preview."

    markdown = f"""# Research Report: {model_name}

## 1. Experimental Setup

- Model: `{model_name}`
- Data file: `{config_summary['data_path']}`
- Random seed: `{config_summary['seed']}`
- Train/test split: `{dataset_summary['train_size']}` / `{dataset_summary['test_size']}`
- Epochs: `{config_summary['epochs']}`
- Batch size: `{config_summary['batch_size']}`
- Max token length: `{config_summary['max_length']}`
- Learning rate: `{config_summary['learning_rate']}`
- Weight decay: `{config_summary['weight_decay']}`
- Best checkpoint: `{best_checkpoint or 'N/A'}`
- Best selection metric: `{round(best_metric, 4) if best_metric is not None else 'N/A'}`

## 2. Dataset Summary

- Total examples: `{dataset_summary['total_examples']}`
- Full label distribution: `{dataset_summary['label_distribution']['all']}`
- Train label distribution: `{dataset_summary['label_distribution']['train']}`
- Test label distribution: `{dataset_summary['label_distribution']['test']}`
- Character length summary (all): `{dataset_summary['text_length_characters']['all']}`
- Representation summary (train): `{token_length_summary['train']}`
- Representation summary (test): `{token_length_summary['test']}`

## 3. Overall Test Metrics

{_format_markdown_table(overall_rows)}

## 4. Per-Class Performance

{_format_markdown_table(per_class_rows)}

## 5. Confusion Matrix

{_format_markdown_table(confusion_rows)}

## 6. Error Analysis Preview

{error_section}

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
"""
    (output_dir / "research_report.md").write_text(markdown, encoding="utf-8")


def save_report(
    output_dir: Path,
    model_name: str,
    test_texts: list[str],
    true_ids: list[int],
    pred_ids: list[int],
    confidences: list[float],
    id_to_label: dict[int, str],
    metrics: dict[str, float],
    dataset_summary: dict[str, object],
    config_summary: dict[str, object],
    token_length_summary: dict[str, object],
    training_history: list[dict[str, object]],
    best_checkpoint: str | None,
    best_metric: float | None,
) -> None:
    ordered_labels = [id_to_label[idx] for idx in sorted(id_to_label)]
    report_text = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(ordered_labels))),
        target_names=ordered_labels,
        zero_division=0,
    )
    report_dict = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(ordered_labels))),
        target_names=ordered_labels,
        zero_division=0,
        output_dict=True,
    )

    metadata = {
        "model_name": model_name,
        "train_size": dataset_summary["train_size"],
        "test_size": dataset_summary["test_size"],
        "metrics": metrics,
        "labels": id_to_label,
        "dataset_summary": dataset_summary,
        "config": config_summary,
        "token_length_summary": token_length_summary,
        "best_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "classification_report": report_dict,
    }

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_per_class_metrics_csv(output_dir, ordered_labels, report_dict)
    confusion = save_confusion_matrix_csv(output_dir, true_ids, pred_ids, ordered_labels)
    save_training_history(output_dir, training_history)
    mistake_examples = save_misclassified_examples(
        output_dir,
        test_texts,
        true_ids,
        pred_ids,
        confidences,
        id_to_label,
    )
    save_predictions(output_dir, test_texts, true_ids, pred_ids, confidences, id_to_label)
    save_markdown_report(
        output_dir=output_dir,
        model_name=model_name,
        metrics=metrics,
        dataset_summary=dataset_summary,
        config_summary=config_summary,
        token_length_summary=token_length_summary,
        ordered_labels=ordered_labels,
        report_dict=report_dict,
        confusion=confusion,
        best_checkpoint=best_checkpoint,
        best_metric=best_metric,
        mistake_examples=mistake_examples,
    )
    print(f"\nModel: {model_name}")
    print(f"Train/Test: {dataset_summary['train_size']}/{dataset_summary['test_size']}")
    print(report_text)
