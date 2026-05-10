from __future__ import annotations

import csv
import json
from pathlib import Path


def metric_value(metrics: dict[str, float], primary: str, fallback: str) -> float:
    return float(metrics.get(primary, metrics.get(fallback, 0.0)))


def load_saved_result(output_dir: Path, display_name: str, method_type: str) -> dict[str, object]:
    metrics_path = output_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "display_name": display_name,
        "method_type": method_type,
        "metrics": payload["metrics"],
        "output_dir": output_dir,
    }


def save_model_comparison(output_dir: Path, results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "method": result["display_name"],
                "type": result["method_type"],
                "accuracy": round(metric_value(metrics, "eval_accuracy", "accuracy"), 4),
                "macro_precision": round(metric_value(metrics, "eval_precision_macro", "precision_macro"), 4),
                "macro_recall": round(metric_value(metrics, "eval_recall_macro", "recall_macro"), 4),
                "macro_f1": round(metric_value(metrics, "eval_f1_macro", "f1_macro"), 4),
                "weighted_precision": round(
                    metric_value(metrics, "eval_precision_weighted", "precision_weighted"), 4
                ),
                "weighted_recall": round(metric_value(metrics, "eval_recall_weighted", "recall_weighted"), 4),
                "weighted_f1": round(metric_value(metrics, "eval_f1_weighted", "f1_weighted"), 4),
                "report_dir": str(result["output_dir"]),
            }
        )

    rows.sort(key=lambda item: (item["macro_f1"], item["accuracy"]), reverse=True)

    csv_path = output_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "type",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "report_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best_macro = rows[0] if rows else None
    best_accuracy = max(rows, key=lambda item: item["accuracy"]) if rows else None

    markdown_lines = [
        "# Model Comparison",
        "",
        "## Summary",
        "",
    ]
    if best_macro:
        markdown_lines.append(
            f"- Best macro F1: `{best_macro['method']}` ({best_macro['macro_f1']})"
        )
    if best_accuracy:
        markdown_lines.append(
            f"- Best accuracy: `{best_accuracy['method']}` ({best_accuracy['accuracy']})"
        )
    markdown_lines.extend(
        [
            "",
            "## Comparison Table",
            "",
            "| Method | Type | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        markdown_lines.append(
            f"| {row['method']} | {row['type']} | {row['accuracy']} | {row['macro_precision']} | "
            f"{row['macro_recall']} | {row['macro_f1']} | {row['weighted_precision']} | "
            f"{row['weighted_recall']} | {row['weighted_f1']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Report Folders",
            "",
        ]
    )
    for row in rows:
        markdown_lines.append(f"- `{row['method']}`: `{row['report_dir']}`")

    (output_dir / "model_comparison.md").write_text("\n".join(markdown_lines), encoding="utf-8")
