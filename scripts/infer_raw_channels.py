from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ytcomment_finetune.data import preprocess_text


DEFAULT_INPUT_PATH = PROJECT_ROOT / "Data" / "raw_all_channels.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "bert-base-chinese" / "model"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "bert_raw_all_channels_predictions.csv"
DEFAULT_COMPARE_PATH = PROJECT_ROOT / "outputs" / "bert_channel_sentiment_comparison.csv"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 256
PROGRESS_EVERY = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RoBERTa inference on raw YouTube comments and summarize sentiment by channel."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the raw comments CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the saved RoBERTa model directory.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path for the row-level predictions CSV.",
    )
    parser.add_argument(
        "--compare-path",
        default=str(DEFAULT_COMPARE_PATH),
        help="Path for the per-channel sentiment comparison CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of comments to score per batch.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length for model inputs.",
    )
    parser.add_argument(
        "--empty-label",
        default="neutral",
        help="Fallback label for rows whose text becomes empty after preprocessing.",
    )
    parser.add_argument(
        "--disable-dynamic-quantization",
        action="store_true",
        help="Disable CPU dynamic quantization for the model.",
    )
    return parser.parse_args()


def _normalize_id_to_label(raw_mapping: dict) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for key, value in raw_mapping.items():
        normalized[int(key)] = str(value)
    return normalized


def predict_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    id_to_label: dict[int, str],
) -> list[str]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits

    predictions = torch.argmax(logits, dim=-1).tolist()
    return [id_to_label[prediction] for prediction in predictions]


def write_compare_file(
    compare_path: Path,
    label_order: Iterable[str],
    channel_counts: dict[str, dict[str, int]],
) -> None:
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = list(label_order)

    with compare_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["channel_name", *ordered_labels, "total_comments"],
        )
        writer.writeheader()

        for channel_name in sorted(channel_counts):
            counts = channel_counts[channel_name]
            row = {"channel_name": channel_name, "total_comments": sum(counts.values())}
            for label in ordered_labels:
                row[label] = counts.get(label, 0)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)
    compare_path = Path(args.compare_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Saved model directory not found: {model_dir}")

    csv.field_size_limit(sys.maxsize)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if device.type == "cpu" and not args.disable_dynamic_quantization:
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    model.to(device)
    model.eval()

    id_to_label = _normalize_id_to_label(model.config.id2label)
    label_order = [id_to_label[index] for index in sorted(id_to_label)]
    if args.empty_label not in label_order:
        raise ValueError(
            f"empty-label must be one of {label_order}, but received: {args.empty_label}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    pending_rows: list[dict[str, str]] = []
    pending_texts: list[str] = []
    processed_rows = 0

    print(
        f"Running inference on {input_path} with device={device.type}, "
        f"batch_size={args.batch_size}, dynamic_quantization="
        f"{device.type == 'cpu' and not args.disable_dynamic_quantization}"
    )

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        required_columns = {"channel_name", "text"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        with output_path.open("w", encoding="utf-8-sig", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=["channel_name", "text", "predicted"],
            )
            writer.writeheader()

            def flush_pending() -> None:
                nonlocal processed_rows
                if not pending_rows:
                    return

                predicted_labels = predict_batch(
                    model=model,
                    tokenizer=tokenizer,
                    texts=pending_texts,
                    device=device,
                    max_length=args.max_length,
                    id_to_label=id_to_label,
                )
                for row_data, predicted_label in zip(pending_rows, predicted_labels):
                    writer.writerow(
                        {
                            "channel_name": row_data["channel_name"],
                            "text": row_data["text"],
                            "predicted": predicted_label,
                        }
                    )
                    channel_counts[row_data["channel_name"]][predicted_label] += 1
                    processed_rows += 1

                pending_rows.clear()
                pending_texts.clear()

                if processed_rows % PROGRESS_EVERY == 0:
                    print(f"Processed {processed_rows} comments...")

            for row in reader:
                channel_name = (row.get("channel_name") or "").strip()
                original_text = row.get("text") or ""
                cleaned_text = preprocess_text(original_text)

                if not channel_name:
                    channel_name = "Unknown"

                if not cleaned_text:
                    writer.writerow(
                        {
                            "channel_name": channel_name,
                            "text": original_text,
                            "predicted": args.empty_label,
                        }
                    )
                    channel_counts[channel_name][args.empty_label] += 1
                    processed_rows += 1
                    if processed_rows % PROGRESS_EVERY == 0:
                        print(f"Processed {processed_rows} comments...")
                    continue

                pending_rows.append(
                    {
                        "channel_name": channel_name,
                        "text": original_text,
                    }
                )
                pending_texts.append(cleaned_text)

                if len(pending_rows) >= args.batch_size:
                    flush_pending()

            flush_pending()

    write_compare_file(compare_path, label_order, channel_counts)
    print(f"Processed {processed_rows} comments in total.")
    print(f"Saved predictions to: {output_path}")
    print(f"Saved channel comparison to: {compare_path}")


if __name__ == "__main__":
    main()
