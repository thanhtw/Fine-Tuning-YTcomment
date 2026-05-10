from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ytcomment_finetune.comparison import save_model_comparison
from ytcomment_finetune.config import AppConfig
from ytcomment_finetune.data import build_dataset, build_label_encoder, load_examples
from ytcomment_finetune.lexicon import run_corpus_baseline
from ytcomment_finetune.reporting import build_dataset_summary, save_report, slugify_model_name, summarize_numeric
from ytcomment_finetune.visualization import generate_visualizations


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True, desc="Tokenizing text")
    return tokenized.remove_columns(["text"])


def compute_metrics_factory(id_to_label: dict[int, str]):
    ordered_labels = [id_to_label[idx] for idx in sorted(id_to_label)]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(
            labels,
            predictions,
            labels=list(range(len(ordered_labels))),
            target_names=ordered_labels,
            zero_division=0,
            output_dict=True,
        )
        return {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": macro_precision,
            "recall_macro": macro_recall,
            "f1_macro": macro_f1,
            "support_total": report["macro avg"]["support"],
        }

    return compute_metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def prepare_training_run(config: AppConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    set_seed(config.seed)

    examples = load_examples(str(config.data_path))
    label_encoder = build_label_encoder(examples)
    dataset = build_dataset(examples, label_encoder, config.test_size, config.seed)
    dataset_summary = build_dataset_summary(
        all_texts=[example["text"] for example in examples],
        train_texts=dataset["train"]["text"],
        test_texts=dataset["test"]["text"],
        all_label_ids=[label_encoder.label_to_id[example["label"]] for example in examples],
        train_label_ids=dataset["train"]["labels"],
        test_label_ids=dataset["test"]["labels"],
        id_to_label=label_encoder.id_to_label,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "label_mapping.json").write_text(
        json.dumps(label_encoder.label_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded {len(examples)} usable rows from {config.data_path}")
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    print(f"Labels: {label_encoder.label_to_id}")
    return dataset, label_encoder, dataset_summary


def fine_tune_model(model_name: str, dataset, label_encoder, dataset_summary, config: AppConfig) -> dict[str, object]:
    print(f"\nStarting fine-tuning for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, config.max_length)
    token_length_summary = {
        "train": summarize_numeric([len(ids) for ids in tokenized_dataset["train"]["input_ids"]]),
        "test": summarize_numeric([len(ids) for ids in tokenized_dataset["test"]["input_ids"]]),
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.label_to_id),
        id2label=label_encoder.id_to_label,
        label2id=label_encoder.label_to_id,
    )

    model_output_dir = config.output_dir / slugify_model_name(model_name)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_factory(label_encoder.id_to_label),
    )

    trainer.train()
    metrics = trainer.evaluate()
    predictions = trainer.predict(tokenized_dataset["test"])
    probabilities = softmax(predictions.predictions)
    pred_ids = np.argmax(predictions.predictions, axis=-1).tolist()
    confidences = np.max(probabilities, axis=-1).tolist()
    true_ids = predictions.label_ids.tolist()
    test_texts = dataset["test"]["text"]

    trainer.save_model(str(model_output_dir / "model"))
    tokenizer.save_pretrained(str(model_output_dir / "model"))

    serializable_metrics = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in metrics.items()
    }
    save_report(
        output_dir=model_output_dir,
        model_name=model_name,
        test_texts=test_texts,
        true_ids=true_ids,
        pred_ids=pred_ids,
        confidences=confidences,
        id_to_label=label_encoder.id_to_label,
        metrics=serializable_metrics,
        dataset_summary=dataset_summary,
        config_summary={
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        token_length_summary=token_length_summary,
        training_history=trainer.state.log_history,
        best_checkpoint=trainer.state.best_model_checkpoint,
        best_metric=trainer.state.best_metric,
    )
    return {
        "display_name": model_name,
        "method_type": "transformer",
        "metrics": serializable_metrics,
        "output_dir": model_output_dir,
    }


def run_training(config: AppConfig) -> None:
    dataset, label_encoder, dataset_summary = prepare_training_run(config)
    experiment_results = [run_corpus_baseline(dataset, label_encoder, dataset_summary, config)]
    for model_name in config.models:
        experiment_results.append(fine_tune_model(model_name, dataset, label_encoder, dataset_summary, config))
    save_model_comparison(config.output_dir, experiment_results)
    generate_visualizations(config.output_dir, experiment_results)
