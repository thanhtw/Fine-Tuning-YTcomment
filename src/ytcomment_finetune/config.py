from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_MODELS = [
    "bert-base-chinese",
    "hfl/chinese-roberta-wwm-ext",
]


@dataclass
class AppConfig:
    data_path: Path
    output_dir: Path
    models: list[str]
    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    test_size: float
    seed: int


def _split_models(raw_value: str) -> list[str]:
    return [model.strip() for model in raw_value.split(",") if model.strip()]


def _resolve_path(raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _env_value(name: str, default: str) -> str:
    return os.getenv(name, default)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT and RoBERTa models on the YT comment dataset."
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to a .env config file.")
    parser.add_argument("--data-path", help="Override the Excel file path.")
    parser.add_argument("--output-dir", help="Override the output directory.")
    parser.add_argument("--models", nargs="+", help="Override one or more Hugging Face model names.")
    parser.add_argument("--max-length", type=int, help="Override max token length.")
    parser.add_argument("--batch-size", type=int, help="Override per-device batch size.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay.")
    parser.add_argument("--test-size", type=float, help="Override test split ratio.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    return parser.parse_args(argv)


def load_config(argv: Sequence[str] | None = None) -> AppConfig:
    args = parse_args(argv)
    env_file = Path(args.env_file)
    load_dotenv(env_file if env_file.is_absolute() else PROJECT_ROOT / env_file)

    models = args.models or _split_models(_env_value("MODELS", ",".join(DEFAULT_MODELS)))
    return AppConfig(
        data_path=_resolve_path(args.data_path or _env_value("DATA_PATH", "Data/YTcomment.xlsx")),
        output_dir=_resolve_path(args.output_dir or _env_value("OUTPUT_DIR", "outputs")),
        models=models,
        max_length=args.max_length or int(_env_value("MAX_LENGTH", "256")),
        batch_size=args.batch_size or int(_env_value("BATCH_SIZE", "8")),
        epochs=args.epochs or int(_env_value("EPOCHS", "3")),
        learning_rate=args.learning_rate or float(_env_value("LEARNING_RATE", "2e-5")),
        weight_decay=args.weight_decay or float(_env_value("WEIGHT_DECAY", "0.01")),
        test_size=args.test_size or float(_env_value("TEST_SIZE", "0.2")),
        seed=args.seed or int(_env_value("SEED", "42")),
    )

