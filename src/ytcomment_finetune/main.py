from __future__ import annotations

from typing import Sequence

from ytcomment_finetune.config import load_config
from ytcomment_finetune.trainer import run_training


def main(argv: Sequence[str] | None = None) -> None:
    config = load_config(argv)
    run_training(config)

