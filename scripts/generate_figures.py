from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ytcomment_finetune.comparison import load_saved_result
from ytcomment_finetune.visualization import generate_visualizations


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs"
    results = []
    candidate_methods = [
        ("corpus-lexicon-baseline", "Corpus Lexicon Baseline", "corpus"),
        ("bert-base-chinese", "bert-base-chinese", "transformer"),
        ("hfl__chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext", "transformer"),
    ]

    for folder_name, display_name, method_type in candidate_methods:
        model_dir = output_dir / folder_name
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            results.append(load_saved_result(model_dir, display_name, method_type))

    generate_visualizations(output_dir, results)
    print(f"Saved figures under: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
