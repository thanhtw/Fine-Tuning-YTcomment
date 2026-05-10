# Fine-Tuning-YTcomment

This project compares three sentiment classification approaches on YouTube comments:

1. `bert-base-chinese`
2. `hfl/chinese-roberta-wwm-ext`
3. A corpus-based lexicon baseline using `Data/opinion_word_utf8.csv`

The dataset is loaded from `Data/YTcomment.xlsx`:

- input text: `text`
- ground-truth label: `label`

The pipeline includes:

- text preprocessing
- train/test split with ratio `8:2`
- corpus-based classification
- BERT fine-tuning
- RoBERTa fine-tuning
- detailed evaluation reports
- side-by-side comparison across methods

## Project Structure

```text
.
├── Data/
│   ├── YTcomment.xlsx
│   └── opinion_word_utf8.csv
├── outputs/
├── scripts/
│   └── train.py
├── src/
│   └── ytcomment_finetune/
│       ├── comparison.py
│       ├── config.py
│       ├── data.py
│       ├── lexicon.py
│       ├── main.py
│       ├── reporting.py
│       └── trainer.py
├── .env
├── .env.example
├── environment.yml
├── requirements.txt
├── setup_env.sh
└── train_hf_models.py
```

## Data Sources

### Main Dataset

- File: `Data/YTcomment.xlsx`
- Required columns:
  - `text`
  - `label`

### Opinion Corpus

- File: `Data/opinion_word_utf8.csv`
- Fields:
  - `score`: CopeOpi numerical sentiment score
  - `Pos`: number of positive annotations
  - `Neu`: number of neutral annotations
  - `Neg`: number of negative annotations
  - `Non`: number of non-opinionated annotations
  - `Not`: number of not-a-word annotations

The corpus baseline uses these fields to score matched opinion terms in each comment and predict `Positive`, `Neutral`, or `Negative`.

## Preprocessing

Before classification, text is cleaned by:

- Unicode normalization
- URL removal
- emoji and icon removal
- whitespace normalization

This preprocessing is shared by the corpus baseline and the transformer pipeline.

## Environment Setup

Create or update the Conda environment:

```bash
bash setup_env.sh
```

Activate it:

```bash
conda activate ytcomment-hf
```

Because this machine has user-level Python packages that may interfere with Conda, run Python with:

```bash
PYTHONNOUSERSITE=1
```

## How to Run

Run the full experiment from the project root:

```bash
conda activate ytcomment-hf
PYTHONNOUSERSITE=1 python train_hf_models.py
```

This will:

1. load the Excel dataset
2. split data into train/test with ratio `80/20`
3. run the corpus baseline on the test split
4. fine-tune BERT
5. fine-tune RoBERTa
6. save detailed reports for each method
7. generate a comparison report

## Configuration

Default configuration is stored in `.env`.

Example values:

```env
DATA_PATH=Data/YTcomment.xlsx
CORPUS_PATH=Data/opinion_word_utf8.csv
OUTPUT_DIR=outputs
MODELS=bert-base-chinese,hfl/chinese-roberta-wwm-ext
MAX_LENGTH=256
BATCH_SIZE=8
EPOCHS=3
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
TEST_SIZE=0.2
SEED=42
CORPUS_POSITIVE_THRESHOLD=0.05
CORPUS_NEGATIVE_THRESHOLD=-0.05
```

### Corpus Thresholds

The corpus baseline sums the `score` values of matched opinion terms in a comment.

- If the total score is greater than `CORPUS_POSITIVE_THRESHOLD`, the comment is predicted as `Positive`.
- If the total score is lower than `CORPUS_NEGATIVE_THRESHOLD`, the comment is predicted as `Negative`.
- If the total score falls between the two thresholds, the classifier falls back to annotation vote counts from the corpus fields such as `Pos`, `Neg`, `Neu`, and `Non`.
- If the vote counts are still inconclusive, the final prediction is `Neutral`.

With the current defaults:

- `CORPUS_POSITIVE_THRESHOLD=0.05`
- `CORPUS_NEGATIVE_THRESHOLD=-0.05`

This means:

- scores above `0.05` are treated as clearly positive
- scores below `-0.05` are treated as clearly negative
- scores between `-0.05` and `0.05` are treated as ambiguous and resolved by corpus vote counts

You can tune these thresholds depending on how strict you want the lexicon baseline to be:

- smaller absolute thresholds make the classifier more sensitive
- larger absolute thresholds make it more conservative
- symmetric thresholds such as `0.1` and `-0.1` are usually easier to explain in a research paper

You can also override config from the command line. Example:

```bash
conda activate ytcomment-hf
PYTHONNOUSERSITE=1 python train_hf_models.py --epochs 5 --batch-size 16
```

Example for corpus thresholds:

```bash
conda activate ytcomment-hf
PYTHONNOUSERSITE=1 python train_hf_models.py \
  --corpus-positive-threshold 0.1 \
  --corpus-negative-threshold -0.1
```

## Outputs

### Shared Outputs

- `outputs/dataset_summary.json`
- `outputs/label_mapping.json`
- `outputs/model_comparison.csv`
- `outputs/model_comparison.md`

### Corpus Baseline Outputs

- `outputs/corpus-lexicon-baseline/research_report.md`
- `outputs/corpus-lexicon-baseline/metrics.json`
- `outputs/corpus-lexicon-baseline/per_class_metrics.csv`
- `outputs/corpus-lexicon-baseline/confusion_matrix.csv`
- `outputs/corpus-lexicon-baseline/corpus_match_analysis.csv`
- `outputs/corpus-lexicon-baseline/misclassified_examples.csv`
- `outputs/corpus-lexicon-baseline/test_predictions.csv`

### BERT Outputs

- `outputs/bert-base-chinese/research_report.md`
- `outputs/bert-base-chinese/metrics.json`
- `outputs/bert-base-chinese/per_class_metrics.csv`
- `outputs/bert-base-chinese/confusion_matrix.csv`
- `outputs/bert-base-chinese/training_history.csv`
- `outputs/bert-base-chinese/misclassified_examples.csv`
- `outputs/bert-base-chinese/test_predictions.csv`

### RoBERTa Outputs

- `outputs/hfl__chinese-roberta-wwm-ext/research_report.md`
- `outputs/hfl__chinese-roberta-wwm-ext/metrics.json`
- `outputs/hfl__chinese-roberta-wwm-ext/per_class_metrics.csv`
- `outputs/hfl__chinese-roberta-wwm-ext/confusion_matrix.csv`
- `outputs/hfl__chinese-roberta-wwm-ext/training_history.csv`
- `outputs/hfl__chinese-roberta-wwm-ext/misclassified_examples.csv`
- `outputs/hfl__chinese-roberta-wwm-ext/test_predictions.csv`

## Comparison

The final comparison is saved in:

- `outputs/model_comparison.csv`
- `outputs/model_comparison.md`

The comparison includes:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted precision
- weighted recall
- weighted F1

This makes it easier to compare:

- transformer vs corpus methods
- overall performance
- minority-class sensitivity through macro F1
- class-imbalance effects through weighted F1

## Notes for Research Use

For a fair comparison:

- run all methods from the same command
- use the same preprocessing
- use the same train/test split
- regenerate reports after changing preprocessing or thresholds

The most useful files for research writing are:

- `research_report.md` for narrative summaries
- `metrics.json` for exact metric values
- `per_class_metrics.csv` for tables
- `confusion_matrix.csv` for error distribution
- `misclassified_examples.csv` for qualitative analysis
- `model_comparison.csv` for cross-method tables

## Main Entry Points

- `train_hf_models.py`: main runner
- `scripts/train.py`: alternative runner

Both launch the same pipeline.
