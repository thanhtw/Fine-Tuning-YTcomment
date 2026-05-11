# Research Report: bert-base-chinese

## 1. Experimental Setup

- Model: `bert-base-chinese`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/NewYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `2329` / `583`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/bert-base-chinese/checkpoints/checkpoint-292`
- Best selection metric: `0.8138`

## 2. Dataset Summary

- Total examples: `2912`
- Full label distribution: `{'Negative': 83, 'Neutral': 1103, 'Positive': 1726}`
- Train label distribution: `{'Negative': 66, 'Neutral': 882, 'Positive': 1381}`
- Test label distribution: `{'Negative': 17, 'Neutral': 221, 'Positive': 345}`
- Character length summary (all): `{'count': 2912, 'min': 1, 'max': 2999, 'mean': 29.6696, 'median': 15.0}`
- Representation summary (train): `{'count': 2329, 'min': 3, 'max': 256, 'mean': 26.6299, 'median': 16}`
- Representation summary (test): `{'count': 583, 'min': 3, 'max': 256, 'mean': 25.2607, 'median': 16}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8268 |
| Macro Precision | 0.5461 |
| Macro Recall | 0.5589 |
| Macro F1 | 0.5523 |
| Weighted Precision | 0.8017 |
| Weighted Recall | 0.8268 |
| Weighted F1 | 0.8138 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Negative | 0.0 | 0.0 | 0.0 | 17 |
| Neutral | 0.789 | 0.7783 | 0.7836 | 221 |
| Positive | 0.8493 | 0.8986 | 0.8732 | 345 |

## 5. Confusion Matrix

| True / Pred | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Negative | 0 | 11 | 6 |
| Neutral | 0 | 172 | 49 |
| Positive | 0 | 35 | 310 |

## 6. Error Analysis Preview

1. True: `Neutral`, Pred: `Positive`, Confidence: 0.981555
   Text: 辛苦了!上帝同在

2. True: `Neutral`, Pred: `Positive`, Confidence: 0.979671
   Text: you gave up too early, Pat !

3. True: `Neutral`, Pred: `Positive`, Confidence: 0.97869
   Text: 我買了 期待收到

4. True: `Neutral`, Pred: `Positive`, Confidence: 0.976944
   Text: 免簽國真的很棒!

5. True: `Neutral`, Pred: `Positive`, Confidence: 0.974291
   Text: 導演你真的很棒!

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
