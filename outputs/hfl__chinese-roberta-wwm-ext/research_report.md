# Research Report: hfl/chinese-roberta-wwm-ext

## 1. Experimental Setup

- Model: `hfl/chinese-roberta-wwm-ext`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/NewYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `2329` / `583`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/hfl__chinese-roberta-wwm-ext/checkpoints/checkpoint-1460`
- Best selection metric: `0.812`

## 2. Dataset Summary

- Total examples: `2912`
- Full label distribution: `{'Negative': 83, 'Neutral': 1103, 'Positive': 1726}`
- Train label distribution: `{'Negative': 66, 'Neutral': 882, 'Positive': 1381}`
- Test label distribution: `{'Negative': 17, 'Neutral': 221, 'Positive': 345}`
- Character length summary (all): `{'count': 2912, 'min': 1, 'max': 2999, 'mean': 29.6696, 'median': 15.0}`
- Representation summary (train): `{'count': 2329, 'min': 3, 'max': 256, 'mean': 26.7943, 'median': 16}`
- Representation summary (test): `{'count': 583, 'min': 3, 'max': 256, 'mean': 25.4151, 'median': 16}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8199 |
| Macro Precision | 0.6903 |
| Macro Recall | 0.6034 |
| Macro F1 | 0.6276 |
| Weighted Precision | 0.812 |
| Weighted Recall | 0.8199 |
| Weighted F1 | 0.812 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Negative | 0.4286 | 0.1765 | 0.25 | 17 |
| Neutral | 0.8103 | 0.7149 | 0.7596 | 221 |
| Positive | 0.832 | 0.9188 | 0.8733 | 345 |

## 5. Confusion Matrix

| True / Pred | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Negative | 3 | 9 | 5 |
| Neutral | 4 | 158 | 59 |
| Positive | 0 | 28 | 317 |

## 6. Error Analysis Preview

1. True: `Neutral`, Pred: `Positive`, Confidence: 0.999792
   Text: 我才发现好久没看你的影片

2. True: `Neutral`, Pred: `Positive`, Confidence: 0.999787
   Text: 是啊 好希望讓更多人知道這件事

3. True: `Neutral`, Pred: `Positive`, Confidence: 0.999785
   Text: 導演你真的很棒!

4. True: `Neutral`, Pred: `Positive`, Confidence: 0.99978
   Text: you gave up too early, Pat !

5. True: `Neutral`, Pred: `Positive`, Confidence: 0.999773
   Text: 刚从邦劳潜水回来几天而以,你又让我十分的想念大海了

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
