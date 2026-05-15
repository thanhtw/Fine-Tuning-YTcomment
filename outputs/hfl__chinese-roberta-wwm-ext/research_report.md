# Research Report: hfl/chinese-roberta-wwm-ext

## 1. Experimental Setup

- Model: `hfl/chinese-roberta-wwm-ext`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/FinalYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `4751` / `1188`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/hfl__chinese-roberta-wwm-ext/checkpoints/checkpoint-1782`
- Best selection metric: `0.8248`

## 2. Dataset Summary

- Total examples: `5939`
- Full label distribution: `{'negative': 1327, 'neutral': 1886, 'positive': 2726}`
- Train label distribution: `{'negative': 1061, 'neutral': 1509, 'positive': 2181}`
- Test label distribution: `{'negative': 266, 'neutral': 377, 'positive': 545}`
- Character length summary (all): `{'count': 5939, 'min': 1, 'max': 3049, 'mean': 38.5553, 'median': 17}`
- Representation summary (train): `{'count': 4751, 'min': 3, 'max': 256, 'mean': 31.3496, 'median': 18}`
- Representation summary (test): `{'count': 1188, 'min': 3, 'max': 256, 'mean': 29.6801, 'median': 17.0}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8258 |
| Macro Precision | 0.8298 |
| Macro Recall | 0.8233 |
| Macro F1 | 0.8262 |
| Weighted Precision | 0.8247 |
| Weighted Recall | 0.8258 |
| Weighted F1 | 0.8248 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.8927 | 0.8759 | 0.8843 | 266 |
| neutral | 0.7612 | 0.7188 | 0.7394 | 377 |
| positive | 0.8354 | 0.8752 | 0.8548 | 545 |

## 5. Confusion Matrix

| True / Pred | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 233 | 23 | 10 |
| neutral | 22 | 271 | 84 |
| positive | 6 | 62 | 477 |

## 6. Error Analysis Preview

1. True: `neutral`, Pred: `positive`, Confidence: 0.999078
   Text: 恭喜

2. True: `neutral`, Pred: `positive`, Confidence: 0.998728
   Text: 店家长得真好。祝愿素食情侣早生天才后代。

3. True: `neutral`, Pred: `positive`, Confidence: 0.998634
   Text: 原來、了解!太好了!以前有碰到過,但都不知道該怎麼辦...

4. True: `positive`, Pred: `neutral`, Confidence: 0.998576
   Text: 想請教rose spa那間是可以用什麼方式預約的

5. True: `positive`, Pred: `negative`, Confidence: 0.99856
   Text: 今天廣告特別多

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
