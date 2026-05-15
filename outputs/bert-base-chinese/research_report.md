# Research Report: bert-base-chinese

## 1. Experimental Setup

- Model: `bert-base-chinese`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/FinalYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `4751` / `1188`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/bert-base-chinese/checkpoints/checkpoint-2970`
- Best selection metric: `0.8291`

## 2. Dataset Summary

- Total examples: `5939`
- Full label distribution: `{'negative': 1327, 'neutral': 1886, 'positive': 2726}`
- Train label distribution: `{'negative': 1061, 'neutral': 1509, 'positive': 2181}`
- Test label distribution: `{'negative': 266, 'neutral': 377, 'positive': 545}`
- Character length summary (all): `{'count': 5939, 'min': 1, 'max': 3049, 'mean': 38.5553, 'median': 17}`
- Representation summary (train): `{'count': 4751, 'min': 3, 'max': 256, 'mean': 31.1899, 'median': 17}`
- Representation summary (test): `{'count': 1188, 'min': 3, 'max': 256, 'mean': 29.5455, 'median': 17.0}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8291 |
| Macro Precision | 0.8301 |
| Macro Recall | 0.83 |
| Macro F1 | 0.83 |
| Weighted Precision | 0.829 |
| Weighted Recall | 0.8291 |
| Weighted F1 | 0.8291 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.8797 | 0.8797 | 0.8797 | 266 |
| neutral | 0.7553 | 0.7533 | 0.7543 | 377 |
| positive | 0.8553 | 0.8569 | 0.8561 | 545 |

## 5. Confusion Matrix

| True / Pred | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 234 | 25 | 7 |
| neutral | 21 | 284 | 72 |
| positive | 11 | 67 | 467 |

## 6. Error Analysis Preview

1. True: `positive`, Pred: `neutral`, Confidence: 0.999782
   Text: 想請教rose spa那間是可以用什麼方式預約的

2. True: `positive`, Pred: `neutral`, Confidence: 0.99973
   Text: Luke 同學怎麼有點貌似金宣虎

3. True: `neutral`, Pred: `positive`, Confidence: 0.999706
   Text: 耶!歡迎來我家鄉~推坑成功就是開心!

4. True: `positive`, Pred: `neutral`, Confidence: 0.999706
   Text: 為什麼這麼害怕洋叫妳老婆?

5. True: `neutral`, Pred: `positive`, Confidence: 0.999666
   Text: 不但喜欢你的影片,还喜欢你的穿搭。以后可以分享一下吗?眼镜和帽子都超好看。

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
