# Research Report: Corpus Lexicon Baseline

## 1. Experimental Setup

- Model: `Corpus Lexicon Baseline`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/NewYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `2329` / `583`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `N/A`
- Best selection metric: `N/A`

## 2. Dataset Summary

- Total examples: `2912`
- Full label distribution: `{'Negative': 83, 'Neutral': 1103, 'Positive': 1726}`
- Train label distribution: `{'Negative': 66, 'Neutral': 882, 'Positive': 1381}`
- Test label distribution: `{'Negative': 17, 'Neutral': 221, 'Positive': 345}`
- Character length summary (all): `{'count': 2912, 'min': 1, 'max': 2999, 'mean': 29.6696, 'median': 15.0}`
- Representation summary (train): `{'count': 2329, 'note': 'No training stage for corpus baseline.'}`
- Representation summary (test): `{'match_count': {'count': 583, 'min': 0, 'max': 151, 'mean': 3.4786, 'median': 2}, 'absolute_score': {'count': 583, 'min': 0.0, 'max': 17.997567, 'mean': 0.6379, 'median': 0.3768}}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.5729 |
| Macro Precision | 0.4034 |
| Macro Recall | 0.4276 |
| Macro F1 | 0.3906 |
| Weighted Precision | 0.5709 |
| Weighted Recall | 0.5729 |
| Weighted F1 | 0.5553 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Negative | 0.0816 | 0.2353 | 0.1212 | 17 |
| Neutral | 0.4667 | 0.2534 | 0.3284 | 221 |
| Positive | 0.6618 | 0.7942 | 0.722 | 345 |

## 5. Confusion Matrix

| True / Pred | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| Negative | 4 | 3 | 10 |
| Neutral | 35 | 56 | 130 |
| Positive | 10 | 61 | 274 |

## 6. Error Analysis Preview

1. True: `Neutral`, Pred: `Positive`, Confidence: 1.0
   Text: 黑色糖果就是「台灣八角」味同款、喜歡的人很喜歡、但不喜歡的人也很多。 瑞典酒飲不便宜,所以去酒吧 會適量的喝、午餐/套餐、也會比晚餐便宜一些、大多數人會回家自己煮才不會開銷過大。如果你們還在斯德哥摩 可以去看看室內圖書館/很漂亮的圖書館。 還有瓦薩船博物館,文化館等都很不錯。瑞典地鐵站的「每一站圖案、顏色都不一樣、不能只看一站。

2. True: `Neutral`, Pred: `Positive`, Confidence: 1.0
   Text: 太快了吧

3. True: `Neutral`, Pred: `Positive`, Confidence: 1.0
   Text: 我盡量

4. True: `Neutral`, Pred: `Positive`, Confidence: 1.0
   Text: 崇崇跟肚子小粉紅跟詹 感覺就是整個畫面會很亂的感覺

5. True: `Negative`, Pred: `Positive`, Confidence: 1.0
   Text: 畫面調整很煩...

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
