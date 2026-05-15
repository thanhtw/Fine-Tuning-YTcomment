# Research Report: Corpus Lexicon Baseline

## 1. Experimental Setup

- Model: `Corpus Lexicon Baseline`
- Data file: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/Data/FinalYTcomment.xlsx`
- Random seed: `42`
- Train/test split: `4751` / `1188`
- Epochs: `10`
- Batch size: `8`
- Max token length: `256`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Best checkpoint: `N/A`
- Best selection metric: `N/A`

## 2. Dataset Summary

- Total examples: `5939`
- Full label distribution: `{'negative': 1327, 'neutral': 1886, 'positive': 2726}`
- Train label distribution: `{'negative': 1061, 'neutral': 1509, 'positive': 2181}`
- Test label distribution: `{'negative': 266, 'neutral': 377, 'positive': 545}`
- Character length summary (all): `{'count': 5939, 'min': 1, 'max': 3049, 'mean': 38.5553, 'median': 17}`
- Representation summary (train): `{'count': 4751, 'note': 'No training stage for corpus baseline.'}`
- Representation summary (test): `{'match_count': {'count': 1188, 'min': 0, 'max': 360, 'mean': 4.5488, 'median': 2.0}, 'absolute_score': {'count': 1188, 'min': 0.0, 'max': 12.100717, 'mean': 0.616, 'median': 0.3697}}`

## 3. Overall Test Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.5354 |
| Macro Precision | 0.5123 |
| Macro Recall | 0.4957 |
| Macro F1 | 0.49 |
| Weighted Precision | 0.5119 |
| Weighted Recall | 0.5354 |
| Weighted F1 | 0.5084 |

## 4. Per-Class Performance

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.5955 | 0.4925 | 0.5391 | 266 |
| neutral | 0.3773 | 0.2202 | 0.2781 | 377 |
| positive | 0.5642 | 0.7743 | 0.6527 | 545 |

## 5. Confusion Matrix

| True / Pred | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 131 | 31 | 104 |
| neutral | 72 | 83 | 222 |
| positive | 17 | 106 | 422 |

## 6. Error Analysis Preview

1. True: `neutral`, Pred: `positive`, Confidence: 1.0
   Text: 這個時間蠻有趣的(?

2. True: `neutral`, Pred: `positive`, Confidence: 1.0
   Text: 越南其實很安全喔!

3. True: `neutral`, Pred: `negative`, Confidence: 1.0
   Text: 不客氣

4. True: `negative`, Pred: `positive`, Confidence: 1.0
   Text: IAN和Jeannie是否該 "跳出不要再一起拍片的舒適圈了"?????????????????????????????????????? 舉例來說:我想看全英文的電影~結果出現有中文的片段然後占了2/3~大家做何感想???????????? 我再舉一個例子:我跟我老公約會~結果出現第三者來當電燈泡~大家做何感想????????????????? 這頻道是DODOMEN不是DODOMEN AND WOMEN OKOKOKOKOKOKOKOKOKOKOKOKOKOKOK JENNIE可以自己搞一個頻道和PODCAST還是好好當幕後都行~真的可以減少不必要的出現~感恩感恩 我講不好聽的~其他工作人員都可以乖乖在幕後~為何要獨厚JENNIE一個人???????同家公司還要不平等對待!? 在沙咪位置做好該做的事很難?!好好想想吧~DODOMEN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 加油加油加油加油加油加油加油加油加油加油加油加油加油加油加油加油加油加油加油:") 這已經不是跳脫不跳脫舒適圈了~JENNIE已經給我感覺有點"鳩佔鵲巢"了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 忠言逆耳~真的大家一起討論和思考!!!!!!!如果只是單純要口水戰真的免了~感恩感恩:"> BTW~DO式圈的JENNIE出鏡率已經也算很高~到底為何主頻道還要有JENNIE????????!!!!!!!!!!!!!吼!!!!!!!!!!!!!

5. True: `negative`, Pred: `positive`, Confidence: 1.0
   Text: @graceliu5955 現在也再罵阿哈哈哈哈

## 7. Saved Artifacts

- `classification_report.txt`
- `metrics.json`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `misclassified_examples.csv`
- `test_predictions.csv`
