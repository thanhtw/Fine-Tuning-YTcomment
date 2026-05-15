# Model Comparison

## Summary

- Best macro F1: `bert-base-chinese` (0.83)
- Best accuracy: `bert-base-chinese` (0.8291)

## Comparison Table

| Method | Type | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-base-chinese | transformer | 0.8291 | 0.8301 | 0.83 | 0.83 | 0.829 | 0.8291 | 0.8291 |
| hfl/chinese-roberta-wwm-ext | transformer | 0.8258 | 0.8298 | 0.8233 | 0.8262 | 0.8247 | 0.8258 | 0.8248 |
| Corpus Lexicon Baseline | corpus | 0.5354 | 0.5123 | 0.4957 | 0.49 | 0.5119 | 0.5354 | 0.5084 |

## Report Folders

- `bert-base-chinese`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/bert-base-chinese`
- `hfl/chinese-roberta-wwm-ext`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/hfl__chinese-roberta-wwm-ext`
- `Corpus Lexicon Baseline`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/corpus-lexicon-baseline`