# Model Comparison

## Summary

- Best macro F1: `hfl/chinese-roberta-wwm-ext` (0.6276)
- Best accuracy: `bert-base-chinese` (0.8268)

## Comparison Table

| Method | Type | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hfl/chinese-roberta-wwm-ext | transformer | 0.8199 | 0.6903 | 0.6034 | 0.6276 | 0.812 | 0.8199 | 0.812 |
| bert-base-chinese | transformer | 0.8268 | 0.5461 | 0.5589 | 0.5523 | 0.8017 | 0.8268 | 0.8138 |
| Corpus Lexicon Baseline | corpus | 0.5729 | 0.4034 | 0.4276 | 0.3906 | 0.5709 | 0.5729 | 0.5553 |

## Report Folders

- `hfl/chinese-roberta-wwm-ext`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/hfl__chinese-roberta-wwm-ext`
- `bert-base-chinese`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/bert-base-chinese`
- `Corpus Lexicon Baseline`: `/home/selab/Desktop/Thomas/Fine-Tuning-YTcomment/outputs/corpus-lexicon-baseline`