from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from ytcomment_finetune.config import AppConfig
from ytcomment_finetune.reporting import save_report, slugify_model_name, summarize_numeric


@dataclass
class CorpusEntry:
    text: str
    score: float
    pos: int
    neu: int
    neg: int
    non: int
    not_word: int


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.entry: CorpusEntry | None = None


class CorpusMatcher:
    def __init__(self, entries: list[CorpusEntry]) -> None:
        self.root = TrieNode()
        for entry in entries:
            node = self.root
            for char in entry.text:
                node = node.children.setdefault(char, TrieNode())
            node.entry = entry

    def find_matches(self, text: str) -> list[CorpusEntry]:
        matches: list[CorpusEntry] = []
        index = 0
        while index < len(text):
            node = self.root
            cursor = index
            longest_entry: CorpusEntry | None = None
            longest_end = index
            while cursor < len(text) and text[cursor] in node.children:
                node = node.children[text[cursor]]
                cursor += 1
                if node.entry is not None:
                    longest_entry = node.entry
                    longest_end = cursor
            if longest_entry is not None:
                matches.append(longest_entry)
                index = longest_end
            else:
                index += 1
        return matches


def load_corpus_entries(corpus_path: Path) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    with corpus_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            entries.append(
                CorpusEntry(
                    text=text,
                    score=float(row.get("score") or 0.0),
                    pos=int(row.get("Pos") or 0),
                    neu=int(row.get("Neu") or 0),
                    neg=int(row.get("Neg") or 0),
                    non=int(row.get("Non") or 0),
                    not_word=int(row.get("Not") or 0),
                )
            )
    if not entries:
        raise ValueError(f"No usable corpus rows found in {corpus_path}.")
    return entries


def classify_with_corpus(
    text: str,
    matcher: CorpusMatcher,
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[str, float, dict[str, object]]:
    matches = matcher.find_matches(text)
    if not matches:
        return (
            "Neutral",
            0.0,
            {
                "matches": [],
                "match_count": 0,
                "total_score": 0.0,
                "pos_votes": 0,
                "neg_votes": 0,
                "neu_votes": 0,
                "invalid_votes": 0,
            },
        )

    total_score = sum(entry.score for entry in matches)
    pos_votes = sum(entry.pos for entry in matches)
    neg_votes = sum(entry.neg for entry in matches)
    neu_votes = sum(entry.neu + entry.non for entry in matches)
    invalid_votes = sum(entry.not_word for entry in matches)

    if total_score > positive_threshold:
        label = "Positive"
    elif total_score < negative_threshold:
        label = "Negative"
    elif pos_votes > neg_votes and pos_votes > neu_votes:
        label = "Positive"
    elif neg_votes > pos_votes and neg_votes > neu_votes:
        label = "Negative"
    else:
        label = "Neutral"

    vote_total = pos_votes + neg_votes + neu_votes + invalid_votes
    vote_confidence = max(pos_votes, neg_votes, neu_votes, invalid_votes) / vote_total if vote_total else 0.0
    score_confidence = min(abs(total_score), 1.0)
    confidence = max(vote_confidence, score_confidence)

    analysis = {
        "matches": [entry.text for entry in matches],
        "match_count": len(matches),
        "total_score": round(total_score, 6),
        "pos_votes": pos_votes,
        "neg_votes": neg_votes,
        "neu_votes": neu_votes,
        "invalid_votes": invalid_votes,
    }
    return label, confidence, analysis


def save_corpus_match_analysis(
    output_dir: Path,
    texts: list[str],
    true_ids: list[int],
    pred_ids: list[int],
    confidences: list[float],
    analyses: list[dict[str, object]],
    id_to_label: dict[int, str],
) -> None:
    analysis_path = output_dir / "corpus_match_analysis.csv"
    with analysis_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "text",
                "true_label",
                "predicted_label",
                "prediction_confidence",
                "match_count",
                "total_score",
                "pos_votes",
                "neg_votes",
                "neu_votes",
                "invalid_votes",
                "matched_terms",
            ],
        )
        writer.writeheader()
        for text, true_id, pred_id, confidence, analysis in zip(
            texts,
            true_ids,
            pred_ids,
            confidences,
            analyses,
        ):
            writer.writerow(
                {
                    "text": text,
                    "true_label": id_to_label[true_id],
                    "predicted_label": id_to_label[pred_id],
                    "prediction_confidence": round(confidence, 6),
                    "match_count": analysis["match_count"],
                    "total_score": analysis["total_score"],
                    "pos_votes": analysis["pos_votes"],
                    "neg_votes": analysis["neg_votes"],
                    "neu_votes": analysis["neu_votes"],
                    "invalid_votes": analysis["invalid_votes"],
                    "matched_terms": " | ".join(analysis["matches"]),
                }
            )


def run_corpus_baseline(
    dataset,
    label_encoder,
    dataset_summary: dict[str, object],
    config: AppConfig,
) -> dict[str, object]:
    entries = load_corpus_entries(config.corpus_path)
    matcher = CorpusMatcher(entries)
    test_texts = dataset["test"]["text"]
    true_ids = dataset["test"]["labels"]

    pred_labels: list[str] = []
    confidences: list[float] = []
    analyses: list[dict[str, object]] = []
    for text in test_texts:
        label, confidence, analysis = classify_with_corpus(
            text,
            matcher,
            config.corpus_positive_threshold,
            config.corpus_negative_threshold,
        )
        pred_labels.append(label)
        confidences.append(confidence)
        analyses.append(analysis)

    pred_ids = [label_encoder.label_to_id[label] for label in pred_labels]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_ids, pred_ids, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_ids, pred_ids, average="macro", zero_division=0
    )
    accuracy = accuracy_score(true_ids, pred_ids)
    report = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(label_encoder.id_to_label))),
        target_names=[label_encoder.id_to_label[idx] for idx in sorted(label_encoder.id_to_label)],
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "support_total": report["macro avg"]["support"],
    }

    output_dir = config.output_dir / slugify_model_name("corpus-lexicon-baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    match_counts = [analysis["match_count"] for analysis in analyses]
    match_score_abs = [abs(float(analysis["total_score"])) for analysis in analyses]
    representation_summary = {
        "train": {"count": len(dataset["train"]), "note": "No training stage for corpus baseline."},
        "test": {
            "match_count": summarize_numeric(match_counts),
            "absolute_score": summarize_numeric(match_score_abs),
        },
    }

    config_summary = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    config_summary["corpus_entry_count"] = len(entries)
    config_summary["corpus_fields"] = ["score", "Pos", "Neu", "Neg", "Non", "Not"]

    save_report(
        output_dir=output_dir,
        model_name="Corpus Lexicon Baseline",
        test_texts=test_texts,
        true_ids=true_ids,
        pred_ids=pred_ids,
        confidences=confidences,
        id_to_label=label_encoder.id_to_label,
        metrics=metrics,
        dataset_summary=dataset_summary,
        config_summary=config_summary,
        token_length_summary=representation_summary,
        training_history=[],
        best_checkpoint=None,
        best_metric=None,
    )
    save_corpus_match_analysis(
        output_dir,
        test_texts,
        true_ids,
        pred_ids,
        confidences,
        analyses,
        label_encoder.id_to_label,
    )
    (output_dir / "corpus_summary.json").write_text(
        json.dumps(
            {
                "corpus_path": str(config.corpus_path),
                "entry_count": len(entries),
                "positive_threshold": config.corpus_positive_threshold,
                "negative_threshold": config.corpus_negative_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "display_name": "Corpus Lexicon Baseline",
        "method_type": "corpus",
        "metrics": metrics,
        "output_dir": output_dir,
    }
