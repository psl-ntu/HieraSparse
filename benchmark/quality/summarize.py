import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    "Single-Document QA",
    "Multi-Document QA",
    "Summarization",
    "Few-shot Learning",
    "Synthetic Task",
    "Code Completion",
]

CATEGORY_MAP = {
    "narrativeqa": "Single-Document QA",
    "qasper": "Single-Document QA",
    "multifieldqa_en": "Single-Document QA",
    "hotpotqa": "Multi-Document QA",
    "2wikimqa": "Multi-Document QA",
    "musique": "Multi-Document QA",
    "gov_report": "Summarization",
    "qmsum": "Summarization",
    "multi_news": "Summarization",
    "trec": "Few-shot Learning",
    "triviaqa": "Few-shot Learning",
    "samsum": "Few-shot Learning",
    "passage_count": "Synthetic Task",
    "passage_retrieval_en": "Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",
}


def summarize(input_path):

    input_path = Path(input_path)
    output_path = input_path.parent / "summarization.json"

    with open(input_path, "r") as f:
        scores = json.load(f)

    category_values = defaultdict(list)
    overall_values = []

    for dataset, value in scores.items():
        category_values[CATEGORY_MAP[dataset]].append(value)
        overall_values.append(value)

    result = {}
    result["overall"] = sum(overall_values) / len(overall_values)

    per_cat = {}
    for cat in CATEGORIES:
        vals = category_values[cat]
        per_cat[cat] = sum(vals) / len(vals) if vals else None
    result["per_category"] = per_cat

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the evaluation results JSON file")
    args = parser.parse_args()
    summarize(args.input_path)
