# attach ground-truth answers to the HotPotQA questions
from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"
HOTPOT_CSV = DATA_DIR / "_repo" / "data" / "HotPotQA Filtered Questions.csv"
OUTPUT_PATH = DATA_DIR / "hotpotqa_sample" / "questions_with_answers.json"


def load_ms_questions() -> dict[str, str]:
    with open(HOTPOT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["question_id"]: row["question_text"] for row in reader}


def load_ground_truth() -> dict[str, str]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return {row["id"]: row["answer"] for row in ds}


def main() -> None:
    ms_questions = load_ms_questions()
    ground_truth = load_ground_truth()
    
    joined = []
    missing = []
    for qid, question in ms_questions.items():
        answer = ground_truth.get(qid)
        if answer is None:
            missing.append(qid)
            continue
        joined.append({"question_id": qid, "question": question, "answer": answer})
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(joined, f, indent=2)
    
    print(f"Joined {len(joined)} questions with ground truth")
    if missing:
        print(f"{len(missing)} question_id(s) had no match in the validation split, skipped")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()