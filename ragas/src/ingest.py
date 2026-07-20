"""
Download the PubMedQA labeled subset (PQA-L, 1000 expert-annotated
question/context/answer instances) directly from the official
pubmedqa/pubmedqa GitHub repo, and convert it into a flat JSONL of
question/ground_truth records plus a plain-text corpus for indexing.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

SOURCE_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_OUT = OUT_DIR / "corpus"
QA_OUT = OUT_DIR / "qa_pairs.jsonl"


def build_corpus_and_qa() -> None:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    records = response.json()

    CORPUS_OUT.mkdir(parents=True, exist_ok=True)
    with open(QA_OUT, "w", encoding="utf-8") as qa_file:
        for pmid, record in records.items():
            context_text = "\n".join(record["CONTEXTS"])

            with open(CORPUS_OUT / f"{pmid}.json", "w", encoding="utf-8") as f:
                json.dump({"id": pmid, "text": context_text}, f, ensure_ascii=False)

            qa_record = {
                "question": record["QUESTION"],
                "ground_truth": record["LONG_ANSWER"],
                "doc_id": pmid,
                "final_decision": record["final_decision"],
            }
            qa_file.write(json.dumps(qa_record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} documents to {CORPUS_OUT} and QA pairs to {QA_OUT}")


if __name__ == "__main__":
    build_corpus_and_qa()