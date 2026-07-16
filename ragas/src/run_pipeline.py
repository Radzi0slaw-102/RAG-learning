"""
Run the RAG pipeline over the QA pairs for every configured Ollama model,
recording question, retrieved contexts, generated answer, and ground
truth to results/raw/{model}.jsonl for later ragas evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline import set_index, get_query_engine, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = PROJECT_ROOT / "data" / "qa_pairs.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"/ "raw"


def load_qa_pairs(max_pairs: int | None = None) -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    return pairs[:max_pairs] if max_pairs is not None else pairs


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_")


def run_for_model(index, model_name: str, config: dict, qa_pairs: list[dict]) -> None:
    engine = get_query_engine(index, model_name, config)
    out_path = RESULTS_DIR / f"{sanitize_model_name(model_name)}.jsonl"

    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, qa in enumerate(qa_pairs, start=1):
            response = engine.query(qa["question"])
            contexts = [node.get_content() for node in response.source_nodes]

            record = {
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "contexts": contexts,
                "answer": str(response),
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{model_name}] {i}/{len(qa_pairs)}")


def main() -> None:
    config = load_config()
    index = set_index(config)
    qa_pairs = load_qa_pairs(max_pairs=config.get("max_qa_pairs"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in config["llm_models"]:
        run_for_model(index, model["name"], config, qa_pairs)


if __name__ == "__main__":
    main()