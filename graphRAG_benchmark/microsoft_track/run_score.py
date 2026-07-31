# index the HotPotQA sample with Microsoft's graphrag, answer each question
# score answers against ground truth with ROUGE-L
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rouge_score import rouge_scorer

DATA_DIR = Path(__file__).parent / "data"
WORKSPACE = DATA_DIR / "hotpotqa_sample"
QUESTIONS_PATH = WORKSPACE / "questions_with_answers.json"
RESULTS_PATH = WORKSPACE / "results.json"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "llama3.1:70b"


def write_graphrag_settings() -> None:
    # point graphrag's settings.yaml at the local Ollama server instead of OpenAI 
    settings_path = WORKSPACE / "settings.yaml"
    settings_path.write_text(f"""\
models:
  default_chat_model:
    type: openai_chat
    api_base: {OLLAMA_BASE_URL}
    api_key: ollama
    model: {LLM_MODEL}
    model_supports_json: true
  default_embedding_model:
    type: openai_embedding
    api_base: {OLLAMA_BASE_URL}
    api_key: ollama
    model: nomic-embed-text
""", encoding="utf-8")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    # run a graphrag cli command with full stdout/stderr on failure
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(args)}")
        print(f"--- stdout ---\n{result.stdout}")
        print(f"--- stderr ---\n{result.stderr}")
        result.check_returncode()
    return result
    

def run_indexing() -> None:
    output_dir = WORKSPACE / "output"
    settings_path = WORKSPACE / "settings.yaml"
    
    if not settings_path.exists():
        _run(["graphrag", "init", "--root", str(WORKSPACE)])
    write_graphrag_settings()
    _run(["graphrag", "index", "--root", str(WORKSPACE)])


def query(question: str, method: str = "local") -> str:
    result = _run(["graphrag", "query", "--root", str(WORKSPACE), "--method", method, "--query", question])
    return result.stdout.strip()


def score(answer: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ground_truth, answer)["rougeL"].fmeasure


def main() -> None:
    run_indexing()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    results = []
    for entry in questions:
        answer = query(entry["question"])
        rouge_l = score(answer, entry["answer"])
        results.append({
            "question_id": entry["question_id"],
            "question": entry["question"],
            "ground_truth": entry["answer"],
            "system_answer": answer,
            "rouge_l": rouge_l,
        })
        print(f"[{entry['question_id']}] ROUGE-L={rouge_l:.3f}")

    avg_rouge_l = sum(r["rouge_l"] for r in results) / len(results) if results else 0.0
    RESULTS_PATH.write_text(json.dumps({"avg_rouge_l": avg_rouge_l, "results": results}, indent=2), encoding="utf-8")
    print(f"\nAverage ROUGE-L: {avg_rouge_l:.3f}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()