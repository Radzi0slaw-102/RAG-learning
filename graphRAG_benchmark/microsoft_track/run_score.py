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
EMBED_MODEL = "qwen3-embedding:8b"


def write_graphrag_settings() -> None:
    settings_path = WORKSPACE / "settings.yaml"
    settings_path.write_text(f"""\
completion_models:
  default_completion_model:
    model_provider: openai
    model: {LLM_MODEL}
    auth_method: api_key
    api_key: ollama
    api_base: {OLLAMA_BASE_URL}
    call_args:
      timeout: 1800

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: {EMBED_MODEL}
    auth_method: api_key
    api_key: ollama
    api_base: {OLLAMA_BASE_URL}
    call_args:
      timeout: 1800

input:
  type: text

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

cluster_graph:
  max_cluster_size: 10

extract_claims:
  enabled: false
  completion_model_id: default_completion_model
  prompt: "prompts/extract_claims.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 1

community_reports:
  completion_model_id: default_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
""", encoding="utf-8")


def _run_streaming(args: list[str]) -> None:
    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args)


def _run_captured(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(args)}")
        print(f"--- stdout ---\n{result.stdout}")
        print(f"--- stderr ---\n{result.stderr}")
        result.check_returncode()
    return result


def run_indexing() -> None:
    output_dir = WORKSPACE / "output"
    if output_dir.exists() and any(output_dir.iterdir()):
        return
    
    _run_streaming(["graphrag", "init", "--root", str(WORKSPACE), "--force",
                     "--model", LLM_MODEL, "--embedding", EMBED_MODEL])
    write_graphrag_settings()
    _run_streaming(["graphrag", "index", "--root", str(WORKSPACE), "--verbose"])


def query(question: str, method: str = "local") -> str:
    result = _run_captured(["graphrag", "query", "--root", str(WORKSPACE), "--method", method, "--query", question])
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