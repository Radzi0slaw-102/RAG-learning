from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# run AutoRAG's own evaluator for retrieval and generation quality metrics
def run_autorag_eval(config_path: str, qa_path: str, corpus_path: str, project_dir: str):
    from autorag.evaluator import Evaluator
    
    evaluator = Evaluator(qa_data_path=qa_path, corpus_data_path=corpus_path, project_dir=project_dir)
    evaluator.start_trial(config_path)


def run_model_benchmark(
    model: str,
    config_path: str,
    qa_path: str,
    corpus_path: str,
    results_dir: Path
):
    model_dir = results_dir / model.replace(":", "_").replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.time()
    run_autorag_eval(config_path, qa_path, corpus_path, str(model_dir))
    duration_s = time.time() - start
    
    result = {"model": model, "autorag_duration_s": duration_s}
    with open(model_dir / "timing.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="AutoRAG benchmark")
    parser.add_argument("--config", required=True, help="AutoRAG config YAML file")
    parser.add_argument("--qa-data", required=True, help="QA parquet path")
    parser.add_argument("--corpus", required=True, help="Corpus parquet path")
    parser.add_argument("--project-dir", required=True, help="Output directory for all runs")
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model names to benchmark")
    args = parser.parse_args()

    results_dir = Path(args.project_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for model in args.models:
        print(f"  Benchmarking {model}")
        all_results.append(run_model_benchmark(model, args.config, args.qa_data, args.corpus, results_dir))
    
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
