from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
import yaml

from system_monitor import SystemMonitor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# find every llama_index_llm generator module in config with model name
def extract_generator_modules(config_data: dict) -> list[dict]:
    modules = []
    for node_line in config_data.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("node_type") == "generator":
                for module in node.get("modules", []):
                    if module.get("module_type") == "llama_index_llm" and "model" in module:
                        modules.append(module["model"])
    return modules

# run AutoRAG's own evaluator for retrieval and generation quality metrics
def run_autorag_eval(config_path: str, qa_path: str, corpus_path: str, project_dir: str):
    from autorag.evaluator import Evaluator
    
    evaluator = Evaluator(qa_data_path=qa_path, corpus_data_path=corpus_path, project_dir=project_dir)
    evaluator.start_trial(config_path)


def main():
    parser = argparse.ArgumentParser(description="AutoRAG benchmark")
    parser.add_argument("--config", required=True, help="AutoRAG config YAML file")
    parser.add_argument("--qa-data", required=True, help="QA parquet path")
    parser.add_argument("--corpus", required=True, help="Corpus parquet path")
    parser.add_argument("--project-dir", required=True, help="Output directory for all runs")
    parser.add_argument("--monitor-interval", type=float, default=0.5, help="System monitor sampling interval, seconds")
    args = parser.parse_args()

    results_dir = Path(args.project_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_data = load_config(args.config)
    models_to_benchmark = extract_generator_modules(config_data)
    print(f"Found models in YAML to benchmark: {models_to_benchmark}")
    
    monitor = SystemMonitor(interval_s=args.monitor_interval)
    monitor.start()
    start = time.time()
    run_autorag_eval(args.config, args.qa_data, args.corpus, str(results_dir))
    duration_s = time.time() - start
    resource_usage = monitor.stop().as_dict()
    monitor.shutdown()
    
    summary = {
        "models_benchmarked": models_to_benchmark,
        "total_duration_s": duration_s,
        "resource_usage": resource_usage,
    }
    
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved execution summary to {summary_path}")


if __name__ == "__main__":
    main()
