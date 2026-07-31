# combine results from both tracks into a single side-by-side comparison
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
MS_RESULTS = ROOT / "microsoft_track" / "data" / "hotpotqa_sample" / "results.json"
GRAPHRAGBENCH_RESULTS = ROOT / "graphragbench_track" / "data" / "eval_results.json"
OUTPUT_PATH = Path(__file__).parent / "comparison.json"


def summarize_ms(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text())
    return {
        "available": True,
        "n_questions": len(data["results"]),
        "avg_rouge_l": data["avg_rouge_l"],
    }


def summarize_graphragbench(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    data = json.loads(path.read_text())
    return {
        "available": True,
        "n_questions": len(data["results"]),
        "avg_rouge_l": data["avg_rouge_l"],
        "avg_answer_correctness": data["avg_answer_correctness"],
        "avg_coverage": data["avg_coverage"],
    }


def main() -> None:
    comparison = {
        "microsoft_graphrag": summarize_ms(MS_RESULTS),
        "graphrag_bench_lightrag": summarize_graphragbench(GRAPHRAGBENCH_RESULTS),
    }
    OUTPUT_PATH.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()