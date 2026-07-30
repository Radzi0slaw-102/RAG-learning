# clone GraphRAG-Bench and pick a small sample of corpus+questions to test with
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_URL = "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git"
TRACK_DIR = Path(__file__).parent
REPO_DIR = TRACK_DIR / "_repo"
SAMPLE_PATH = TRACK_DIR / "data" / "sample.json"

DOMAIN = "novel" # or "medical"
SAMPLE_SIZE = 20  # number of questions to keep
MAX_SOURCES = 4  # spread the sample across at most this many corpus documents


def clone_repo() -> Path:
    if REPO_DIR.exists():
        return REPO_DIR
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR


def build_sample(repo_dir: Path, domain: str, sample_size: int, max_sources: int) -> list[dict]:
    corpus = json.loads((repo_dir / "Datasets" / "Corpus" / f"{domain}.json").read_text())
    questions = json.loads((repo_dir / "Datasets" / "Questions" / f"{domain}_questions.json").read_text())

    corpus_by_name = {c["corpus_name"]: c["context"] for c in corpus}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if q["source"] in corpus_by_name:
            grouped[q["source"]].append(q)

    sources = list(grouped.keys())[:max_sources]
    per_source = max(1, sample_size // len(sources)) if sources else 0

    sample = []
    for source in sources:
        for q in grouped[source][:per_source]:
            if len(sample) >= sample_size:
                break
            sample.append({
                "id": q["id"],
                "source": source,
                "context": corpus_by_name[source],
                "question": q["question"],
                "answer": q["answer"],
                "question_type": q["question_type"],
            })
    return sample


def main() -> None:
    repo_dir = clone_repo()
    sample = build_sample(repo_dir, DOMAIN, SAMPLE_SIZE, MAX_SOURCES)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sample, indent=2))
    print(f"Wrote {len(sample)} questions ({DOMAIN} domain) to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()