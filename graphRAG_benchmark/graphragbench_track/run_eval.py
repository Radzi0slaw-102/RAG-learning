from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_URL = "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git"
TRACK_DIR = Path(__file__).parent
REPO_DIR = TRACK_DIR / "_repo"
SAMPLE_PATH = TRACK_DIR / "data" / "sample.json"

DOMAIN = "novel"  # or "medical"
SAMPLE_SIZE = 20  # number of questions to keep
MAX_SOURCES = 4  # spread the sample across at most this many corpus documents
CONTEXT_CHAR_LIMIT = 8000  # truncate each corpus doc to keep local Ollama indexing fast for the MVP


def _is_answerable(answer: str, context: str) -> bool:
    """Rough check that a truncated context still contains the answer text.

    Not a rigorous evidence check (the repo's `evidence` field isn't kept
    here), but it filters out the clear cases where truncation strands
    the answer outside the kept context, which would unfairly tank scores.
    """
    return answer.strip().lower() in context.lower()


def clone_repo() -> Path:
    if REPO_DIR.exists():
        return REPO_DIR
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR


def build_sample(repo_dir: Path, domain: str, sample_size: int, max_sources: int, context_char_limit: int) -> list[dict]:
    corpus = json.loads((repo_dir / "Datasets" / "Corpus" / f"{domain}.json").read_text(encoding="utf-8"))
    questions = json.loads((repo_dir / "Datasets" / "Questions" / f"{domain}_questions.json").read_text(encoding="utf-8"))

    corpus_by_name = {c["corpus_name"]: c["context"][:context_char_limit] for c in corpus}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if q["source"] not in corpus_by_name:
            continue
        if not _is_answerable(q["answer"], corpus_by_name[q["source"]]):
            continue
        grouped[q["source"]].append(q)

    sources = [s for s in grouped if grouped[s]][:max_sources]
    per_source = max(1, sample_size // len(sources)) if sources else 0

    sample = []
    skipped_sources = [s for s in list({q["source"] for q in questions} & corpus_by_name.keys()) if s not in grouped]
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

    if skipped_sources:
        print(f"Note: {len(skipped_sources)} source(s) had no answerable questions "
              f"within the first {context_char_limit} chars, skipped entirely")

    return sample


def main() -> None:
    repo_dir = clone_repo()
    sample = build_sample(repo_dir, DOMAIN, SAMPLE_SIZE, MAX_SOURCES, CONTEXT_CHAR_LIMIT)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample)} questions ({DOMAIN} domain) to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()