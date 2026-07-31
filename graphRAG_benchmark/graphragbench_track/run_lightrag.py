from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from functools import partial

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

TRACK_DIR = Path(__file__).parent
SAMPLE_PATH = TRACK_DIR / "data" / "sample.json"
WORKING_ROOT = TRACK_DIR / "data" / "lightrag_storage"
PREDICTIONS_PATH = TRACK_DIR / "data" / "predictions.json"

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.1:70b"
EMBED_MODEL = "qwen3-embedding:8b"
EMBED_DIM = 4096


async def build_rag(working_dir: Path) -> LightRAG:
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host": OLLAMA_HOST, "options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=partial(ollama_embed.func, embed_model=EMBED_MODEL, host=OLLAMA_HOST),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def process_source(source: str, entries: list[dict]) -> list[dict]:
    working_dir = WORKING_ROOT / source
    rag = await build_rag(working_dir)
    await rag.ainsert(entries[0]["context"])

    predictions = []
    for entry in entries:
        response = await rag.aquery(entry["question"], param=QueryParam(mode="hybrid", top_k=10))
        predictions.append({
            "id": entry["id"],
            "source": source,
            "question": entry["question"],
            "answer": entry["answer"],
            "question_type": entry["question_type"],
            "prediction": str(response),
        })
        print(f"[{entry['id']}] answered")
    return predictions


async def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "not-needed-for-ollama")
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in sample:
        grouped[entry["source"]].append(entry)

    all_predictions = []
    for source, entries in grouped.items():
        print(f"Processing corpus: {source} ({len(entries)} question(s))")
        all_predictions.extend(await process_source(source, entries))

    PREDICTIONS_PATH.write_text(json.dumps(all_predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_predictions)} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())