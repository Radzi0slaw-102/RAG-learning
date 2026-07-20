"""
Build (or load) a LlamaIndex vector index over the PubMedQA corpus, and
expose a query engine backed by a given Ollama LLM. The index is built
once with a fixed embedding model and persisted to disk, then reused
across every LLM being benchmarked.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
INDEX_PERSIST_DIR = PROJECT_ROOT / "data" / "index"
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus_documents(max_documents: int | None = None) -> list[Document]:
    paths = sorted(CORPUS_DIR.glob("*.json"))
    if max_documents is not None:
        paths = paths[:max_documents]

    documents = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        documents.append(Document(text=record["text"], doc_id=record["id"]))
    return documents


def set_index(config: dict) -> VectorStoreIndex:
    Settings.embed_model = OllamaEmbedding(
        model_name=config["embedding_model"],
        base_url=config["ollama_base_url"],
    )

    if INDEX_PERSIST_DIR.exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_PERSIST_DIR))
        return load_index_from_storage(storage_context)
    
    documents = load_corpus_documents(max_documents=config.get("max_corpus_documents"))
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(INDEX_PERSIST_DIR))
    return index


def get_query_engine(index: VectorStoreIndex, llm_model_name: str, config: dict, top_k: int = 3):
    llm = Ollama(
        model=llm_model_name,
        base_url=config["ollama_base_url"],
        request_timeout=300.0
    )
    return index.as_query_engine(llm=llm, similarity_top_k=top_k)


if __name__ == "__main__":
    cfg = load_config()
    idx = set_index(cfg)
    first_model = cfg["llm_models"][0]["name"]
    engine = get_query_engine(idx, first_model, cfg)

    response = engine.query("Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?")
    print(response)