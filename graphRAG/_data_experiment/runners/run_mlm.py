"""
Runner for the Machine Learning Mastery GraphRAG tutorial method.
Replaces the CSV-URL data loader with one that reads the canonical
dataset manifest, then reuses build_index/run_query unchanged.
 
Usage:
    python run_mlm.py --dataset datasets/npm_lodash --questions eval/questions_npm_lodash.yaml
"""
import argparse
import json
import sys

sys.path.insert(0, "../mlm_tutorial/src")

import pandas as pd
import yaml
from llama_index.core import Document

from load_from_manifest import load_documents_from_manifest
from pipeline import build_index, run_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--persist-dir", default="storage")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    documents = load_documents_from_manifest(args.dataset, args.limit)
    
    # build_index expects a csv_url and calls load_documents(csv_url, limit) internally;
    # here it's bypassed by constructing the index directly with our own documents
    # if build_index's signature can't be reused as-is, inline its body with `documents`
    # substituted for `load_documents(csv_url, limit=100)`
    from pipeline import GraphRAGStore, GraphRAGQueryEngine, RebelExtractor
    from llama_index.core import PropertyGraphIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
    from config import OLLAMA_MODEL, OLLAMA_EMBED_MODEL, REQUEST_TIMEOUT, CHUNK_SIZE, CHUNK_OVERLAP
    
    llm = Ollama(model=OLLAMA_MODEL, request_timeout=REQUEST_TIMEOUT)
    embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, request_timeout=REQUEST_TIMEOUT)
    nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).get_nodes_from_documents(documents)
    
    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=GraphRAGStore(),
        kg_extractors=[RebelExtractor()],
        embed_model=embed_model,
        show_progress=True,
    )
    index.property_graph_store.build_communities()
    index.property_graph_store.persist(f"{args.persist_dir}/{args.dataset.split('/')[-1]}")
    
    with open(args.questions, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    
    results = []
    for q in questions:
        answer = run_query(index, llm, q["question"])
        results.append({"id": q["id"], "question": q["question"], "answer": answer})
    
    out_path = args.out or f"results/mlm_{args.dataset.split('/')[-1]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} answers to {out_path}")


if __name__ == "__main__":
    main()