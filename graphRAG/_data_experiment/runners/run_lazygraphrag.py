"""
Runner for lazyGraphRAG.
Replaces the HuggingFace Gutenberg dataset loader with one that reads
the canonical dataset manifest. Runs the whole thing in a subprocess
with a hard timeout.

Usage:
    python run_lazygraphrag.py --dataset datasets/npm_lodash --questions eval/questions_npm_lodash.yaml --timeout 900
"""
import argparse
import json
import multiprocessing
import re
import sys

sys.path.insert(0, "../lazyGraphRAG")

import pandas as pd
import yaml
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from question_keywords import question_keywords


def load_documents_from_manifest(dataset_dir: str) -> list[Document]:
    data = pd.read_csv(f"{dataset_dir}/manifest.csv")
    return [Document(text=f"{row['title']}: {row['text']}") for _, row in data.iterrows()]


def load_nodes_from_manifest(dataset_dir: str, chunk_size: int = 512, chunk_overlap: int = 50, max_nodes: int | None = None):
    documents = load_documents_from_manifest(dataset_dir)
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(documents)
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    return nodes


def query_graph(graph, question: str) -> str:
    keywords = question_keywords(question)
    if not keywords:
        return "NO_MATCH"
    
    matched_nodes = [
        n for n, data in graph.nodes(data=True)
        if any(kw in n or kw in data.get("name", "").lower() for kw in keywords)
    ]
    
    hits = []
    for u, v, data in graph.edges(data=True):
        if u in matched_nodes or v in matched_nodes:
            u_name = graph.nodes[u].get("name", u)
            v_name = graph.nodes[v].get("name", v)
            hits.append(f"{u_name} -[{data.get('relation')}]-> {v_name}")
    
    return "; ".join(hits) if hits else "NO_MATCH"


def _worker(dataset_dir: str, questions_path: str, out_path: str, max_nodes: int | None):
    # import here, not at module level, so the timeout wrapper can kill this cleanly
    from pipeline import build_graph_from_nodes
    
    nodes = load_nodes_from_manifest(dataset_dir, max_nodes=max_nodes)
    graph_builder = build_graph_from_nodes(nodes)
    graph = graph_builder.graph
    
    with open(questions_path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    
    results = []
    for q in questions:
        answer = query_graph(graph, q["question"])
        results.append({"id": q["id"], "question": q["question"], "answer": answer})
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--max-nodes", type=int, default=None, help="Cap on chunks sent through LLM extraction")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds before giving up")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    out_path = args.out or f"results/lazygraphrag_{args.dataset.split('/')[-1]}.json"
    
    proc = multiprocessing.Process(
        target=_worker, args=(args.dataset, args.questions, out_path, args.max_nodes)
    )
    proc.start()
    proc.join(timeout=args.timeout)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"lazyGraphRAG timed out after {args.timeout}s on {args.dataset} -- marking as TIMEOUT")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"status": "TIMEOUT", "dataset": args.dataset}, f, indent=2)
        return
    
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()