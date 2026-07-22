"""
Runner for the deterministic spaCy-based method.
Builds one merged knowledge graph from all documents in a dataset manifest,
then answers each question by keyword-matching entities/edges in the graph.
 
Usage:
    python run_deterministic.py --dataset datasets/npm_lodash --questions eval/questions_npm_lodash.yaml
"""
import argparse
import csv
import json
import sys

sys.path.insert(0, "../deterministic")

from coref import preprocess_text
from extraction import extract_triples, load_pipeline
from graph_builder import build_graph
from normalize import EntityResolver

import networkx as nx
import yaml


def load_manifest(dataset_dir: str) -> list[dict]:
    with open(f"{dataset_dir}/manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_kg_from_corpus(rows: list[dict], model: str = "en_core_web_sm") -> nx.MultiDiGraph:
    nlp = load_pipeline(model)
    resolver = EntityResolver()
    merged = nx.MultiDiGraph()
    
    for row in rows:
        resolved_text = preprocess_text(nlp, row["text"])
        doc = nlp(resolved_text)
        triples = extract_triples(doc)
        graph = build_graph(triples, resolver)
        merged = nx.compose(merged, graph)
    
    return merged


def answer_question(graph: nx.MultiDiGraph, question: str) -> str:
    # naive lookup: return edges whose nodes or relation match a question keyword
    q_lower = question.lower()
    hits = []
    for u, v, data in graph.edges(data=True):
        blob = f"{u} {data.get('relation', '')} {v}".lower()
        if any(word in blob for word in q_lower.split() if len(word) > 3):
            hits.append(f"{u} -[{data.get('relation')}]-> {v}")
    return "; ".join(hits) if hits else "NO_MATCH"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    rows = load_manifest(args.dataset)
    graph = build_kg_from_corpus(rows)
    
    with open(args.questions, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
        
    results = []
    for q in questions:
        answer = answer_question(graph, q["question"])
        results.append({"id": q["id"], "question": q["question"], "answer": answer})
        
    out_path = args.out or f"results/deterministic_{args.dataset.split('/')[-1]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} answers to {out_path}")


if __name__ == "__main__":
    main()