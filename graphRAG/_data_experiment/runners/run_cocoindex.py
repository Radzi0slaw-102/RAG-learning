"""
Runner for the cocoindex method.
cocoindex reads .md files from a source directory, so script simply points
sourcedir at the canonical dataset's docs folder - no format conversion
needed. This script wraps the existing app_main pipeline and adds a
question-answering loop over the resulting index.

Usage:
    python run_cocoindex.py --dataset datasets/npm_lodash --questions eval/questions_npm_lodash.yaml
"""
import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, "../cocoindex")

import cocoindex as coco
import yaml
from neo4j import GraphDatabase

from question_keywords import question_keywords
from main import app_main


def clear_database(driver) -> None:
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        session.run("MATCH (n) DETACH DELETE n")


def build_index(dataset_dir: str):
    driver = get_driver()
    try:
        clear_database(driver)
    finally:
        driver.close()
    
    app = coco.App(
        coco.AppConfig(name="DocsToKnowledgeGraph"),
        app_main,
        sourcedir=pathlib.Path(dataset_dir) / "docs",
    )
    app.run()
    return app


def get_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "cocoindex")
    return GraphDatabase.driver(uri, auth=(user, password))


def query_index(driver, question: str) -> str:
    keywords = question_keywords(question)
    if not keywords:
        return "NO_MATCH"
    
    # entity values or documents containing any keyword, then their 1-hop relationships
    cypher = """
    MATCH (e:Entity)
    WHERE any(kw IN $keywords WHERE toLower(e.value) CONTAINS toLower(kw))
    OPTIONAL MATCH (e)-[r:RELATIONSHIP]->(o:Entity)
    OPTIONAL MATCH (s:Entity)-[r2:RELATIONSHIP]->(e)
    RETURN e.value AS entity,
           collect(DISTINCT {predicate: r.predicate, object: o.value}) AS out_edges,
           collect(DISTINCT {subject: s.value, predicate: r2.predicate}) AS in_edges
    LIMIT 20
    """
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        records = session.run(cypher, keywords=keywords).data()
    
    hits = []
    for rec in records:
        entity = rec["entity"]
        for edge in rec["out_edges"]:
            if edge["object"]:
                hits.append(f"{entity} -[{edge['predicate']}]-> {edge['object']}")
        for edge in rec["in_edges"]:
            if edge["subject"]:
                hits.append(f"{edge['subject']} -[{edge['predicate']}]-> {entity}")
    
    return "; ".join(hits) if hits else "NO_MATCH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    build_index(args.dataset)
    
    with open(args.questions, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    
    driver = get_driver()
    try:
        results = []
        for q in questions:
            answer = query_index(driver, q["question"])
            results.append({"id": q["id"], "question": q["question"], "answer": answer})
    finally:
        driver.close()
    
    out_path = args.out or f"results/cocoindex_{args.dataset.split('/')[-1]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} answers to {out_path}")


if __name__ == "__main__":
    main()