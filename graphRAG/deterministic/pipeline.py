import networkx as nx

from coref import preprocess_text
from extraction import extract_triples, load_pipeline
from graph_builder import build_graph
from normalize import EntityResolver


def build_kg_from_text(text: str, model: str = "en_core_web_sm") -> nx.MultiDiGraph:
    nlp = load_pipeline(model)
    
    resolved_text = preprocess_text(nlp, text)
    doc = nlp(resolved_text)
    
    triples = extract_triples(doc)
    resolver = EntityResolver()
    return build_graph(triples, resolver)


if __name__ == "__main__":
    sample = (
        "Marie Curie was a physicist. She discovered radium. "
        "Radium is a radioactive element. Curie won the Nobel Prize in Physics. "
        "The Nobel Prize is awarded in Stockholm. Paris is the capital of France. "
        "Curie later moved to Paris, a city in France."
    )
    
    graph = build_kg_from_text(sample)
    
    print(f"Nodes ({graph.number_of_nodes()}):")
    for node in graph.nodes:
        print(f"  - {node}")
    
    print(f"\nEdges ({graph.number_of_edges()}):")
    for u, v, data in graph.edges(data=True):
        print(f"  ({u}) -[{data['relation']}]-> ({v})")
        print(f"      from: \"{data['sentence']}\"")