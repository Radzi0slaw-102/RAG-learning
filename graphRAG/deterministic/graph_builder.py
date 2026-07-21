# builds a networkx MultiDiGraph from triples, applying entity normalization

import networkx as nx

from extraction import Triple
from normalize import EntityResolver


def build_graph(triples: list[Triple], resolver: EntityResolver | None = None) -> nx.MultiDiGraph:
    resolver = resolver or EntityResolver()
    graph = nx.MultiDiGraph()
    
    seen: set[tuple[str, str, str]] = set()
    
    for triple in triples:
        subj = resolver.resolve(triple.subject)
        obj = resolver.resolve(triple.object)
        relation = triple.relation.strip().lower()
        
        if not subj or not obj or subj == obj:
            continue
        
        dedup_key = (subj, relation, obj)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        
        graph.add_node(subj)
        graph.add_node(obj)
        graph.add_edge(subj, obj, relation=relation, sentence=triple.sentence)
    
    return graph


def graph_summary(graph: nx.MultiDiGraph) -> str:
    lines = [f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"]
    for u, v, data in graph.edges(data=True):
        lines.append(f"  ({u}) -[{data['relation']}]-> ({v})")
    return "\n".join(lines)