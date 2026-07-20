# builds a networkx knowledge graph from extraction results

import networkx as nx

from entity_extraction import ExtractionResult


def _normalize_name(name: str) -> str:
    return name.strip().lower()


class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_extraction(self, result: ExtractionResult) -> None:
        for entity in result.entities:
            key = _normalize_name(entity.name)
            if key in self.graph.nodes:
                node = self.graph.nodes[key]
                node["source_node_ids"].add(entity.source_node_id)
                if entity.description:
                    node["descriptions"].add(entity.description)
            else:
                self.graph.add_node(
                    key,
                    name=entity.name,
                    type=entity.type,
                    descriptions={entity.description} if entity.description else set(),
                    source_node_ids={entity.source_node_id},
                )
        
        for relation in result.relations:
            src, tgt = _normalize_name(relation.source), _normalize_name(relation.target)
            if src not in self.graph.nodes or tgt not in self.graph.nodes:
                continue
            self.graph.add_edge(
                src,
                tgt,
                relation=relation.relation,
                descriptions=relation.description,
                source_node_id=relation.source_node_id,
            )
    
    def merge_results(self, results: list[ExtractionResult]) -> None:
        for result in results:
            self.add_extraction(result)
    
    def stats(self) -> dict:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
        }
    
    def save(self, path: str) -> None:
        # graphml can't store sets, so flatten them to strings on export
        export_graph = self.graph.copy()
        for _, data in export_graph.nodes(data=True):
            data["descriptions"] = " | ".join(data.get("descriptions", []))
            data["source_node_ids"] = ",".join(data.get("source_node_ids", []))
        nx.write_graphml(export_graph, path)
    
    @classmethod
    def load(cls, path: str) -> "GraphBuilder":
        builder = cls()
        builder.graph = nx.read_graphml(path)
        return builder