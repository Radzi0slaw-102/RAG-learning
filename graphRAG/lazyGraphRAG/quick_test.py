# quick_test.py — testowanie samej ekstrakcji bez folderu data/
from llama_index.core.schema import TextNode
from pipeline import build_graph_from_nodes

sample_text = """
Marie Curie was a physicist and chemist who conducted pioneering research
on radioactivity. She was born in Warsaw, Poland, and later worked at the
University of Paris. Curie was the first woman to win a Nobel Prize.
"""

nodes = [TextNode(text=sample_text, id_="test-node-1")]
graph_builder = build_graph_from_nodes(nodes)
print(graph_builder.stats())

for node_id, data in graph_builder.graph.nodes(data=True):
    print(node_id, data)

for source, target, data in graph_builder.graph.edges(data=True):
    print(source, "->", target, data)