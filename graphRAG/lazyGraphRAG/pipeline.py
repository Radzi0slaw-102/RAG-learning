from llama_index.core.schema import TextNode

from entity_extraction import EntityExtractor
from graph_builder import GraphBuilder
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from config import DEFAULT_MODEL, HOST

def build_graph_from_nodes(
    nodes: list[TextNode],
    model: str = DEFAULT_MODEL,
    host: str | None = HOST
) -> GraphBuilder:
    extractor = EntityExtractor(model=model, host=host)
    builder = GraphBuilder()
    
    for node in nodes:
        result = extractor.extract(text=node.get_content(), node_id=node.node_id)
        builder.add_extraction(result)
    
    return builder


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data").load_data()
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
    
    graph_builder = build_graph_from_nodes(nodes)
    print(graph_builder.stats())
    graph_builder.save("knowledge_graph.graphml")