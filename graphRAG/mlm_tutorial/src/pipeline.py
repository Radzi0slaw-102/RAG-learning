from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from config import OLLAMA_MODEL, OLLAMA_EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_PATHS_PER_CHUNK, KG_TRIPLET_EXTRACT_TMPL
from data_loader import load_documents
from extraction import GraphRAGEXtractor, parse_fn
from graph_store import GraphRAGStore
from query_engine import GraphRAGQueryEngine

def build_index(csv_url: str):
    llm = Ollama(model=OLLAMA_MODEL)
    embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)
    
    documents = load_documents(csv_url)
    nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).get_nodes_from_documents(documents)
    kg_extractor = GraphRAGEXtractor(
        llm=llm, extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=MAX_PATHS_PER_CHUNK, parse_fn=parse_fn
    )
    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=GraphRAGStore(),
        kg_extractors=[kg_extractor],
        embed_model=embed_model,
        show_progress=True
    )
    index.property_graph_store.build_communities()
    return index, llm

def run_query(index, llm, question: str) -> str:
    engine = GraphRAGQueryEngine(graph_store=index.property_graph_store, llm=llm)
    return engine.query(question).response