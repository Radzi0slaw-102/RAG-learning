import os
import nest_asyncio
nest_asyncio.apply()  # allows safe reuse of the event loop across extractor and embedder async calls

from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from config import (
    OLLAMA_MODEL, OLLAMA_EMBED_MODEL, REQUEST_TIMEOUT,
    CHUNK_SIZE, CHUNK_OVERLAP,
)
from data_loader import load_documents
from rebel_extraction import RebelExtractor
from graph_store import GraphRAGStore
from query_engine import GraphRAGQueryEngine


def build_index(csv_url: str, persist_dir: str = "storage", force_rebuild: bool = False):
    llm = Ollama(model=OLLAMA_MODEL, request_timeout=REQUEST_TIMEOUT)

    if not force_rebuild and os.path.exists(persist_dir):
        graph_store = GraphRAGStore.load(persist_dir)
        if not graph_store.community_summary:
            graph_store.build_communities()
            graph_store.persist(persist_dir)
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, request_timeout=REQUEST_TIMEOUT),
        )
        return index, llm

    embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL, request_timeout=REQUEST_TIMEOUT)
    documents = load_documents(csv_url, limit=100)
    nodes = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).get_nodes_from_documents(documents)

    kg_extractor = RebelExtractor()
    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=GraphRAGStore(),
        kg_extractors=[kg_extractor],
        embed_model=embed_model,
        show_progress=True,
    )
    index.property_graph_store.build_communities()
    index.property_graph_store.persist(persist_dir)
    return index, llm

def run_query(index, llm, question: str) -> str:
    engine = GraphRAGQueryEngine(graph_store=index.property_graph_store, llm=llm)
    return engine.query(question).response