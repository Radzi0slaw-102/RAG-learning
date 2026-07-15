import os
import nest_asyncio
# allows safe reuse of the event loop across extractor and embedder async calls
nest_asyncio.apply()

from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from config import OLLAMA_MODEL, OLLAMA_EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_PATHS_PER_CHUNK, REQUEST_TIMEOUT, KG_TRIPLET_EXTRACT_TMPL
from data_loader import load_documents
from extraction import GraphRAGEXtractor, parse_fn
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
    documents = load_documents(csv_url, limit=10)
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