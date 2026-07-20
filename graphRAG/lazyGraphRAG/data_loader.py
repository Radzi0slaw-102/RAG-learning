from datasets import load_dataset
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from config import DEFAULT_TITLES


def load_documents(titles: list[str] = DEFAULT_TITLES) -> list[Document]:
    dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    
    found = {}
    for row in dataset:
        metadata = row.get("metadata", {})
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        for target in titles:
            if target.lower() in title.lower() and target not in found:
                found[target] = row
        if len(found) == len(titles):
            break
    
    missing = set(titles) - set(found)
    if missing:
        print(f"Warning: titles not found in dataset: {missing}")
    
    return [
        Document(text=row["text"], metadata={"title": title})
        for title, row in found.items()
    ]


def load_nodes(
    titles: list[str] = DEFAULT_TITLES,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    max_nodes: int | None = 200
):
    documents = load_documents(titles)
    nodes = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).get_nodes_from_documents(documents)
    
    if max_nodes is not None:
        nodes = nodes[:max_nodes]