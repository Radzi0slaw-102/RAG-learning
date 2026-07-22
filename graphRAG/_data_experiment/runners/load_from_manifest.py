import pandas as pd
from llama_index.core import Document

def load_documents_from_manifest(dataset_dir: str) -> list[Document]:
    data = pd.read_csv(f"{dataset_dir}/manifest.csv")
    return [Document(text=f"{row['title']}: {row['text']}") for _, row in data.iterrows()]