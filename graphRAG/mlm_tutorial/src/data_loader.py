import pandas as pd
from llama_index.core import Document

def load_documents(url: str, limit: int = 50) -> list[Document]:
    data = pd.read_csv(url)[:limit]
    return [Document(text=f"{row['title']}: {row['text']}") for _, row in data.iterrows()]