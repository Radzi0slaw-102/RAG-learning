from openai import OpenAI
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load local client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def get_embedding(text, model="nomic-embed-text"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


url = "https://www.archives.gov/founding-docs/constitution-transcript"
loader = WebBaseLoader(url)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)
documents = text_splitter.split_documents(data)
# extract page content from the documents
documents = [doc.page_content for doc in documents]

# Generate embeddings for each document
embeddings = [get_embedding(doc) for doc in documents]
ids = [f"id{i}" for i in range(len(documents))]

#create chroma database client
chroma_client = chromadb.Client()
#create a collection
collection = chroma_client.create_collection(name="documents")

collection.add(
    embeddings=embeddings,
    documents=documents,    
    ids=ids
)

def query_chromadb(query, top_n=2):
    """Returns the text of the top 2 results from the ChromaDB collection
    """    
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],    
        n_results=top_n
    )
    return [(id, score, text) for id, score, text in 
            zip(results['ids'][0], results['distances'][0], results['documents'][0])]
        

# Input Loop for Search Queries
while True:
    query = input("Enter a search query (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break
    top_n = int(input("How many top matches do you want to see? "))
    search_results = query_chromadb(query, top_n)
    
    print("Top Matched Documents:")
    for id, score, text in search_results:
        print(f"ID:{id} TEXT: {text} SCORE: {round(score, 2)}")

    print("\n")

# Blind tests:
#
# 1. "What are the requirements to become the President?"
# Perfect match. Successfully found the exact passage 
# stating the age limit (35) and natural-born citizen requirement (score 0.62).
#
# 2. "Can the government censor newspapers or bad speech?"
# Weak matches. Model showed generic text about government secrecy
# and congressional speech immunity (0.84 - 0.90)
# Original USA constitution conscript does not contain the Bill of Rights.
#
# 3. "Who has the power to declare war?"
# Missed the exact answer. It retrieved generic sections about 
# treaties and states engaging in war (0.71 - 0.73), but missed the 
# explicit section giving Congress the power to declare war
# Possible reason - model struggled with the linguistic gap
# between modern phrasing and archaic 18th-century legal syntax.
#
# Summary: Chunking big document into pieces won't help,
# if there are no additional context and documents.