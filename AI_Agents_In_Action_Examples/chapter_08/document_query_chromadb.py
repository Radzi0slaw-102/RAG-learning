from openai import OpenAI
import chromadb

# Load local client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def get_embedding(text, model="nomic-embed-text"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Sample documents
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!",
    "Sentence that won't have any similarity to others, I swear.",
    "The industrial revolution completely transformed manufacturing processes."
]

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
        print(f"ID:{id} TEXT: {text} DISTANCE: {round(score, 2)}")

    print("\n")

# Conclusions:
# 
# 1. The embedding model looks at the main idea
# i.e. it matched "kingdom" with "king's" or "history" with "revolution". 
# TF-IDF would give a 0.0 score because the words are different.
#
# 2. The model cares about the climate of the sentence and "relations" between words.
# i.e. when searching for "love", it picked sentences about a "beautiful sky"
# over a breakfast sentence that actually used the word "love".
#
# 3. Searching for "anything" did not trick it into matching the word "any",
# because it checks the meaning, not just letters.