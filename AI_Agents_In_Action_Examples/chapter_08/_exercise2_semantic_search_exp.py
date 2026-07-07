import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

with open("test_queries.txt", "r", encoding="utf-8") as f:
    test_queries = [line.strip() for line in f if line.strip()]

# Prepare global TF-IDF vectorizer
global_tf_idf = TfidfVectorizer()

# Prepare the docs and setup model for Word2Vec (Skip-Gram)
tokenized_docs = [doc.lower().replace(".", "").replace("!", "").split() for doc in documents]

w2v_start_train = time.time()
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    min_count=1,
    vector_size=100,
    window=5,
    sg=1,
    epochs=100
)
w2v_train_time = time.time() - w2v_start_train

# Vectorize with TF-IDF Vectorizer
def tf_idf_vectorize():
    X = global_tf_idf.fit_transform(documents)
    return X.toarray()

# Vectorize with Word2Vec model
def w2v_vectorize():
    db_vectors = []
    for tokens in tokenized_docs:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            doc_vector = np.mean(vectors, axis=0)
        else:
            doc_vector = np.zeros(w2v_model.vector_size)
        db_vectors.append(doc_vector)
    return np.array(db_vectors)

def get_single_w2v_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(w2v_model.vector_size)

# Vectorize with "based on BERT" model
def bert_vectorize(text):
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN", ""),
    )
    try:
        result = client.feature_extraction(
            text,
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        return np.array(result)
    except HfHubHTTPError as e:
        raise Exception(f"API Error: {e}")

# Cosine Similarity Search Function, unified for all presented methods
def cosine_similarity_search(query_vec, database, top_n=5):
    query_vec = query_vec.reshape(1, -1)
    similarities = cosine_similarity(query_vec, database)[0]
    top_indices = np.argsort(-similarities)[:top_n]  # Top n indices
    return [(idx, similarities[idx]) for idx in top_indices]

print("=== Starting database vectorization ===")

tf_idf_start = time.time()
tf_idf_db = tf_idf_vectorize()
tf_idf_build_time = time.time() - tf_idf_start
print(f"TF-IDF database build time: {tf_idf_build_time:.4f} seconds")

w2v_start_db = time.time()
w2v_db = w2v_vectorize()
w2v_build_time = time.time() - w2v_start_db
print(f"Word2Vec database build time: {w2v_build_time:.4f} seconds (Train time: {w2v_train_time:.4f}s)")

bert_start = time.time()
bert_db = bert_vectorize(documents)
bert_build_time = time.time() - bert_start
print(f"BERT API database build time: {bert_build_time:.4f} seconds\n")

tf_idf_total_search_time = 0
w2v_total_search_time = 0
bert_total_search_time = 0

for query in test_queries:
    print(f"=== QUERY: {query} ===")
    
    # TF-IDF
    start_time = time.time()
    tf_idf_q = global_tf_idf.transform([query]).toarray()
    tf_idf_res = cosine_similarity_search(tf_idf_q, tf_idf_db)
    local_time_search = time.time() - start_time
    tf_idf_total_search_time += local_time_search

    print("TF-IDF top matches:")
    for idx, score in tf_idf_res:
        print(f"  - Score: {score:.4f} | {documents[idx]}")
    print(f"Local query search time: {local_time_search}")
    
    # Word2Vec
    start_time = time.time()
    w2v_tokens = query.lower().replace(".", "").replace("!", "").split()
    w2v_q = get_single_w2v_vector(w2v_tokens)
    w2v_res = cosine_similarity_search(w2v_q, w2v_db)
    local_time_search = time.time() - start_time
    w2v_total_search_time += local_time_search

    print("Word2Vec top matches:")
    for idx, score in w2v_res:
        print(f"  - Score: {score:.4f} | {documents[idx]}")
    print(f"Local query search time: {local_time_search}")
    
    # BERT
    start_time = time.time()
    bert_q = bert_vectorize([query])[0]
    bert_res = cosine_similarity_search(bert_q, bert_db)
    local_time_search = time.time() - start_time
    bert_total_search_time += local_time_search

    print("BERT API top matches:")
    for idx, score in bert_res:
        print(f"  - Score: {score:.4f} | {documents[idx]}")
    print(f"Local query search time: {local_time_search}")
    
    print("\n" + "="*40 + "\n")

num_queries = len(test_queries)
print("=== Time performance summary ===")
print(f"TF-IDF - Avg search time: {(tf_idf_total_search_time / num_queries) * 1000:.2f} ms")
print(f"Word2Vec - Avg search time: {(w2v_total_search_time / num_queries) * 1000:.2f} ms")
print(f"BERT API - Avg search time: {(bert_total_search_time / num_queries) * 1000:.2f} ms")