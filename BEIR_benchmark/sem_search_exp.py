while True:
    user_choice = input("Select dataset mode ('custom' for local files / 'beir' for NFCorpus): ").strip().lower()
    if user_choice in ["custom", "beir"]:
        dataset_mode = user_choice
        break
    print("Invalid choice. Please type 'custom' or 'beir'.")

print("\n Loading libraries, please wait...")

import os
import time
import numpy as np
import sys
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


beir_dataset_name = "nfcorpus"
corpus = {}
queries = {}
qrels = {}

if dataset_mode == "custom":
    required_files = ["documents.txt", "test_queries.txt", "qrels.txt"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Error: missing required file(s): {', '.join(missing)}")
        sys.exit(1)
    
    with open("documents.txt", "r", encoding="utf-8") as f:
        docs_lines = [line.strip() for line in f if line.strip()]
    with open("test_queries.txt", "r", encoding="utf-8") as f:
        queries_lines = [line.strip() for line in f if line.strip()]
    
    for i, text in enumerate(docs_lines):
        corpus[f"doc_{i}"] = {"title": "", "text": text}
    for i, text in enumerate(queries_lines):
        queries[f"query_{i}"] = text
    
    with open("qrels.txt", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"Warning: skipping malformed qrels.txt line {line_num}: {line!r}")
                continue
            q_idx, d_idx = parts
            qrels.setdefault(f"query_{q_idx}", {})[f"doc_{d_idx}"] = 1
    
    for i in range(len(queries_lines)):
        qrels.setdefault(f"query_{i}", {})
        
elif dataset_mode == "beir":
    # Fetch, extract and structure standard BEIR nechmark files by built-in data loader
    # corpus: doc_id -> {title, text}, queries: query_id -> text, qrels: query_id -> {doc_id -> relevance_score}
    from beir.util import download_and_unzip
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_dataset_name}.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

corpus_ids = list(corpus.keys())
corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]

# Sparse matrix generation and batch query-to-document cosine similarity calculation
print("\nProcessing TF-IDF...")
tfidf_start = time.time()
tfidf_vectorizer = TfidfVectorizer()
tfidf_corpus = tfidf_vectorizer.fit_transform(corpus_texts)
tfidf_query = tfidf_vectorizer.transform(query_texts)
tfidf_sim = cosine_similarity(tfidf_query, tfidf_corpus)

tfidf_results = {}
for i, qid in enumerate(query_ids):
    tfidf_results[qid] = {corpus_ids[j]: float(tfidf_sim[i][j]) for j in range(len(corpus_ids))}
tfidf_time = time.time() - tfidf_start

# Unsupervised Skip-Gram training followed by average aggregation of word vectors for document embeddings
print("Processing Word2Vec...")
w2v_start = time.time()
tokenized_docs = [text.lower().replace(".", "").replace("!", "").split() for text in corpus_texts]
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    min_count=1,
    vector_size=100,
    window=5,
    sg=1,
    epochs=50
)

def get_w2v_embedding(text):
    tokens = text.lower().replace(".", "").replace("!", "").split()
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(w2v_model.vector_size)

w2v_corpus = np.array([get_w2v_embedding(text) for text in corpus_texts])
w2v_query = np.array([get_w2v_embedding(text) for text in query_texts])
w2v_sim = cosine_similarity(w2v_query, w2v_corpus)

w2v_results = {}
for i, qid in enumerate(query_ids):
    w2v_results[qid] = {corpus_ids[j]: float(w2v_sim[i][j]) for j in range(len(corpus_ids))}
w2v_time = time.time() - w2v_start

# Local transformer execution forced on host RAM and CPU cores without hardware acceleration
print("Processing BERT (CPU)...")
bert_cpu_start = time.time()
bert_cpu_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
bert_cpu_corpus = bert_cpu_model.encode(corpus_texts, show_progress_bar=False)
bert_cpu_query = bert_cpu_model.encode(query_texts, show_progress_bar=False)
bert_cpu_sim = cosine_similarity(bert_cpu_query, bert_cpu_corpus)

bert_cpu_results = {}
for i, qid in enumerate(query_ids):
    bert_cpu_results[qid] = {corpus_ids[j]: float(bert_cpu_sim[i][j]) for j in range(len(corpus_ids))}
bert_cpu_time = time.time() - bert_cpu_start

# Model weights and tensor computations pushed to Nvidia GPU VRAM leveraging CUDA parallel processing
cuda_available = torch.cuda.is_available()
bert_cuda_results = {}
bert_cuda_time = 0.0
if cuda_available:
    print("Processing BERT (CUDA)...")
    bert_cuda_start = time.time()
    bert_cuda_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    bert_cuda_corpus = bert_cuda_model.encode(corpus_texts, show_progress_bar=False)
    bert_cuda_query = bert_cuda_model.encode(query_texts, show_progress_bar=False)
    bert_cuda_sim = cosine_similarity(bert_cuda_query, bert_cuda_corpus)
    
    for i, qid in enumerate(query_ids):
        bert_cuda_results[qid] = {corpus_ids[j]: float(bert_cuda_sim[i][j]) for j in range(len(corpus_ids))}
    bert_cuda_time = time.time() - bert_cuda_start
else:
    print("Warning: CUDA is not available. Check PyTorch/Nvidia drivers installation. Skipping BERT (CUDA).")


print("Generating report file...")
report_filename = f"report_benchmark_{dataset_mode}.txt"
k_values = [1, 3, 5, 10]

report_lines = []
report_lines.append("=" * 50)
report_lines.append(f"BEIR benchmark report - dataset mode: {dataset_mode}")
report_lines.append("=" * 50)
report_lines.append(f"Total documents: {len(corpus_texts)}")
report_lines.append(f"Total queries: {len(query_texts)}")
report_lines.append(
    f"GPU Device detected: {torch.cuda.get_device_name(0)}"
    if cuda_available else "GPU Device detected: None (CUDA Unavailable)"
)
report_lines.append("-" * 50)
report_lines.append(f"TF-IDF execution time: {tfidf_time:.4f}s")
report_lines.append(f"Word2Vec execution time: {w2v_time:.4f}s")
report_lines.append(f"BERT (CPU) execution time: {bert_cpu_time:.4f}s")
if cuda_available:
    report_lines.append(f"BERT (CUDA) execution time: {bert_cuda_time:.4f}s")
report_lines.append("=" * 50 + "\n")

evaluation_targets = [
    ("TF-IDF", tfidf_results),
    ("Word2Vec", w2v_results),
    ("BERT (CPU)", bert_cpu_results),
]
if cuda_available:
    evaluation_targets.append(("BERT (CUDA)", bert_cuda_results))
    
# Evaluate Retrieval triggers an automated stdout print of the benchmark report table.
# It unpacks 4 standard IR metrics: NDCG, MAP, Recall, Precision:
#
# 1. NDCG@K = DCG@K / IDCG@K, where DCG = Sum(Relevance / log2(Position + 1))
# Advanced graded relevance penalizing low-ranked results.
#
# 2. MAP = Mean of Average Precision points where relevant docs are found
# Binary relevance with rank ordering.
#
# 3. Recall@K = (Relevant Docs in Top K) / (Total Existing Relevant Docs)
# Measures system completeness.
#
# 4. Precision@K = (Relevant Docs in Top K) / K
# Measures noise and retrieval purity.
for name, results in evaluation_targets:
    report_lines.append(f"Evaluation metrics for: {name}")
    report_lines.append("-" * 50)
    ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    for metric_dict in (ndcg, map_, recall, precision):
        for metric_name, value in metric_dict.items():
            report_lines.append(f"{metric_name}: {value:.4f}")
    report_lines.append("\n" + "=" * 50 + "\n")

with open(report_filename, "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(report_lines))

print(f"Report saved to '{report_filename}'")