import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Documents
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!",
    "Sentence that won't have any similarity to others, I swear."
]

# Step 1: Vectorize with TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Step 2: Store vectors in a simple vector database (here, a list)
vector_database = X.toarray()

# Step 3: Cosine Similarity Search Function
def cosine_similarity_search(query, database, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_vec, database)[0]
    top_indices = np.argsort(-similarities)[:top_n]  # Top n indices
    return [(idx, similarities[idx]) for idx in top_indices]

# Input Loop for Search Queries
while True:
    query = input("Enter a search query (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break
    top_n = int(input("How many top matches do you want to see? "))
    search_results = cosine_similarity_search(query, vector_database, vectorizer, top_n)
    
    print("Top Matched Documents:")
    for idx, score in search_results:
        print(f"- {documents[idx]} (Score: {score:.4f})")

    print("\n")

# With some blind tests, I have concluded:
#
# 1. Even if the method is simple, cosine similarity does not catch however similar words
# i.e.
# a) with search query "anything" there are no scores greater than zero,
# especially interesting, when compared to word "any" in last sentence
# b) with search query "laziness" there are no scores greater than zero,
# again, compared to "lazy"
#
# 2. Longer queries seems to lead to far better scores than single word,
# due to more key words (if given)
# i.e. for sentence "The sky is blue and beautiful":
# a) "take a look and watch the sky" query gave ~0.67 score
# b) "take a look and watch sky" query gave ~0.57 score
# c) "look and watch sky" query gave ~0.57 score (identical to previous)
# d) "sky" query gave ~0.46 score