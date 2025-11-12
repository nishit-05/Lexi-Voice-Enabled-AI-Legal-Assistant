# retriever_test.py
# Tests reading legal text files and retrieving the most relevant one

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Change this path if your folder name is different
DATA_FOLDER = "sample_data"

# Load all .txt files from the folder
docs = []
filenames = []
for fname in os.listdir(DATA_FOLDER):
    if fname.lower().endswith(".txt"):
        with open(os.path.join(DATA_FOLDER, fname), "r", encoding="utf-8") as f:
            docs.append(f.read())
            filenames.append(fname)

# Build TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(docs)

# User query
query = "What is the statute of limitations for contract claims?"
query_vector = vectorizer.transform([query])

# Compute similarity
similarities = cosine_similarity(query_vector, doc_vectors).flatten()
best_idx = similarities.argmax()

print(f"\n🔎 Most relevant document: {filenames[best_idx]}")
print(f"\n📄 Top text snippet:\n{docs[best_idx][:300]}")
print(f"\n✅ Similarity Score: {similarities[best_idx]:.3f}")

