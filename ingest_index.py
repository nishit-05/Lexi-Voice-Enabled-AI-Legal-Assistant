import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to dataset
DATA_PATH = "data"

# Store documents
documents = []
for filename in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, filename)
    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)

# Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save documents
with open("docstore.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Index built successfully with", len(documents), "documents.")
