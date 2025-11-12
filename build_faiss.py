# build_faiss.py
# Builds embeddings for chunk files and writes a FAISS index + meta.json

import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_FOLDER = "chunks"
EMB_FILE = "embeddings.npy"
META_FILE = "meta.json"
INDEX_FILE = "index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality

# load chunk texts and ids
texts = []
meta = []
for i, fname in enumerate(sorted(os.listdir(CHUNK_FOLDER))):
    if fname.lower().endswith(".txt"):
        path = os.path.join(CHUNK_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            t = f.read()
        texts.append(t)
        meta.append({"id": i, "filename": fname})

if not texts:
    raise SystemExit("No chunk files found in 'chunks/' — run chunk_documents.py first.")

print("Loading sentence-transformers model...")
model = SentenceTransformer(MODEL_NAME)
print("Encoding", len(texts), "chunks...")
embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# normalize vectors for cosine similarity
faiss.normalize_L2(embs)

# save embeddings & meta
np.save(EMB_FILE, embs)
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# build FAISS index
d = embs.shape[1]
index = faiss.IndexFlatIP(d)  # inner-product on normalized vectors => cosine
index.add(embs)
faiss.write_index(index, INDEX_FILE)
print("Saved embeddings to", EMB_FILE)
print("Saved meta to", META_FILE)
print("Saved FAISS index to", INDEX_FILE)
