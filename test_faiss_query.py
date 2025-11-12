# test_faiss_query.py
import faiss, json, numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "index.faiss"
META_FILE = "meta.json"
CHUNK_FOLDER = "chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

index = faiss.read_index(INDEX_FILE)
meta = json.load(open(META_FILE, "r", encoding="utf-8"))
texts = [open("chunks/"+m["filename"], "r", encoding="utf-8").read() for m in meta]

model = SentenceTransformer(MODEL_NAME)

def search(q, k=3):
    q_emb = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), k)
    for dist, idx in zip(D[0], I[0]):
        print(f"Score: {dist:.3f}\nSnippet:\n{texts[idx][:400]}\n---\n")

if __name__ == "__main__":
    query = input("Enter test query: ")
    search(query)
