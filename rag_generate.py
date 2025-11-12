import os
import pickle
import faiss
import numpy as np
import subprocess
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ----------------- CONFIG -----------------
MODEL_NAME = "smollm:1.7b"
# ----- FORCE MODEL (debug / safety) -----
import os, sys
# ensure we're running the file you edited
from pathlib import Path
print(">>> Running file:", Path(__file__).resolve())
# force the environment variable to match the hardcoded model so nothing can override it
MODEL_NAME = "smollm:1.7b"   # <- desired model
os.environ.pop("MODEL_NAME", None)    # remove any existing env var
os.environ["MODEL_NAME"] = MODEL_NAME
print(">>> Forced MODEL_NAME (file):", MODEL_NAME)
print(">>> Forced ENV MODEL_NAME:", os.environ.get("MODEL_NAME"))
# ----------------------------------------

OLLAMA_EXE = r"C:\Users\ADMIN\AppData\Local\Programs\Ollama\ollama.exe"   # adjust if your path differs



INDEX_FILE = "faiss_index.bin"
DOCSTORE_FILE = "docstore.pkl"

EMB_MODEL = "all-MiniLM-L6-v2"   # must match index
TOP_K = 1                        # ✅ only top 1 result
# ------------------------------------------

# --- Load index & docstore ---
def ensure_index_and_docstore():
    if Path(INDEX_FILE).exists() and Path(DOCSTORE_FILE).exists():
        index = faiss.read_index(INDEX_FILE)
        with open(DOCSTORE_FILE, "rb") as f:
            docstore = pickle.load(f)
        if isinstance(docstore, dict) and "chunks" in docstore:
            chunks = docstore["chunks"]
        elif isinstance(docstore, list):
            chunks = docstore
        else:
            raise RuntimeError("Unsupported docstore format.")
        print("Loaded existing FAISS index and docstore.")
        return chunks, index
    else:
        raise RuntimeError("No index/docstore found. Run ingestion first.")

# --- Embed query (same model as index) ---
def embed_query(query: str):
    model = SentenceTransformer(EMB_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    return q_emb.astype("float32")

# --- Call Ollama model ---
def ask_ollama(prompt: str):
    cmd = [OLLAMA_EXE, "run", MODEL_NAME]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    out, _ = proc.communicate(prompt)
    return out.strip()

# --- Main loop ---
def main():
    chunks, index = ensure_index_and_docstore()
    print(f"Ready — {len(chunks)} chunks indexed. Using model '{MODEL_NAME}'.\n")

    while True:
        query = input("Ask your legal question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("Goodbye 👋")
            break
        if not query:
            continue

        # Embed & search
        q_emb = embed_query(query)
        D, I = index.search(q_emb, TOP_K)

        # get top chunk (if exists)
        if I[0][0] == -1:
            print("No relevant context found.")
            continue
        context = chunks[I[0][0]]

        # prompt for model
        prompt = (
            "You are an assistant specialized in Indian legal texts. "
            "Answer strictly based only on the context excerpt. "
            "If the excerpt does not contain the answer, say: "
            "'I don't see an answer in the provided documents.'\n\n"
            f"CONTEXT EXCERPT:\n{context}\n\n"
            f"QUESTION: {query}\n\nAnswer concisely:"
        )

        answer = ask_ollama(prompt)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
