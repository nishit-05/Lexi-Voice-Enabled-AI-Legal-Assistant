# main_pipeline.py
# Full pipeline: STT -> Hierarchical Retriever (TF-IDF doc prefilter -> semantic chunks) -> HF LLM (FLAN-T5-small) -> TTS
# Includes Option C quick wins: TOPK=5, neighbor chunk inclusion, larger generation budget

import os
import sys
import re
import json
import numpy as np
import pyttsx3
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STT wrapper (will fall back to typed input if Vosk/model isn't present)
try:
    from stt_vosk import transcribe as stt_transcribe
except Exception:
    def stt_transcribe(duration=6, fallback_text=None):
        # If the stt_vosk import fails, return None so main will prompt typed input.
        return None

# Transformers model imports (we'll load model lazily)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _transformers_available = True
except Exception as e:
    print("Warning: transformers not available or failed to import:", e)
    _transformers_available = False

# Config
DATA_FOLDER = "data"           # folder with original documents (used as TF-IDF doc-level)
CHUNK_FOLDER = "chunks"        # folder with chunk files (used for chunk-level search)
TOPK = 5                       # number of top excerpts to send to LLM
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # embedding model for semantic search
_FAISS_INDEX_FILE = "index.faiss"
_FAISS_META_FILE = "meta.json"
_ANNOY_INDEX_FILE = "index.ann"
_EMB_FILE = "embeddings.npy"   # produced by build_faiss.py
_META_FILE = "meta.json"       # produced by build_faiss.py
LOG_FILE = "session_log.jsonl"

# -----------------------
# FAISS retriever (with neighbor inclusion) - fallback
# -----------------------
_faiss_index = None
_faiss_meta = None
_st_model = None
_faiss_available = None

def _ensure_faiss_loaded():
    global _faiss_index, _faiss_meta, _st_model, _faiss_available
    if _faiss_available is True:
        return
    _faiss_available = False
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except Exception:
        # faiss or sentence-transformers not installed
        return

    if not (os.path.exists(_FAISS_INDEX_FILE) and os.path.exists(_FAISS_META_FILE)):
        return

    try:
        _faiss_index = faiss.read_index(_FAISS_INDEX_FILE)
        with open(_FAISS_META_FILE, "r", encoding="utf-8") as f:
            _faiss_meta = json.load(f)
        _st_model = SentenceTransformer(_EMBED_MODEL_NAME)
        _faiss_available = True
    except Exception as e:
        print("Failed to load FAISS index or embeddings:", e)
        _faiss_index = None
        _faiss_meta = None
        _st_model = None
        _faiss_available = False

def faiss_retrieve_topk_with_neighbors(query, topk=TOPK):
    """
    FAISS fallback: return (meta_idx, score, joined_neighbor_texts)
    """
    _ensure_faiss_loaded()
    if not _faiss_available:
        return []
    try:
        q_emb = _st_model.encode([query], convert_to_numpy=True).astype("float32")
        import faiss
        faiss.normalize_L2(q_emb)
        D, I = _faiss_index.search(q_emb, topk)
        texts = [open(os.path.join(CHUNK_FOLDER, m["filename"]), "r", encoding="utf-8").read() for m in _faiss_meta]
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(texts):
                continue
            neighbor_texts = []
            for n in (idx-1, idx, idx+1):
                if 0 <= n < len(texts):
                    neighbor_texts.append(texts[n])
            joined = "\n\n".join(neighbor_texts)
            try:
                score = float(dist)
            except Exception:
                score = 0.0
            results.append((idx, score, joined))
        return results
    except Exception as e:
        print("FAISS retrieve error:", e)
        return []

# -----------------------
# Annoy retriever fallback (with neighbors)
# -----------------------
_annoy_index = None
_annoy_meta = None
_annoy_model = None
_annoy_available = None

def _ensure_annoy_loaded():
    global _annoy_index, _annoy_meta, _annoy_model, _annoy_available
    if _annoy_available is True:
        return
    _annoy_available = False
    try:
        from annoy import AnnoyIndex
        from sentence_transformers import SentenceTransformer
    except Exception:
        return

    if not (os.path.exists(_ANNOY_INDEX_FILE) and os.path.exists(_FAISS_META_FILE)):
        return

    try:
        with open(_FAISS_META_FILE, "r", encoding="utf-8") as f:
            _annoy_meta = json.load(f)
        _annoy_model = SentenceTransformer(_EMBED_MODEL_NAME)
        vec = _annoy_model.encode(["test"], convert_to_numpy=True)[0]
        dim = vec.shape[0]
        t = AnnoyIndex(dim, 'angular')
        t.load(_ANNOY_INDEX_FILE)
        _annoy_index = t
        _annoy_available = True
    except Exception as e:
        print("Failed to load Annoy index:", e)
        _annoy_index = None
        _annoy_meta = None
        _annoy_model = None
        _annoy_available = False

def annoy_retrieve_topk_with_neighbors(query, topk=TOPK):
    _ensure_annoy_loaded()
    if not _annoy_available:
        return []
    try:
        q_emb = _annoy_model.encode([query], convert_to_numpy=True)[0]
        ids, dists = _annoy_index.get_nns_by_vector(q_emb.tolist(), topk, include_distances=True)
        texts = [open(os.path.join(CHUNK_FOLDER, m["filename"]), "r", encoding="utf-8").read() for m in _annoy_meta]
        results = []
        for idx, dist in zip(ids, dists):
            if idx < 0 or idx >= len(texts):
                continue
            neighbor_texts = []
            for n in (idx-1, idx, idx+1):
                if 0 <= n < len(texts):
                    neighbor_texts.append(texts[n])
            joined = "\n\n".join(neighbor_texts)
            score = float(1.0 / (1.0 + dist)) if dist >= 0 else 0.0
            results.append((idx, score, joined))
        return results
    except Exception as e:
        print("Annoy retrieve error:", e)
        return []

# -----------------------
# TF-IDF fallback (document-level)
# -----------------------
class SimpleRetriever:
    def __init__(self, docs):
        if not docs:
            raise ValueError("No documents provided to SimpleRetriever.")
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words='english').fit(self.docs)
        self.docvecs = self.vectorizer.transform(self.docs)

    def retrieve_topk(self, query, topk=TOPK):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.docvecs).flatten()
        idxs = sims.argsort()[::-1][:topk]
        results = [(idx, float(sims[idx]), self.docs[idx]) for idx in idxs]
        return results

# -----------------------
# Embeddings + hierarchical retrieval (doc prefilter -> chunk candidates)
# -----------------------
_embeddings = None
_meta = None
_sentence_model = None

def _ensure_embeddings_loaded():
    global _embeddings, _meta, _sentence_model
    if _embeddings is not None and _meta is not None and _sentence_model is not None:
        return True
    if not (os.path.exists(_EMB_FILE) and os.path.exists(_META_FILE)):
        return False
    try:
        _embeddings = np.load(_EMB_FILE)
        with open(_META_FILE, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(_EMBED_MODEL_NAME)
        _embeddings = _embeddings.astype("float32")
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        _embeddings = _embeddings / norms
        return True
    except Exception as e:
        print("Failed to load embeddings/meta for hierarchical retrieval:", e)
        _embeddings = None
        _meta = None
        _sentence_model = None
        return False

def hierarchical_retrieve(query, top_docs=3, topk_chunks=TOPK):
    """
    1) Run TF-IDF over whole documents to get top_docs filenames.
    2) Compute query embedding and search only chunks that belong to those docs.
    Returns list of (chunk_meta_idx, score, joined_chunk_text)
    """
    docs, filenames = load_docs()
    if not docs or not filenames:
        return []

    # doc-level TF-IDF
    retriever = SimpleRetriever(docs)
    doc_hits = retriever.retrieve_topk(query, topk=top_docs)
    top_doc_names = [filenames[h[0]] for h in doc_hits]

    ok = _ensure_embeddings_loaded()
    if not ok:
        return []  # fall back to FAISS/Annoy later

    # candidate chunk meta indices belonging to top docs
    candidate_indices = []
    for i, m in enumerate(_meta):
        chunk_fname = m.get("filename", "")
        base = chunk_fname.split("__chunk_")[0]
        for docname in top_doc_names:
            docbase = os.path.splitext(docname)[0]
            if chunk_fname.startswith(docbase) or base == docbase:
                candidate_indices.append(i)
                break

    if not candidate_indices:
        return []

    # embed query and compute cosine similarities with candidates
    q_emb = _sentence_model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    cand_embs = _embeddings[candidate_indices]
    sims = (cand_embs @ q_emb.T).flatten()

    order = np.argsort(sims)[::-1][:topk_chunks]
    results = []
    for rpos in order:
        cand_idx = candidate_indices[int(rpos)]
        sc = float(sims[int(rpos)])
        # include neighbor chunks for context
        neighbor_texts = []
        for n in (cand_idx-1, cand_idx, cand_idx+1):
            if 0 <= n < len(_meta):
                fname = _meta[n].get("filename")
                path = os.path.join(CHUNK_FOLDER, fname)
                try:
                    with open(path, "r", encoding='utf-8', errors='ignore') as fh:
                        neighbor_texts.append(fh.read())
                except Exception:
                    continue
        joined = "\n\n".join(neighbor_texts) if neighbor_texts else ""
        results.append((cand_idx, sc, joined))
    return results

# -----------------------
# LLM: FLAN-T5-small (loaded lazily)
# -----------------------
_tokenizer = None
_model = None

def _ensure_model_loaded():
    global _tokenizer, _model
    if not _transformers_available:
        raise RuntimeError("Transformers library isn't available. Install 'transformers' and 'torch' in the venv.")
    if _tokenizer is None or _model is None:
        print("Loading FLAN-T5-small model (this may take a few minutes the first time)...")
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        print("Model loaded.")

def simple_llm_answer(query, top_snippet):
    """
    Improved prompt + retry logic to avoid instruction-echoing and produce longer,
    factual answers. `top_snippet` should be the combined labeled excerpts string.
    """
    if not _transformers_available:
        years = re.findall(r"\b\d{1,4}\b", top_snippet)
        if "statute of limitations" in (query.lower() + top_snippet.lower()) and years:
            return f"The document mentions a statute of limitations of {years[0]} years."
        return (top_snippet[:300] + "...")

    _ensure_model_loaded()

    MAX_SNIPPET_CHARS = 1600
    snippet = top_snippet
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[-MAX_SNIPPET_CHARS:]

    primary_prompt = (
        "You are a careful legal assistant. Use ONLY the DOCUMENT EXCERPTS below to answer the question. "
        "If the excerpts do not contain the information needed, reply exactly: I don't know — check the documents.\n\n"
        "DOCUMENT EXCERPTS:\n"
        f"{snippet}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "Answer below (start immediately; do not restate the question or excerpts):\n"
    )

    inputs = _tokenizer(primary_prompt, return_tensors="pt", truncation=True, max_length=512)
    gen = _model.generate(
        **inputs,
        max_new_tokens=350,
        num_beams=6,
        do_sample=False,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=3
    )
    ans = _tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    instruction_echo_signals = [
        "keep the answer factual",
        "do not hallucinate",
        "do not repeat the instructions",
        "check the documents"
    ]
    lower_ans = ans.lower()
    echoed = any(sig in lower_ans for sig in instruction_echo_signals)

    if echoed or len(ans.split()) < 6:
        retry_prompt = (
            "Answer the question below using ONLY the following excerpts. "
            "Do not repeat the excerpts. If the answer is not present, reply: I don't know — check the documents.\n\n"
            f"EXCERPTS:\n{snippet}\n\nQUESTION:\n{query}\n\nAnswer:"
        )
        inputs2 = _tokenizer(retry_prompt, return_tensors="pt", truncation=True, max_length=512)
        gen2 = _model.generate(
            **inputs2,
            max_new_tokens=350,
            num_beams=6,
            do_sample=False,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        ans2 = _tokenizer.decode(gen2[0], skip_special_tokens=True).strip()
        if len(ans2.split()) < 4 or any(sig in ans2.lower() for sig in instruction_echo_signals):
            return "I don't know — check the documents."
        return ans2

    return ans

# -----------------------
# TTS
# -----------------------
def speak(text):
    print("\nSPEAK:", text[:800], "\n")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS failed:", e)

# -----------------------
# Document loader
# -----------------------
def load_docs(folder=DATA_FOLDER):
    docs = []
    filenames = []
    if not os.path.exists(folder):
        print(f"Data folder '{folder}' not found.")
        return docs, filenames
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append(f.read())
                    filenames.append(fname)
            except Exception:
                continue
    return docs, filenames

# -----------------------
# Main
# -----------------------
def main():
    print("Starting pipeline: STT -> Hierarchical Retriever -> LLM -> TTS")

    # 1) STT (try mic first)
    query = None
    try:
        query = stt_transcribe(duration=6)
    except Exception as e:
        print("STT raised an exception (will prompt typed input):", e)
        query = None

    if not query:
        query = input("Type your query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return

    print("Query received:", query)

    # 2) Retrieval: try hierarchical first (doc TF-IDF -> chunk search), else FAISS/Annoy/TF-IDF fallback
    combined_snippet = ""
    retrieved_source = None
    retrieved_id_or_name = None
    retrieved_score = None

    hier_hits = hierarchical_retrieve(query, top_docs=3, topk_chunks=TOPK)
    if hier_hits:
        retrieved_source = "hierarchical"
        parts = []
        for i, (idx, sc, snip_joined) in enumerate(hier_hits, start=1):
            parts.append(f"=== Excerpt {i} (chunk_meta_idx={idx}, score={sc:.3f}) ===\n{snip_joined}\n")
        combined_snippet = "\n".join(parts)
        retrieved_id_or_name = hier_hits[0][0]
        retrieved_score = float(hier_hits[0][1])
        print(f"Retrieved (Hierarchical) top {len(hier_hits)} chunks. Top meta idx: {retrieved_id_or_name} (score={retrieved_score:.3f})")
    else:
        faiss_hits = faiss_retrieve_topk_with_neighbors(query, topk=TOPK)
        if faiss_hits:
            retrieved_source = "faiss"
            parts = []
            for i, (idx, sc, snip_joined) in enumerate(faiss_hits, start=1):
                parts.append(f"=== Excerpt {i} (chunk_id={idx}, score={sc:.3f}) ===\n{snip_joined}\n")
            combined_snippet = "\n".join(parts)
            retrieved_id_or_name = faiss_hits[0][0]
            retrieved_score = float(faiss_hits[0][1])
            print(f"Retrieved (FAISS) top {len(faiss_hits)} chunks (with neighbors). Top id: {retrieved_id_or_name} (score={retrieved_score:.3f})")
        else:
            annoy_hits = annoy_retrieve_topk_with_neighbors(query, topk=TOPK)
            if annoy_hits:
                retrieved_source = "annoy"
                parts = []
                for i, (idx, sc, snip_joined) in enumerate(annoy_hits, start=1):
                    parts.append(f"=== Excerpt {i} (chunk_id={idx}, score={sc:.3f}) ===\n{snip_joined}\n")
                combined_snippet = "\n".join(parts)
                retrieved_id_or_name = annoy_hits[0][0]
                retrieved_score = float(annoy_hits[0][1])
                print(f"Retrieved (Annoy) top {len(annoy_hits)} chunks (with neighbors). Top id: {retrieved_id_or_name} (score={retrieved_score:.3f})")
            else:
                docs, filenames = load_docs()
                if not docs:
                    print("No documents present in the data folder. Put .txt files in the folder and retry.")
                    return
                retriever = SimpleRetriever(docs)
                results = retriever.retrieve_topk(query, topk=TOPK)
                parts = []
                for i, (idx, sc, doctext) in enumerate(results, start=1):
                    excerpt_text = doctext[:2500].strip()
                    parts.append(f"=== Document {i} (file={filenames[idx]}, score={sc:.3f}) ===\n{excerpt_text}\n")
                combined_snippet = "\n".join(parts)
                retrieved_source = "tfidf"
                retrieved_id_or_name = filenames[results[0][0]]
                retrieved_score = float(results[0][1])
                print(f"Retrieved (TF-IDF) doc: {retrieved_id_or_name} (score={retrieved_score:.3f})")

    # 3) Generate answer via LLM using the combined_snippet
    try:
        answer_text = simple_llm_answer(query, combined_snippet)
    except Exception as e:
        print("LLM generation failed:", e)
        answer_text = (combined_snippet[:400] + "...")

    # 4) Log session (append)
    try:
        log_entry = {
            "ts": datetime.datetime.now().isoformat(),
            "query": query,
            "source": retrieved_source,
            "id_or_name": str(retrieved_id_or_name),
            "score": float(retrieved_score) if retrieved_score is not None else None,
            "answer": answer_text
        }
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 5) Speak the answer
    print("Answer:", answer_text)
    speak(answer_text)

# replace existing run_query(...) with this block in main_pipeline.py

EXIT_PHRASES = {"exit", "quit", "stop", "goodbye", "bye", "cancel", "stop listening", "stop now", "shutdown"}

def run_query(query_text):
    """
    Flask-friendly query function.
    Returns (answer_text, snippets, source) OR the special exit token with stop flag.
    """
    try:
        q = (query_text or "").strip()
        if not q:
            return "Please say something or type a question.", [], "empty"

        qlow = q.lower()

        # exact match or full phrase match check for exit
        if qlow in EXIT_PHRASES or any(phr in qlow for phr in EXIT_PHRASES):
            # return a clear machine-readable stop token + source control
            return "__EXIT__", [], "control"

        # normal retrieval flow (TF-IDF fallback baseline)
        docs, filenames = load_docs()
        if not docs:
            return "No documents found. Put .txt files into the 'data' folder and restart the server.", [], "no-data"

        retriever = SimpleRetriever(docs)
        results = retriever.retrieve_topk(query_text, topk=TOPK)
        if not results:
            return "I couldn't find anything relevant in the documents.", [], "tfidf-empty"

        idx, score, snippet = results[0]
        answer_text = simple_llm_answer(query_text, snippet)
        snippets = [{"text": snippet[:500], "meta": filenames[idx]}]
        return answer_text, snippets, "tfidf"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "An internal error occurred processing the query. Check server logs.", [], "error"



if __name__ == "__main__":
    main()

