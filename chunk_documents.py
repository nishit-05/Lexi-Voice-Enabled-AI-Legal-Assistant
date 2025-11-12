# chunk_documents.py
# Splits .txt files in DATA_FOLDER into smaller chunk files inside 'chunks/'.

import os
import textwrap

DATA_FOLDER = "data"   # change if your folder differs
OUT_FOLDER = "chunks"
CHUNK_SIZE = 400  # approx characters per chunk (adjustable)
OVERLAP = 80      # overlap characters between chunks

os.makedirs(OUT_FOLDER, exist_ok=True)

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    start = 0
    chunks = []
    n = len(text)
    while start < n:
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

count = 0
for fname in sorted(os.listdir(DATA_FOLDER)):
    if not fname.lower().endswith(".txt"):
        continue
    path = os.path.join(DATA_FOLDER, fname)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r\n", "\n")
    parts = chunk_text(text)
    base = os.path.splitext(fname)[0]
    for i, p in enumerate(parts):
        out_name = f"{base}__chunk_{i:03d}.txt"
        out_path = os.path.join(OUT_FOLDER, out_name)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(p)
        count += 1

print(f"Created {count} chunk files in '{OUT_FOLDER}'")
