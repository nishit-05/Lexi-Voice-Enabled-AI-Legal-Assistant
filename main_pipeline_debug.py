# main_pipeline_debug.py
import os, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3

DATA_FOLDER = "sample_data"

def load_docs(folder=DATA_FOLDER):
    docs = []; filenames = []
    if not os.path.exists(folder):
        print("Data folder not found:", folder); return docs, filenames
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            print(" - found", path)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read()); filenames.append(fname)
    return docs, filenames

def speak(text):
    print("SPEAK:", text[:300])
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text); engine.runAndWait()
        print("TTS completed.")
    except Exception as e:
        print("TTS failed:", e)

def main():
    print("Verbose pipeline — forcing typed input.")
    typed = input("Type your query (force input): ").strip()
    if not typed:
        print("No input. Exiting."); return
    query = typed
    docs, filenames = load_docs()
    if not docs:
        print("No documents found in", DATA_FOLDER); return
    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    docvecs = vectorizer.fit_transform(docs)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, docvecs).flatten()
    best = sims.argmax()
    print(f"Best doc: {filenames[best]}, score={sims[best]:.3f}")
    snippet = docs[best][:800]
    answer = "I found this in the documents: " + snippet
    speak(answer)

if __name__ == "__main__":
    main()

