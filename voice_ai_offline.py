# voice_ai_offline.py
import os, json, wave, tempfile, soundfile as sf, gradio as gr
from vosk import Model, KaldiRecognizer
import pyttsx3

# ----- Setup paths -----
VOSK_PATH = "vosk-model"
if not os.path.exists(VOSK_PATH):
    raise RuntimeError("Download vosk-model-small-en-us-0.15 and unzip to ./vosk-model")

# ----- Load Vosk -----
model = Model(VOSK_PATH)
engine = pyttsx3.init()
engine.setProperty("rate", 175)

# ----- Model placeholder (use your real model later) -----
try:
    from rag_generate import ask_ollama
except Exception:
    def ask_ollama(prompt): return "Placeholder reply — model not connected."

# ----- Speech-to-text -----
def transcribe(audio_path):
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            parts.append(res.get("text", ""))
    res = json.loads(rec.FinalResult())
    parts.append(res.get("text", ""))
    return " ".join([p for p in parts if p]).strip()

# ----- Handle user audio -----
def handle(audio):
    if not audio:
        return "No audio", "", None
    # Gradio can return tuple (sr, np.array)
    if isinstance(audio, (tuple, list)):
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data, sr)
        path = tmp.name
    else:
        path = audio

    text = transcribe(path)
    if not text:
        return "(couldn't transcribe)", "", None

    prompt = f"User said: {text}\nAnswer concisely:"
    reply = ask_ollama(prompt)

    # speak
    engine.say(reply)
    engine.runAndWait()

    return text, reply, None

# ----- Gradio UI -----
with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Offline Voice Assistant — Vosk + pyttsx3")
    mic = gr.Audio(type="filepath", label="Record your question")
    out_text = gr.Textbox(label="Transcribed text")
    out_reply = gr.Textbox(label="Model reply")
    btn = gr.Button("Ask")
    btn.click(handle, inputs=[mic], outputs=[out_text, out_reply, gr.Audio(visible=False)])

demo.launch(server_port=7860)
