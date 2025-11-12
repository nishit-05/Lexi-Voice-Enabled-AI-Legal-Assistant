# stt_vosk.py
# Requires: vosk, sounddevice
# If you don't have a mic or Vosk model, use the fallback: type input when asked.

import queue, sys, json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pathlib import Path

MODEL_PATH = "model"  # change if your vosk model folder is named differently
SAMPLE_RATE = 16000
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print("Vosk status:", status, file=sys.stderr)
    q.put(bytes(indata))

def transcribe(duration=6, fallback_text=None):
    """
    Attempt to transcribe from mic using Vosk for `duration` seconds.
    If model path not found or any error occurs, will return fallback_text or prompt user to type.
    """
    if fallback_text:
        return fallback_text

    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        print(f"Vosk model folder '{MODEL_PATH}' not found. Falling back to typed input.")
        return input("Type your query: ")

    try:
        model = Model(str(model_dir))
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            print(f"Speak now for ~{duration} seconds...")
            import time as t
            t0 = t.time()
            full = ""
            while t.time() - t0 < duration:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    if text:
                        full += " " + text
            final = json.loads(rec.FinalResult())
            full += " " + final.get("text", "")
            return full.strip() or input("I couldn't hear you. Type your query: ")
    except Exception as e:
        print("Vosk transcribe error:", e)
        return input("Type your query: ")
