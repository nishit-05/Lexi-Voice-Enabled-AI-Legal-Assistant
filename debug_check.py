# debug_check.py
import sys, os, platform, importlib
print("=== DEBUG CHECK ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version.replace('\n',' '))
print("Current dir:", os.getcwd())
print("Files here:", os.listdir("."))
print("Virtualenv present:", ".venv" in os.listdir(".") or "venv" in os.listdir("."))
print("Platform:", platform.platform())

# Test imports
reqs = ["pyttsx3", "sklearn"]
for r in reqs:
    try:
        importlib.import_module(r)
        print(f"OK import: {r}")
    except Exception as e:
        print(f"IMPORT FAIL: {r} -> {e}")

# Quick TTS test
print("\nTrying TTS speak test...")
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.say("Debug check. If you hear this, text to speech works.")
    engine.runAndWait()
    print("TTS ran without exception.")
except Exception as e:
    print("TTS error:", e)

print("\nDEBUG CHECK DONE")
