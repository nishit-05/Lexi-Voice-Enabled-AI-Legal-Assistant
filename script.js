// Final script: overlay placed in .hero-right, only hero-left blurs, no snippet auto-append
document.addEventListener("DOMContentLoaded", () => {
  const API_URL = "/api/query";
  const TRANSCRIBE_URL = "/transcribe_audio";
  const chatbox = document.getElementById("chatbox");
  const input = document.getElementById("userInput");
  const micBtn = document.getElementById("micBtn");
  const sendBtn = document.getElementById("sendBtn");
  const statusBadge = document.getElementById("statusBadge");

  // Ensure overlay exists inside .hero-right (beside the chat-card)
  function ensureListeningOverlay() {
    const heroRight = document.querySelector(".hero-right");
    if (!heroRight) return null;
    let overlay = heroRight.querySelector(".listening-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "listening-overlay";
      const stage = document.createElement("div"); stage.className = "listening-stage";
      const r1 = document.createElement("div"); r1.className = "ring r1";
      const r2 = document.createElement("div"); r2.className = "ring r2";
      const r3 = document.createElement("div"); r3.className = "ring r3";
      const core = document.createElement("div"); core.className = "core";
      core.innerHTML = ""; // optional mic SVG can be inserted
      const wave = document.createElement("div"); wave.className = "wave";
      for (let i=0;i<5;i++){ const bar = document.createElement("div"); bar.className = "bar"; wave.appendChild(bar); }
      const label = document.createElement("div"); label.className = "listening-label"; label.textContent = "Listening...";
      stage.appendChild(r1); stage.appendChild(r2); stage.appendChild(r3); stage.appendChild(wave); stage.appendChild(core);
      overlay.appendChild(stage); overlay.appendChild(label);
      heroRight.appendChild(overlay);
    }
    return overlay;
  }

  const overlayEl = ensureListeningOverlay();
  const log = (...args) => console.log("[Lexi]", ...args);

  // status helpers
  function showStatus(text) {
    if (!statusBadge) return;
    statusBadge.style.display = "inline-flex";
    statusBadge.textContent = text;
    const dot = document.createElement("span"); dot.className = "dot";
    statusBadge.appendChild(dot);
  }
  function hideStatus() {
    if (!statusBadge) return;
    statusBadge.style.display = "none";
    statusBadge.textContent = "";
  }

  // message appends (NO automatic snippet printing — removed to avoid extra block)
  function appendUser(text) {
    const el = document.createElement("div"); el.className = "user-msg"; el.textContent = text;
    chatbox.appendChild(el); chatbox.scrollTop = chatbox.scrollHeight;
  }
  function appendLexi(text) {
    const el = document.createElement("div"); el.className = "lexi-msg"; el.textContent = text;
    chatbox.appendChild(el); chatbox.scrollTop = chatbox.scrollHeight;
  }

  function createTypingBubble(){
    const wrap = document.createElement("div"); wrap.className = "typing-bubble";
    const d1 = document.createElement("div"); d1.className = "dot";
    const d2 = document.createElement("div"); d2.className = "dot";
    const d3 = document.createElement("div"); d3.className = "dot";
    wrap.appendChild(d1); wrap.appendChild(d2); wrap.appendChild(d3);
    wrap.style.marginRight = "auto";
    return wrap;
  }

  function speak(text) {
    if (!("speechSynthesis" in window) || !text) return;
    try { window.speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text); u.rate = 1; window.speechSynthesis.speak(u); }
    catch (e){ console.warn("TTS error", e); }
  }

  async function sendMessage(){
    const text = (input.value || "").trim();
    if (!text) return;
    appendUser(text);
    input.value = "";

    showStatus("Processing…");
    const typing = createTypingBubble();
    chatbox.appendChild(typing);
    chatbox.scrollTop = chatbox.scrollHeight;

    try {
      const resp = await fetch(API_URL, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ query: text }) });
      const data = await resp.json();

      typing.remove();
      hideStatus();

      const ans = data.answer || data.reply || "I couldn't process that.";
      if (ans === "__EXIT__" || data.stop === true) {
        window.speechSynthesis.cancel();
        appendLexi("Stopped listening.");
        return;
      }

      appendLexi(ans);
      speak(ans);

      // NO auto-printed snippets to avoid duplicate long blocks under answers.
      // If you want snippets shown, we can add a "Show source" button per message instead.

    } catch (err) {
      try{ typing.remove(); }catch(e){}
      hideStatus();
      appendLexi("Server error.");
      console.error("sendMessage error", err);
    }
  }

  if (sendBtn) sendBtn.addEventListener("click", sendMessage);
  if (input) input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });

  // Speech recognition + fallback
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let finalTranscript = "";
  let interimTranscript = "";
  const MAX_MS = 8000;
  let recognitionTimer = null;

  function showListeningOverlay() {
    if (!overlayEl) return;
    document.body.classList.add("overlay-active"); // triggers hero-left blur/dim (css)
    overlayEl.classList.add("show");
    showStatus("Listening…");
  }
  function hideListeningOverlay() {
    if (!overlayEl) return;
    overlayEl.classList.remove("show");
    document.body.classList.remove("overlay-active");
    hideStatus();
  }

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
      finalTranscript = ""; interimTranscript = "";
      showListeningOverlay();
      clearTimeout(recognitionTimer);
      recognitionTimer = setTimeout(()=>{ try{ recognition.stop(); }catch(e){} }, MAX_MS);
      log("Recognition started");
    };

    recognition.onresult = (e) => {
      interimTranscript = "";
      for (let i = e.resultIndex; i < e.results.length; ++i){
        const r = e.results[i];
        if (r.isFinal) finalTranscript += r[0].transcript;
        else interimTranscript += r[0].transcript;
      }
      input.value = (finalTranscript + interimTranscript).trim();
    };

    recognition.onerror = (e) => {
      console.warn("Recognition error:", e);
      hideListeningOverlay();
      try { recognition.stop(); } catch (err) {}
      recordAndTranscribe(); // fallback
    };

    recognition.onend = () => {
      hideListeningOverlay();
      clearTimeout(recognitionTimer);
      const captured = (finalTranscript + interimTranscript).trim();
      if (captured) { input.value = captured; sendMessage(); }
      log("Recognition ended");
    };
  }

  async function recordAndTranscribe(){
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { alert("Microphone not supported."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size) chunks.push(ev.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
        showStatus("Transcribing…");
        try {
          const fd = new FormData(); fd.append("audio", blob, "recording.webm");
          const r = await fetch(TRANSCRIBE_URL, { method: "POST", body: fd });
          const j = await r.json();
          hideListeningOverlay();
          hideStatus();
          const text = j.text || j.transcription || "";
          if (text) { input.value = text; sendMessage(); } else { alert("No transcription returned."); }
        } catch (err) {
          hideListeningOverlay();
          hideStatus();
          console.error("Transcription upload failed", err);
          alert("Transcription failed. See console.");
        }
      };
      recorder.start();
      showListeningOverlay();
      showStatus("Recording…");
      setTimeout(()=>{ try{ recorder.stop(); }catch(e){} }, MAX_MS);
    } catch (err) {
      console.error("getUserMedia error", err);
      alert("Microphone permission denied or unavailable.");
    }
  }

  if (micBtn) micBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    try { await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (permErr) { alert("Microphone permission denied. Please enable mic access and try again."); return; }

    if (recognition) {
      try { finalTranscript = ""; interimTranscript = ""; recognition.start(); return; } catch (err) { console.warn("recognition.start failed:", err); }
    }
    recordAndTranscribe();
  });

  window.lexi_sendMessage = sendMessage;
  if (input) input.focus();
  log("Frontend loaded — final.");
});
