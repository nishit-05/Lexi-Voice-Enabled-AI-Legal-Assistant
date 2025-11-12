import React, { useState, useRef, useEffect } from "react";

// Lexi UI — single-file React component (Tailwind CSS required in host app)
// Features:
// - Clean two-column layout (chat + RHS info panel)
// - Mic button (uses Web Speech API for STT if available)
// - Text input, send button
// - Displays transcript, retrieved excerpts, source badges
// - Plays TTS using Web Speech Synthesis (browser)
// - Settings panel for toggles (Local LLM / OpenAI placeholder)
// - Export session log button

export default function LexiUI({ apiEndpoint = "http://127.0.0.1:5000/api/query" }) {

  const [messages, setMessages] = useState([]); // {role: 'user'|'assistant', text, source?}
  const [input, setInput] = useState("");
  const [listening, setListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [snippets, setSnippets] = useState([]); // retrieved excerpts
  const [source, setSource] = useState(null); // retrieval source
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [useOpenAI, setUseOpenAI] = useState(false);
  const [modelName, setModelName] = useState("FLAN-T5-small (local)");
  const recognitionRef = useRef(null);
  const synthRef = useRef(typeof window !== "undefined" ? window.speechSynthesis : null);
  const messagesRef = useRef(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  function pushMessage(role, text, meta = {}) {
    const m = { id: Date.now() + Math.random(), role, text, ...meta };
    setMessages((s) => [...s, m]);
    return m;
  }

  function toggleListen() {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      alert("Speech recognition not supported in this browser. Please type your query.");
      return;
    }

    if (listening) {
      stopListening();
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recog = new SpeechRecognition();
    recog.lang = "en-IN";
    recog.interimResults = false;
    recog.maxAlternatives = 1;

    recog.onstart = () => setListening(true);
    recog.onerror = (e) => {
      console.warn("Speech error", e);
      setListening(false);
      try { recog.stop(); } catch {}
    };
    recog.onend = () => setListening(false);
    recog.onresult = (ev) => {
      const text = ev.results[0][0].transcript;
      setInput(text);
      pushMessage("user", text);
      handleSend(text);
    };

    recognitionRef.current = recog;
    recog.start();
  }

  function stopListening() {
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
      recognitionRef.current = null;
    }
    setListening(false);
  }

  function speakText(text) {
    if (!synthRef.current) return;
    try {
      synthRef.current.cancel();
      const ut = new SpeechSynthesisUtterance(text);
      ut.lang = "en-US";
      ut.rate = 1.0;
      synthRef.current.speak(ut);
    } catch (e) {
      console.warn("TTS failed", e);
    }
  }

  async function handleSend(forcedText) {
    const q = (forcedText !== undefined) ? forcedText : input.trim();
    if (!q) return;
    setInput("");
    setIsLoading(true);
    pushMessage("user", q);

    const placeholder = pushMessage("assistant", "Thinking...");

    try {
      const body = JSON.stringify({ query: q, useOpenAI: useOpenAI });
      const res = await fetch(apiEndpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body });
      if (!res.ok) throw new Error("Server error " + res.status);
      const data = await res.json();
      const answer = data.answer || "No answer returned.";
      const snips = data.snippets || [];
      const src = data.source || null;

      setMessages((cur) => cur.map(m => m.id === placeholder.id ? { ...m, text: answer } : m));
      setSnippets(snips);
      setSource(src);
      speakText(answer);

    } catch (e) {
      console.error("Query failed", e);
      setMessages((cur) => cur.map(m => m.id === placeholder.id ? { ...m, text: "Error: " + (e.message||e) } : m));
    } finally {
      setIsLoading(false);
    }
  }

  function downloadLog() {
    const data = JSON.stringify(messagesRef.current, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "lexi_session.json"; a.click();
    URL.revokeObjectURL(url);
  }

  function clearChat() { setMessages([]); setSnippets([]); setSource(null); }

  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-slate-50 p-6">
      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6">
        <header className="col-span-12 flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="rounded-full bg-gradient-to-r from-rose-400 to-indigo-500 w-12 h-12 flex items-center justify-center text-white text-xl font-bold">L</div>
            <div>
              <h1 className="text-2xl font-semibold">Lexi — Legal Assistant</h1>
              <p className="text-sm text-slate-500">Voice + Semantic search · Offline-first · Cite sources</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setSettingsOpen(s => !s)} className="px-3 py-1 rounded-md border">Settings</button>
            <button onClick={downloadLog} className="px-3 py-1 rounded-md bg-slate-800 text-white">Export Log</button>
          </div>
        </header>

        <main className="col-span-8 bg-white rounded-xl shadow p-4 flex flex-col" style={{ minHeight: '72vh' }}>
          <div className="flex-1 overflow-auto space-y-3 p-2" id="chat-window">
            {messages.map(m => (
              <div key={m.id} className={`max-w-[80%] ${m.role==='user'?'ml-auto text-right':'mr-auto text-left'}`}>
                <div className={`inline-block px-4 py-2 rounded-xl ${m.role==='user'?'bg-rose-100':'bg-slate-100'}`}>
                  <div className="text-sm whitespace-pre-wrap">{m.text}</div>
                </div>
                <div className="text-xs text-slate-400 mt-1">{m.role}{m.meta?` · ${m.meta}`:''}</div>
              </div>
            ))}
          </div>

          <div className="mt-3 flex items-center gap-3">
            <button onClick={toggleListen} className={`p-3 rounded-full ${listening? 'bg-rose-500 text-white':'bg-slate-100'}`} title="Use microphone">
              {listening ? '●' : '🎤'}
            </button>
            <input value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask Lexi anything about the law..." className="flex-1 px-4 py-3 rounded-xl border" onKeyDown={e=>{ if(e.key==='Enter') handleSend(); }} />
            <button onClick={()=>handleSend()} disabled={isLoading} className="px-4 py-2 rounded-xl bg-indigo-600 text-white">Send</button>
          </div>
        </main>

        <aside className="col-span-4 space-y-4">
          <div className="bg-white rounded-xl shadow p-4">
            <h3 className="font-semibold">Session</h3>
            <div className="mt-2 text-sm text-slate-600">Source: <span className="font-medium">{source||'—'}</span></div>
            <div className="mt-3">
              <button onClick={clearChat} className="px-3 py-1 rounded border">Clear</button>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow p-4">
            <h3 className="font-semibold">Retrieved Excerpts</h3>
            <div className="mt-2 space-y-3 max-h-[40vh] overflow-auto">
              {snippets.length===0 && <div className="text-sm text-slate-400">No excerpts yet — ask a question.</div>}
              {snippets.map((s, i) => (
                <div key={i} className="p-2 border rounded">
                  <div className="text-xs text-slate-500 mb-1">Excerpt {i+1} · {s.meta||''}</div>
                  <div className="text-sm whitespace-pre-wrap">{s.text}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow p-4">
            <h3 className="font-semibold">Settings</h3>
            <div className="mt-2 text-sm space-y-2">
              <label className="flex items-center justify-between">
                <span>Use OpenAI (cloud)</span>
                <input type="checkbox" checked={useOpenAI} onChange={e=>setUseOpenAI(e.target.checked)} />
              </label>
              <label className="flex items-center justify-between">
                <span>Model</span>
                <select value={modelName} onChange={e=>setModelName(e.target.value)} className="ml-2">
                  <option>FLAN-T5-small (local)</option>
                  <option>GPT4All (local)</option>
                  <option>OpenAI GPT-4o</option>
                </select>
              </label>
              <div className="text-xs text-slate-400 pt-2">Tip: Use the mic to ask aloud. Export the session for your report/demo.</div>
            </div>
          </div>

        </aside>
      </div>
    </div>
  );
}
