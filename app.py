from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import main_pipeline
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
CORS(app)

# Homepage serving the index.html in templates/
@app.route('/')
def home():
    return render_template('index.html')

# keep your existing API routes (unchanged behavior)
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query", "")
        if not query_text.strip():
            return jsonify({"error": "Empty query"}), 400

        # call your main pipeline
        answer_text, snippets, source = main_pipeline.run_query(query_text)

        return jsonify({
            "answer": answer_text,
            "snippets": snippets,
            "source": source
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Lexi backend running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
