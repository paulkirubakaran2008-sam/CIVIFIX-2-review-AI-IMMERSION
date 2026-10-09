"""
Module 3 + 5 - Web application (Flask)  (CivicFix AI, Review 2)

Run:   python train_category.py && python train_severity.py   (first time only)
       python app.py      then open http://127.0.0.1:5000
Routes:  GET  /        -> complaint form
         POST /predict -> JSON {"complaint": "..."}  full analysis: category, severity, priority P1-P4,
                          SLA, department, safety rules, explanation, low-confidence flag
         POST /batch   -> JSON {"complaints": ["...", "..."]}  analyses every complaint (max 50)
                          and returns them ranked, most urgent first
"""
from flask import Flask, jsonify, render_template, request
from engine import analyse, load_models

app = Flask(__name__)
cat_model, sev_model = load_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = str(data.get("complaint", "")).strip()
    if len(text) < 3:
        return jsonify({"error": "Please enter a complaint (at least 3 characters)."}), 400
    return jsonify(analyse(text[:600], cat_model, sev_model))


@app.route("/batch", methods=["POST"])
def batch_route():
    data = request.get_json(silent=True) or {}
    items = [str(c).strip()[:600] for c in data.get("complaints", []) if len(str(c).strip()) >= 3][:50]
    if not items:
        return jsonify({"error": "Send a list of complaints in 'complaints'."}), 400
    results = [analyse(t, cat_model, sev_model) for t in items]
    results.sort(key=lambda r: -r["priority_score"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    app.run(debug=True)
